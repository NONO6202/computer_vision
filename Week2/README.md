# Week 2

---

## 1. 체크보드 기반 카메라 캘리브레이션 (Camera Calibration)

카메라 캘리브레이션은 3D 실제 좌표와 2D 이미지 좌표의 대응 관계를 이용하여 카메라 파라미터를 추정하는 과정입니다. 

<img width="1280" height="511" alt="image" src="https://github.com/user-attachments/assets/32614d24-6cc4-4a78-9561-873d6e155b9c" />

<img width="448" height="142" alt="image" src="https://github.com/user-attachments/assets/57a1ad0e-8565-4813-8108-8d9d089cf9c0" />

``` 01.Calibration.py
import cv2
import numpy as np
import glob

# 체크보드 내부 코너 개수
CHECKERBOARD = (9, 6)

# 체크보드 한 칸 실제 크기 (mm)
square_size = 25.0

# 코너 정밀화 조건
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 실제 좌표 생성
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# 저장할 좌표
objpoints = [] # 3D 실제 좌표
imgpoints = [] # 2D 이미지 좌표

images = glob.glob("Week2/images/calibration_images/left*.jpg")

img_size = None

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 코너 검출은 그레이스케일에서 수행
    if img_size is None:
        img_size = gray.shape[::-1] # (width, height)

    # 이미지에서 체스보드 코너 위치 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # 코너를 모두 성공적으로 찾은 경우에만 리스트에 추가
    if ret == True:
        objpoints.append(objp) # 실제 좌표 추가
        # 찾아낸 코너 픽셀 좌표를 서브픽셀 단위로 더욱 정교하게 다듬음
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2) # 정밀화된 2D 이미지 좌표 추가

# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------
# 3D 실제 좌표와 2D 이미지 좌표 매칭을 통해 카메라 파라미터 계산
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
# 3D 실제 좌표의 점이 2D 이미지 plane으로 맵핑되는 원리는 Homogeneous Coordinates(동차 좌표계)를 사용한 행렬 곱으로 표현됩니다.
# 수식: s[x, y, 1]^T = K * [R|t] * [X, Y, Z, 1]^T
# (회전 R, 평행이동 t)

print("카메라 내부 파라미터 행렬 K:")
print(K) # 카메라 내부 파라미터 (focal length, principal point)

print("\왜곡 계수:")
print(dist) # 왜곡 계수 (방사 왜곡, 접선 왜곡)
print(f"- 방사 왜곡: {dist[0][0:3]}") # 렌즈 중심에서 멀어질 수록 직선이 휘어지는 현상
print(f"- 접선 왜곡: {dist[0][3:5]}") # 렌즈와 이미지 센서가 완전히 평행하지 않을 때 발생

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------
if len(images) > 0:
    sample_img = cv2.imread(images[0])
    
    # 계산된 K와 dist를 이용하여 원본 이미지의 렌즈 왜곡을 펴줌(보정) 
    undistorted_img = cv2.undistort(sample_img, K, dist, None, K)
    
    # 결과를 화면에 출력하여 비교
    cv2.imshow('Original', sample_img)
    cv2.imshow('Undistorted', undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

```

**주요 목적 및 원리**
- **Homogeneous Coordinates (동차 좌표계)**: 비선형적인 투영 연산을 행렬 곱셈 형태의 선형 연산으로 변환하기 위해 차원을 하나 추가하는 기법입니다 ($[x, y] \rightarrow [x, y, 1]$).
- **Euclidean Transformation (Rigid-body)**: 3D 월드 좌표계를 카메라 좌표계로 일치시키는 회전 행렬($R$)과 평행 이동 벡터($t$)의 결합입니다.
- **Projective Transformation (투영 변환)**: 3차원 공간의 점을 2차원 평면에 맵핑하는 행렬($K$) 연산으로 원근감을 표현합니다.

**핵심 코드 분석**
| 함수명 | 역할 및 설명 |
| :--- | :--- |
| `cv2.findChessboardCorners()` | 체크보드 이미지에서 2D 코너 위치(이미지 좌표)를 검출합니다. |
| `cv2.calibrateCamera()` | 검출된 코너와 실제 3D 좌표를 매칭하여 카메라 내부 파라미터 행렬(K)과 왜곡 계수를 계산합니다. |
| `cv2.undistort()` | 계산된 K 행렬과 왜곡 계수를 바탕으로 원본 이미지의 렌즈 왜곡을 평면으로 펴서 시각화합니다. |

> **AI의 추가 설명:** 코너 검출 후 `cv2.cornerSubPix()`를 추가로 사용하여 픽셀 단위보다 더 정밀하게 위치를 보정한 점은 캘리브레이션의 오차를 획기적으로 줄일수 있다고 했습니다.

---

## 2. 이미지 Rotation & Transformation

한 장의 2D 이미지에 회전(Rotation), 크기 조절(Scaling), 평행이동(Translation)을 동시에 적용하는 아핀 변환(Affine Transformation) 실습입니다.

<img width="2374" height="821" alt="image" src="https://github.com/user-attachments/assets/b3978531-d243-4f92-b918-c6990e5d7467" />

```02.RotationTransformation.py
import cv2
import numpy as np

img = cv2.imread('Week2/images/rose.png') 
h, w = img.shape[:2] # 이미지의 높이(Height)와 너비(Width) 추출

# 1. 회전의 기준이 될 이미지 중심 좌표 계산
center = (w / 2, h / 2)

# 2. 회전 및 크기 조절 변환 행렬 생성
# cv2.getRotationMatrix2D(중심좌표, 회전각도(양수는 반시계), 스케일)
# 요구사항: +30도 회전, 크기 0.8 조절
M = cv2.getRotationMatrix2D(center, 30, 0.8)
# cv2.getRotationMatrix2D는 Rotation과 Scaling이 결합된 Similarity 행렬(2x3)을 반환합니다.

# 3. 평행이동(Translation) 결합 
# 아핀 변환 행렬 M (2x3 행렬)의 마지막 열(index 2)은 각각 x, y축의 평행이동량을 의미함.
# 기존 회전 행렬의 평행이동 성분에 요구사항인 x축 +80, y축 -40을 더해줌 
M[0, 2] += 80  # X축 평행이동
M[1, 2] -= 40  # Y축 평행이동
# Affine 변환은 선형 변환에 평행이동을 더한 변환으로,
# 원본 이미지의 평행성을 결과 이미지에서도 그대로 유지합니다.
# 2x3 행렬 M의 마지막 열(M[0,2], M[1,2])은 각각 x축과 y축의 이동 벡터(t)를 나타냅니다.


# 4. 변환 행렬을 이미지에 최종 적용
# cv2.warpAffine(원본이미지, 2x3변환행렬, 출력결과사이즈)
transformed_img = cv2.warpAffine(img, M, (w, h))
# Perspective 변환과 달리 Affine은 직사각형이 평행사변형으로 변할 수는 있어도 평행선은 보존됩니다.

# 결과 출력
cv2.imshow('Original', img)
cv2.imshow('Rotate + Scale + Translate', transformed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

**주요 목적 및 원리**
* 이미지의 중심을 기준으로 지정된 각도(+30도)만큼 회전시킵니다.
* 회전과 동시에 전체 이미지의 크기를 0.8배로 축소합니다.
* X축 방향으로 +80px, Y축 방향으로 -40px만큼 평행이동을 수행합니다.

**핵심 코드 분석**
| 변수 및 함수명 | 역할 및 설명 |
| :--- | :--- |
| `cv2.getRotationMatrix2D()` | 중심 좌표, 회전 각도, 스케일 비율을 입력받아 2x3 크기의 변환 행렬을 생성합니다. |
| `M[0, 2]`, `M[1, 2]` 조작 | 회전 행렬의 마지막 열 값을 직접 조정하는 방식으로 평행이동 성분을 반영합니다. |
| `cv2.warpAffine()` | 최종적으로 완성된 2x3 아핀 변환 행렬을 원본 이미지에 적용하여 결과 이미지를 산출합니다. |

---

## 3. Stereo Disparity 기반 Depth 추정

같은 장면을 왼쪽 카메라와 오른쪽 카메라에서 촬영한 두 장의 스테레오 이미지를 이용하여 물체의 깊이(Depth)를 추정합니다.

<img width="903" height="410" alt="image" src="https://github.com/user-attachments/assets/826d73c9-eee8-4fc1-b6f8-1fec3d66684f" />

<img width="297" height="266" alt="image" src="https://github.com/user-attachments/assets/55dbcaac-6dd5-4636-a8ea-8bc3f64478d5" />

```03.Depth.py
import cv2
import numpy as np
from pathlib import Path

# 출력 폴더 생성
output_dir = Path("Week2/outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# 좌/우 이미지 불러오기
left_color = cv2.imread("Week2/images/left.png")
right_color = cv2.imread("Week2/images/right.png")

if left_color is None or right_color is None:
    raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다.")


# 카메라 파라미터
f = 700.0 # Focal Length (초점 거리)
B = 0.12 # Baseline (두 카메라 사이의 물리적 거리)

# ROI 설정 (x, y, w, h)
rois = {
    "Painting": (55, 50, 130, 110),
    "Frog": (90, 265, 230, 95),
    "Teddy": (310, 35, 115, 90)
}

# 그레이스케일 변환
left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 1. Disparity 계산
# 양안 시차 원리 적용:
# 동일한 3D 공간상의 점이 좌/우 카메라에 맺힐 때 발생하는 가로 픽셀 위치의 차이를 구합니다.
# -----------------------------
# -----------------------------
# cv2.StereoBM_create()를 사용하여 Disparity map을 계산합니다.
# numDisparities는 16의 배수여야 합니다.
stereo = cv2.StereoBM_create(numDisparities=64)

# 계산된 disparity는 정수형이며 16배 스케일링되어 반환됩니다.
disparity_16S = stereo.compute(left_gray, right_gray)

# 정확한 Depth 계산을 위해 실수형으로 변환 후 16으로 나누어 실제 Disparity 값을 구합니다.
disparity = disparity_16S.astype(np.float32) / 16.0

# -----------------------------
# 2. Depth 계산
# Z = fB / d
# Z(깊이)는 두 카메라 간의 거리(B)와 초점 거리(f)의 곱을 양안 시차(d)로 나눈 값입니다.
# 즉, 물체가 카메라에 가까울수록 시차(Disparity)는 커지고, 멀어질수록 시차는 작아지는 반비례 성질(Similar Triangles)을 이용합니다.
# -----------------------------
# Depth 맵을 저장할 배열을 0으로 초기화합니다.
depth_map = np.zeros_like(disparity, dtype=np.float32)

# Disparity가 0보다 큰 픽셀만 추출하기 위한 마스크를 생성합니다. (0 이하는 유효하지 않은 값)
valid_mask = disparity > 0

# 유효한 픽셀에 대해서만 Z = (f * B) / d 공식을 적용해 깊이를 계산합니다.
depth_map[valid_mask] = (f * B) / disparity[valid_mask]

# -----------------------------
# 3. ROI별 평균 disparity / depth 계산
# -----------------------------
results = {}

for name, (x, y, w, h) in rois.items():
    # 전체 이미지에서 각 ROI 영역에 해당하는 부분만 슬라이싱합니다.
    roi_disp = disparity[y:y+h, x:x+w]
    roi_depth = depth_map[y:y+h, x:x+w]
    roi_mask = valid_mask[y:y+h, x:x+w]
    
    # 해당 ROI 내에 유효한 값이 존재하는지 확인합니다.
    if np.any(roi_mask):
        # 유효한 픽셀들의 평균 disparity와 depth를 계산합니다.
        avg_disp = np.mean(roi_disp[roi_mask])
        avg_depth = np.mean(roi_depth[roi_mask])
    else:
        avg_disp = 0
        avg_depth = 0
        
    results[name] = {"avg_disp": avg_disp, "avg_depth": avg_depth}
    
# -----------------------------
# 4. 결과 출력
# -----------------------------
print("=" * 40)
print("ROI별 평균 Disparity 및 Depth 분석 결과")
print("=" * 40)

min_depth = float('inf')
max_depth = 0
closest_roi = ""
farthest_roi = ""

for name, metrics in results.items():
    print(f"[{name}]")
    print(f" - 평균 Disparity : {metrics['avg_disp']:.2f}")
    print(f" - 평균 Depth     : {metrics['avg_depth']:.2f} m")
    
    # 가장 가까운 영역 탐색 (Depth 값이 가장 작은 영역)
    if 0 < metrics['avg_depth'] < min_depth:
        min_depth = metrics['avg_depth']
        closest_roi = name
        
    # 가장 먼 영역 탐색 (Depth 값이 가장 큰 영역)
    if metrics['avg_depth'] > max_depth:
        max_depth = metrics['avg_depth']
        farthest_roi = name

print("-" * 40)
print(f"가장 가까운 객체: {closest_roi} (Depth: {min_depth:.2f} m)")
print(f"가장 먼 객체   : {farthest_roi} (Depth: {max_depth:.2f} m)")
print("=" * 40)

# -----------------------------
# 5. disparity 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
disp_tmp = disparity.copy()
disp_tmp[disp_tmp <= 0] = np.nan

if np.all(np.isnan(disp_tmp)):
    raise ValueError("유효한 disparity 값이 없습니다.")

d_min = np.nanpercentile(disp_tmp, 5)
d_max = np.nanpercentile(disp_tmp, 95)

if d_max <= d_min:
    d_max = d_min + 1e-6

disp_scaled = (disp_tmp - d_min) / (d_max - d_min)
disp_scaled = np.clip(disp_scaled, 0, 1)

disp_vis = np.zeros_like(disparity, dtype=np.uint8)
valid_disp = ~np.isnan(disp_tmp)
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)

disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

# -----------------------------
# 6. depth 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
depth_vis = np.zeros_like(depth_map, dtype=np.uint8)

if np.any(valid_mask):
    depth_valid = depth_map[valid_mask]

    z_min = np.percentile(depth_valid, 5)
    z_max = np.percentile(depth_valid, 95)

    if z_max <= z_min:
        z_max = z_min + 1e-6

    depth_scaled = (depth_map - z_min) / (z_max - z_min)
    depth_scaled = np.clip(depth_scaled, 0, 1)

    # depth는 클수록 멀기 때문에 반전
    depth_scaled = 1.0 - depth_scaled
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)

depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

# -----------------------------
# 7. Left / Right 이미지에 ROI 표시
# -----------------------------
left_vis = left_color.copy()
right_vis = right_color.copy()

for name, (x, y, w, h) in rois.items():
    cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(left_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.rectangle(right_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(right_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# -----------------------------
# 8. 저장
# -----------------------------

cv2.imwrite(str(output_dir / "01_disparity_map.png"), disparity_color)
cv2.imwrite(str(output_dir / "02_depth_map.png"), depth_color)
cv2.imwrite(str(output_dir / "03_left_roi.png"), left_vis)
cv2.imwrite(str(output_dir / "04_right_roi.png"), right_vis)

# -----------------------------
# 9. 출력
# -----------------------------

# OpenCV 창을 통해 결과 이미지를 화면에 띄웁니다.
cv2.imshow("Disparity Map", disparity_color)
cv2.imshow("Left ROI", left_vis)
# 스테레오 카메라는 멀리 있는 물체일수록 거리 측정의 정밀도가 급격히 떨어지는 의도 파악을 위해 있는거 같음
# cv2.imshow("Depth Map", depth_color)

# 임의의 키를 누를 때까지 창을 대기시킵니다.
cv2.waitKey(0)
# 열려있는 모든 OpenCV 창을 닫습니다.
cv2.destroyAllWindows()
```

**주요 목적 및 원리**
* 왼쪽 이미지와 오른쪽 이미지에서 같은 물체의 픽셀 위치 차이를 Disparity라고 하며, 이 값이 클수록 카메라와 가까운 물체입니다.
* 카메라의 초점 거리(f)와 두 카메라 사이 거리(B)를 바탕으로 $Z=\frac{fB}{d}$ 공식을 적용하여 실제 거리 정보를 계산합니다.

**핵심 코드 분석**
| 함수명 및 수식 | 역할 및 설명 |
| :--- | :--- |
| `cv2.StereoBM_create()` | 두 그레이스케일 이미지를 비교하여 정수형 Disparity map을 16배 스케일로 계산하여 반환합니다. |
| 타입 변환 및 스케일링 | 정확한 실수 연산을 위해 반환된 값을 실수형(float)으로 변경 후 16으로 나누어 실제 Disparity를 구합니다. |
| `depth_map[valid_mask]` | 0 이하의 잘못된 Disparity 값을 마스킹(제외)한 후 $Z=\frac{fB}{d}$ 연산을 일괄 적용하여 안전하게 Depth를 도출합니다. |