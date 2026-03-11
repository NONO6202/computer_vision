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