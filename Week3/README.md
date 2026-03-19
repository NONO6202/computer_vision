# Week 3

---

## 1. 소벨 에지 검출 및 결과 시각화

- `edgeDetectionImage.jpg` 이미지를 그레이스케일로 변환
- `cv.Sobel()`로 x축, y축 방향의 에지를 각각 계산
- `cv.magnitude()`로 전체 에지 강도를 구한 뒤 `Matplotlib`으로 시각화

**이미지**

```01.SobelEdge.py
import cv2 as cv  
import matplotlib.pyplot as plt  
from pathlib import Path  

week3_dir = Path(__file__).resolve().parent 
image_path = week3_dir / "edgeDetectionImage.jpg"

# 요구사항: cv.imread()를 사용하여 이미지를 불러옴
img = cv.imread(str(image_path))  # 입력 이미지를 BGR 색상 형식으로 읽어 온다.

# 요구사항: cv.cvtColor()를 사용하여 그레이스케일로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 컬러 이미지를 소벨 연산용 그레이스케일 이미지로 변환한다.
# 요구사항: cvSobel()을 사용하여 x축(cv.CV_64F, 1, 0)과 y축(cv.CV_64F, 0,1) 방향의 에지를 검출
# 힌트: cv.Sobel()의 ksize는 3 또는 5로 설정
grad_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)  # x축 방향 미분을 계산해 세로 방향 경계 변화를 강조한다.
grad_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)  # y축 방향 미분을 계산해 가로 방향 경계 변화를 강조한다.
# 요구사항: cv.magnitude()를 사용하여 에지 강도 계산
edge_magnitude = cv.magnitude(grad_x, grad_y)  # x축과 y축 기울기를 합쳐 전체 에지 강도를 계산한다.
# 힌트: cv.convertScaleAbs()를 사용하여 에지 강도 이미지를 uint8로 변환
edge_strength = cv.convertScaleAbs(edge_magnitude) 

rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Matplotlib에서 올바른 색으로 보이도록 BGR 이미지를 RGB로 변환한다.
# 요구사항: Matplotlib를 사용하여 원본 이미지와 에지 강도 이미지를 나란히 시각화
figure, axes = plt.subplots(1, 2, figsize=(12, 5)) 
axes[0].imshow(rgb_img)  # 첫 번째 축에 원본 이미지를 표시한다.
axes[0].set_title("Original Image")
# 힌트: plt.imshow()에서 cmap='gray'를 사용하여 흑백으로 시각화
axes[1].imshow(edge_strength, cmap="gray")
axes[1].set_title("Sobel Strength")
figure.tight_layout()  # 제목과 이미지가 겹치지 않도록 레이아웃을 정리한다.

plt.show() 
plt.close(figure) 
```

- **Sobel 미분 분리**

`cv.Sobel(gray, cv.CV_64F, 1, 0)`과 `cv.Sobel(gray, cv.CV_64F, 0, 1)`로 x축, y축 방향 경계를 따로 계산했습니다.

- **에지 강도 계산**

`cv.magnitude()`를 사용해 두 방향의 기울기를 하나의 에지 강도 영상으로 합쳤고, `cv.convertScaleAbs()`로 화면 출력용 `uint8` 자료형으로 변환했습니다.

- **Matplotlib 시각화**

PDF 요구사항에 맞춰 원본 이미지와 에지 강도 이미지를 한 figure 안에 나란히 배치했고, 결과는 `outputs/01_sobel_result.png`에도 저장되도록 구성했습니다.

---

## 2. 캐니 에지 및 허프 변환을 이용한 직선 검출

- `dabo.jpg` 이미지에서 `cv.Canny()`로 에지 맵 생성
- `cv.HoughLinesP()`로 직선을 검출
- 검출된 직선을 원본 이미지 위에 빨간색으로 표시

**이미지**

```02.CannyHoughLines.py
import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path  

week3_dir = Path(__file__).resolve().parent 
image_path = week3_dir / "dabo.jpg" 

img = cv.imread(str(image_path))  # 입력 이미지를 BGR 색상 형식으로 읽어 온다.

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 캐니 에지 검출을 위해 입력 영상을 그레이스케일로 변환한다.
blurred = cv.GaussianBlur(gray, (5, 5), 0)  # 작은 노이즈를 줄여 허프 변환이 덜 흔들리도록 가우시안 블러를 적용한다.
# 요구사항: cv.line()에서 색상은 (0, 0, 255) (빨간색)과 두께는 2로 설정
# 힌트: Cv.Canny()에서 threshold1과 threshold2는 100과 200으로 설정
edges = cv.Canny(blurred, 100, 200)  # PDF 요구사항에 맞춰 threshold1=100, threshold2=200으로 에지를 검출한다.
line_image = img.copy() 

# 요구사항: cv.HoughtLinesP()를 사용하여 직선 검출
# 힌트: cv.HoughLinesP()에서 rho, theta, threshold, minLineLength, maxLineGap 값을 조정하여 직선 검출 성능을 개선
lines = cv.HoughLinesP(  # 확률적 허프 변환으로 직선 성분을 검출한다.
    edges,  # 입력으로 캐니 에지 맵을 사용한다.
    1,  # rho 해상도는 1픽셀 단위로 설정한다.
    np.pi / 180,  # theta 해상도는 1도로 설정한다.
    105,  # 누적 투표 임계값을 120으로 설정해 구조물의 대표 직선을 안정적으로 찾는다.
    minLineLength=50,  # 최소 직선 길이를 100픽셀로 설정해 짧은 잡음을 줄인다.
    maxLineGap=12,  # 같은 직선으로 볼 수 있는 최대 간격을 10픽셀로 설정한다.
) 

if lines is not None:  # 검출된 직선이 하나 이상 존재하는지 확인한다.
    for line in lines:  # 검출된 각 직선 후보를 하나씩 순회한다.
        x1, y1, x2, y2 = line[0]  # HoughLinesP가 반환한 직선의 양 끝 좌표를 꺼낸다.
        dx = x2 - x1  # 직선의 x축 방향 길이를 계산한다.
        dy = y2 - y1  # 직선의 y축 방향 길이를 계산한다.
        angle = abs(np.degrees(np.arctan2(dy, dx)))  # 직선의 기울기 각도를 절댓값 기준으로 계산한다.
        length = float(np.hypot(dx, dy))  # 직선의 실제 길이를 계산한다.
        mid_x = (x1 + x2) / 2  # 직선 중심의 x좌표를 계산한다.
        if not ((angle <= 12 or angle >= 78) and length >= 100 and mid_x >= 120):  # 배경 나뭇가지 과검출을 줄이기 위해 수평·수직에 가까운 긴 직선만 남긴다.
            continue  # 조건에 맞지 않는 직선은 그리지 않고 건너뛴다.
        # 요구사항: cv.line()을 사용하여 검출된 직선을 원본 이미지에 그림
        # 힌트: cv.line()에서 색상은 (0, 0, 255) (빨간색)과 두께는 2로 설정
        cv.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 필터를 통과한 직선만 빨간색 두께 2로 그린다.

rgb_original = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Matplotlib 표시를 위해 원본 이미지를 RGB로 변환한다.
rgb_line_image = cv.cvtColor(line_image, cv.COLOR_BGR2RGB)  # 직선이 그려진 결과 이미지도 RGB로 변환한다.
# 요구사항: Matplotlib를 사용하여 원본 이미지와 직선이 그려진 이미지를 나란히 시각화
figure, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(rgb_original) 
axes[0].set_title("Original Image") 
axes[1].imshow(rgb_line_image)
axes[1].set_title("Canny + HoughLinesP") 
figure.tight_layout()  # 제목과 이미지가 겹치지 않도록 레이아웃을 정리한다.

plt.show() 
plt.close(figure) 
```

- **Canny 에지 맵 생성**

PDF 요구사항 그대로 `cv.Canny()`의 임계값을 `100`, `200`으로 두었고, 불필요한 점 잡음을 줄이기 위해 얕은 `GaussianBlur`를 한 번 적용했습니다.

- **허프 변환으로 직선 후보 추출**

`cv.HoughLinesP()`를 사용해 픽셀 단위 직선 후보를 뽑았고, `rho`, `theta`, `threshold`, `minLineLength`, `maxLineGap` 값을 `dabo.jpg`에 맞게 조정했습니다.

- **과검출 완화**

좌측 나뭇가지 영역에서 기울어진 선이 과도하게 검출되어, 직선의 각도·길이·중심 위치를 이용해 구조물 위주 직선만 남기도록 후처리를 추가했습니다.

---

## 3. GrabCut을 이용한 대화식 영역 분할 및 객체 추출

- `coffee cup.JPG`에 초기 사각형 영역을 설정
- `cv.grabCut()`으로 전경과 배경을 분할
- 마스크와 배경 제거 결과를 `Matplotlib`으로 시각화

**이미지**

```03.GrabCut.py
import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt  
from pathlib import Path  

week3_dir = Path(__file__).resolve().parent 
image_path = week3_dir / "coffee cup.JPG" 

img = cv.imread(str(image_path))  # 입력 이미지를 BGR 색상 형식으로 읽어 온다.

mask = np.zeros(img.shape[:2], np.uint8) 
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)
# 요구사항: 초기 사각형 영역은 (x, y, width, height) 형식으로 설정
rect = (90, 70, 1100, 720)  # 컵, 받침, 숟가락이 모두 포함되도록 초기 사각형 영역을 설정한다.

# 요구사항: cv.grabCut()를 사용하여 대화식 분할을 수행
# 힌트: cv.grabCut()에서 bgdModel과 fgdModel은 np.zeros((1, 65), np.float64)로 초기화
cv.grabCut(  # 지정한 사각형을 바탕으로 GrabCut 분할을 수행한다.
    img,  # 입력 원본 이미지를 전달한다.
    mask,  # 전경과 배경 라벨을 저장할 마스크를 전달한다.
    rect,  # 사용자가 지정했다고 가정한 초기 사각형 영역을 전달한다.
    bgd_model,  # 배경 모델 버퍼를 전달한다.
    fgd_model,  # 전경 모델 버퍼를 전달한다.
    5,  # 모델을 5회 반복 업데이트하도록 설정한다.
    cv.GC_INIT_WITH_RECT,  # 사각형 초기화 방식으로 GrabCut을 수행한다.
)  # grabCut 호출을 마친다.


# 힌트: np.where()를 사용하여 마스크 값을 0 또는 1로 변경한 후 원본 이미지에 곱하여 배경을 제거
mask_binary = np.where(  # GrabCut 결과에서 전경에 해당하는 픽셀만 1로 남긴다.
    # 힌트: 마스크 값은 cv.GC_BGD, cv.GC_FGD, cv.GC_PR_BGD, cv.GC_PR_FGD를 사용
    (mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD),  # 확실한 전경과 가능한 전경을 모두 전경으로 취급한다.
    1,  # 전경 픽셀은 1로 설정한다.
    0,  # 배경 픽셀은 0으로 설정한다.
).astype("uint8")  # 곱셈에 바로 사용할 수 있도록 uint8 형식으로 변환한다.
mask_visual = mask_binary * 255  # 시각화를 위해 전경은 255, 배경은 0이 되도록 마스크를 확장한다.
# 요구사항: 마스크를 사용하여 원본 이미지에서 배경을 제거
result = img * mask_binary[:, :, np.newaxis]  # 원본 이미지에 마스크를 곱해 배경을 제거한 결과를 만든다.

display_img = img.copy()
top_left = (rect[0], rect[1])  # 사각형의 좌상단 좌표를 계산한다.
bottom_right = (rect[0] + rect[2], rect[1] + rect[3])  # 사각형의 우하단 좌표를 계산한다.
cv.rectangle(display_img, top_left, bottom_right, (0, 255, 0), 4)  # 원본 복사본 위에 초록색 초기 사각형을 그린다.

rgb_display = cv.cvtColor(display_img, cv.COLOR_BGR2RGB)  # Matplotlib 표시를 위해 사각형이 그려진 이미지를 RGB로 변환한다.
rgb_result = cv.cvtColor(result, cv.COLOR_BGR2RGB)  # 배경 제거 결과 이미지도 RGB로 변환한다.
# 요구사항: matplotlib를 사용하여 원본 이미지, 마스크 이미지, 배경 제거 이미지 세 개를 나란히 시각화
figure, axes = plt.subplots(1, 3, figsize=(18, 6))  
axes[0].imshow(rgb_display)  
axes[0].set_title("Original Image + Rectangle") 
axes[1].imshow(mask_visual, cmap="gray") 
axes[1].set_title("GrabCut Mask")  
axes[2].imshow(rgb_result) 
axes[2].set_title("Foreground Extraction") 
figure.tight_layout()  # 제목과 이미지가 겹치지 않도록 레이아웃을 정리한다.

plt.show() 
plt.close(figure)
```

- **사각형 기반 초기화**

GrabCut은 처음부터 모든 전경을 아는 방식이 아니라, 사용자가 지정한 사각형 안에 전경이 들어 있다고 가정하고 분할을 시작합니다. 그래서 컵, 받침, 숟가락이 모두 포함되도록 `rect = (90, 70, 1100, 720)`로 설정했습니다.

- **마스크 후처리**

GrabCut이 반환한 마스크에는 확실한 전경과 가능한 전경이 함께 들어 있으므로, `np.where()`로 둘 다 `1`로 바꾼 뒤 배경은 `0`으로 두어 이진 마스크를 만들었습니다.

- **배경 제거**

`result = img * mask_binary[:, :, np.newaxis]` 방식으로 원본 이미지의 각 채널에 마스크를 곱해 배경은 검게, 전경만 남도록 처리했습니다.
