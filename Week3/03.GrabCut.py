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
