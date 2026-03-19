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
