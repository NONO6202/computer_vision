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
