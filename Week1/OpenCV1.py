import cv2 as cv
import numpy as np

# 이미지 로드
img = cv.imread('Week1/girl_laughing.jpg')

# 이미지를 그레이스케일로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# np.hstack()을 위해 그레이스케일 이미지 차원 변경 (1채널 -> 3채널)
gray_3ch = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

# 원본 이미지와 그레이스케일 이미지를 가로로 연결하여 출력
combined = np.hstack((img, gray_3ch))

# 결과를 화면에 표시하고, 아무 키나 누르면 창 닫기
cv.imshow('OpenCV1', combined)
cv.waitKey(0)
cv.destroyAllWindows()