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