import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path

week4_dir = Path(__file__).resolve().parent
image_path = week4_dir / "mot_color70.jpg"

img = cv.imread(str(image_path))

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # SIFT 특징점을 안정적으로 검출하기 위해 이미지를 그레이스케일로 변환한다.
# 요구사항: cv.SIFT_create()를 사용하여 SIFT 객체를 생성
# 힌트: SIFT_create()의 매개변수를 변경하며 특징점 검출 결과를 비교
# 힌트: 특징점이 너무 많다면 nfeatures 값을 조정하여 제한
sift1 = cv.SIFT_create(nfeatures=2000)  # 특징점이 너무 많아지지 않도록 nfeatures를 2000으로 제한한 SIFT 객체를 만든다.
sift2 = cv.SIFT_create(nfeatures=4000)
# 요구사항: detectAndCompute()를 사용하여 특징점을 검출
keypoints1, descriptors = sift1.detectAndCompute(gray, None)  # 특징점 위치와 각 특징점의 128차원을 동시에 계산한다.
keypoints2, descriptors = sift2.detectAndCompute(gray, None)

# 요구사항: cv.drawKeypoints()를 사용하여 특징점을 이미지에 시각화
# 힌트: cv.drawKeypoints()의 flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS를 설정하면 특징점의 방향과 크기도 표시
keypoint_image1 = cv.drawKeypoints(  # 검출된 특징점을 원본 이미지 위에 시각화한다.
    img,  # 원본 이미지를 바탕으로 사용한다.
    keypoints1,  # 검출된 특징점 목록을 전달한다.
    None,  # 새로운 결과 이미지를 생성하도록 None을 전달한다.
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,  # 특징점의 위치뿐 아니라 크기와 방향까지 함께 그린다.
)
keypoint_image2 = cv.drawKeypoints(  # 검출된 특징점을 원본 이미지 위에 시각화한다.
    img,  # 원본 이미지를 바탕으로 사용한다.
    keypoints2,  # 검출된 특징점 목록을 전달한다.
    None,  # 새로운 결과 이미지를 생성하도록 None을 전달한다.
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,  # 특징점의 위치뿐 아니라 크기와 방향까지 함께 그린다.
)

rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Matplotlib에서 올바른 색으로 보이도록 원본 이미지를 RGB로 변환한다.
rgb_keypoint_image1 = cv.cvtColor(keypoint_image1, cv.COLOR_BGR2RGB)  # 특징점이 그려진 결과 이미지도 RGB로 변환한다.
rgb_keypoint_image2 = cv.cvtColor(keypoint_image2, cv.COLOR_BGR2RGB)  # 특징점이 그려진 결과 이미지도 RGB로 변환한다.
# 요구사항: matplotlib을 이용하여 원본 이미지와 특징점이 시각화된 이미지를 나란히 출력
figure, axes = plt.subplots(1, 3, figsize=(24, 6))
axes[0].imshow(rgb_img)
axes[0].set_title("Original Image") 
axes[1].imshow(rgb_keypoint_image1) 
axes[1].set_title(f"SIFT Keypoints ({len(keypoints1)})") 
axes[2].imshow(rgb_keypoint_image2) 
axes[2].set_title(f"SIFT Keypoints ({len(keypoints2)})") 
figure.tight_layout() 

plt.show()
plt.close(figure)
