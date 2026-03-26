import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 
from pathlib import Path

week4_dir = Path(__file__).resolve().parent
image1_path = week4_dir / "mot_color70.jpg" 
image2_path = week4_dir / "mot_color83.jpg"

# 요구사항: cv.imread()를 사용하여 두 개의 이미지를 불러옴
img1 = cv.imread(str(image1_path))
img2 = cv.imread(str(image2_path))

gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)  # 첫 번째 이미지에서 SIFT 특징점을 추출하기 위해 그레이스케일로 변환한다.
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)  # 두 번째 이미지에서도 같은 방식으로 그레이스케일로 변환한다.
# 요구사항: Cv.SIFT_create()를 사용하여 특징점을 검출
sift = cv.SIFT_create(nfeatures=400)  # 매칭에 쓸 특징점 수를 충분히 확보하기 위해 nfeatures를 400으로 설정한다.
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)  # 첫 번째 이미지의 특징점과 기술자를 계산한다.
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)  # 두 번째 이미지의 특징점과 기술자를 계산한다.

# 요구사항: cv.BFMatcher()와 knnMatch()를 사용하여 특징점을 매칭하고, 좋은 매칭점만 선별
# 힌트: BFMatcher(cv.NORM_L2, crossCheck=True)를 사용하면 간단한 매칭이 가능
matcher = cv.BFMatcher(cv.NORM_L2)  # SIFT 기술자에 맞는 L2 거리 기반 BFMatcher 객체를 만든다.
raw_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)  # 각 특징점마다 가장 가까운 후보 두 개를 찾아 비율 테스트 준비를 한다.
good_matches = [] 
for pair in raw_matches:  # 각 특징점에 대해 찾아진 최근접 이웃 쌍을 하나씩 순회한다.
    if len(pair) < 2:  # 비교할 최근접 이웃이 두 개보다 적은 경우는 건너뛴다.
        continue  
    first_match, second_match = pair  # 가장 가까운 후보와 두 번째로 가까운 후보를 꺼낸다.
    if first_match.distance < 0.75 * second_match.distance:  # Lowe의 거리 비율 테스트를 적용해 모호한 매칭을 제거한다.
        good_matches.append(first_match)  # 조건을 통과한 매칭만 좋은 매칭 리스트에 추가한다.

good_matches.sort(key=lambda match: match.distance)  # 더 좋은 매칭이 먼저 보이도록 거리 기준으로 정렬한다.
show = 75
matches_to_draw = good_matches[:show]  # 시각화가 너무 복잡해지지 않도록 상위 75개 매칭만 그린다.
# 요구사항: cv.drawMatches()를 사용하여 매칭 결과를 시각화
matching_image = cv.drawMatches(  # 두 이미지 사이의 매칭 결과를 하나의 이미지로 시각화한다.
    img1,  # 첫 번째 원본 이미지를 전달한다.
    keypoints1,  # 첫 번째 이미지의 특징점 목록을 전달한다.
    img2,  # 두 번째 원본 이미지를 전달한다.
    keypoints2,  # 두 번째 이미지의 특징점 목록을 전달한다.
    matches_to_draw,  # 선택된 좋은 매칭 목록만 시각화에 사용한다.
    None,  # 새로운 결과 이미지를 생성하도록 None을 전달한다.
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,  # 매칭되지 않은 단독 특징점은 그리지 않도록 설정한다.
)

original_pair = np.hstack((img1, img2))  # 두 원본 이미지를 가로로 이어 붙여 비교용 이미지를 만든다.
rgb_original_pair = cv.cvtColor(original_pair, cv.COLOR_BGR2RGB)  # Matplotlib 표시를 위해 비교용 이미지를 RGB로 변환한다.
rgb_matching_image = cv.cvtColor(matching_image, cv.COLOR_BGR2RGB)  # 매칭 결과 이미지도 RGB로 변환한다.
# 요구사항: matplotlib을 이용하여 매칭 결과를 출력
figure, axes = plt.subplots(1, 2, figsize=(18, 4))
axes[0].imshow(rgb_original_pair)
axes[0].set_title("Original Image Pair")
axes[1].imshow(rgb_matching_image)
axes[1].set_title(f"SIFT Matching Result ({(show)} matches)")
figure.tight_layout()

plt.show()
plt.close(figure)
