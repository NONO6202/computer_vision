# Week 4

---

## 1. SIFT를 이용한 특징점 검출 및 시각화

- `mot_color70.jpg` 이미지에서 SIFT 특징점을 검출
- `cv.drawKeypoints()`로 특징점의 위치, 크기, 방향을 시각화
- `Matplotlib`으로 원본 이미지와 결과 이미지를 나란히 출력

<img width="2391" height="485" alt="image" src="https://github.com/user-attachments/assets/c4bec6a7-b391-477f-9d2a-5f7342ac9684" />

```01.SIFTKeypoints.py
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
```

- **SIFT 객체 생성**

`cv.SIFT_create(nfeatures=2000)`로 특징점 수를 제한해 결과가 지나치게 복잡해지지 않도록 조절했습니다.

`cv.SIFT_create(nfeatures=4000)`로 2000과 비교할수 있게 하였습니다.

- **특징점과 기술자 동시 계산**

`detectAndCompute()`는 특징점 위치와 각 특징점을 설명하는 128차원을 한 번에 계산합니다.

- **풍부한 시각화**

`cv.drawKeypoints()`에 `cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS`를 사용해 특징점의 방향과 스케일까지 함께 보이도록 했습니다.

---

## 2. SIFT를 이용한 두 영상 간 특징점 매칭

- `mot_color70.jpg`와 `mot_color83.jpg` 사이에서 SIFT 특징점을 매칭
- `BFMatcher`와 `knnMatch()`를 사용해 최근접 이웃을 찾음
- 거리 비율 테스트로 좋은 매칭만 남긴 뒤 결과를 시각화

<img width="1792" height="301" alt="image" src="https://github.com/user-attachments/assets/a882c8df-9926-4c26-a765-a28651ed1130" />

```02.SIFTMatching.py
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
```

- **Lowe 거리 비율 테스트**

`knnMatch()`로 최근접 이웃 두 개를 구한 뒤 `m.distance < 0.75 * n.distance` 조건을 적용해 모호한 매칭을 줄였습니다.

- **BFMatcher 사용**

SIFT 기술자는 실수형 벡터이므로 `cv.BFMatcher(cv.NORM_L2)`를 사용해 유클리드 거리 기반으로 비교했습니다.

- **가독성 있는 결과 시각화**

좋은 매칭이 너무 많아 선이 과도하게 겹치지 않도록 정렬 후 상위 75개만 그리도록 제한했습니다.

---

## 3. 호모그래피를 이용한 이미지 정합 (Image Alignment)

- `img1.jpg`와 `img2.jpg`에서 SIFT 특징점을 추출
- `BFMatcher + knnMatch()`로 좋은 매칭을 찾고 `cv.findHomography()`로 호모그래피 계산
- `cv.warpPerspective()`로 `img2`를 `img1` 기준 좌표계로 정렬
- `Distance Transform`을 추가해 거리변환을 이용해 경계선에서 멀어질수록 높은 가중치를 줘, 두 이미지가 겹치는 구간을 자연스럽게 합치는 알파 블렌딩을 수행 // 궁금해서 넣음

<img width="883" height="1168" alt="image" src="https://github.com/user-attachments/assets/56244623-78a6-4a1d-a813-061279289e02" />

```03.ImageAlignment.py
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 
from pathlib import Path 

week4_dir = Path(__file__).resolve().parent
left_image_path = week4_dir / "img1.jpg"
right_image_path = week4_dir / "img2.jpg"

# 요구사항: cv.imread()를 사용하여 두 개의 이미지를 불러옴
left_image = cv.imread(str(left_image_path))
right_image = cv.imread(str(right_image_path))

left_gray = cv.cvtColor(left_image, cv.COLOR_BGR2GRAY)  # 기준 이미지에서 SIFT 특징점을 추출하기 위해 그레이스케일로 변환한다.
right_gray = cv.cvtColor(right_image, cv.COLOR_BGR2GRAY)  # 대상 이미지에서도 같은 방식으로 그레이스케일로 변환한다.
# 요구사항: Cv.SIFT_create()를 사용하여 특징점을 검출
sift = cv.SIFT_create(nfeatures=500)  # 정합 정확도를 높이기 위해 비교적 많은 특징점을 추출하도록 설정한다.
left_keypoints, left_descriptors = sift.detectAndCompute(left_gray, None)  # 기준 이미지의 특징점과 기술자를 계산한다.
right_keypoints, right_descriptors = sift.detectAndCompute(right_gray, None)  # 대상 이미지의 특징점과 기술자를 계산한다.

# 요구사항: cv.BFMatcher()와 knnMatch()를 사용하여 특징점을 매칭하고, 좋은 매칭점만 선별
matcher = cv.BFMatcher(cv.NORM_L2)  # SIFT 기술자에 맞는 L2 거리 기반 BFMatcher 객체를 만든다.
# 힌트: knnMatch()로 두 개의 최근접 이웃을 구한 뒤, 거리 비율이 임계값(예: 0.7) 미만인 매칭점만 선별
raw_matches = matcher.knnMatch(left_descriptors, right_descriptors, k=2)  # 각 특징점마다 가장 가까운 후보 두 개를 구해 비율 테스트를 준비한다.
good_matches = []
for pair in raw_matches:  # 각 특징점에 대한 최근접 이웃 쌍을 하나씩 순회한다.
    if len(pair) < 2:  # 최근접 이웃 후보가 두 개보다 적은 경우는 건너뛴다.
        continue
    first_match, second_match = pair  # 첫 번째 후보와 두 번째 후보를 꺼낸다.
    if first_match.distance < 0.75 * second_match.distance:  # Lowe의 거리 비율 테스트로 신뢰할 수 있는 매칭만 고른다.
        good_matches.append(first_match)  # 조건을 통과한 매칭만 좋은 매칭 리스트에 추가한다.

good_matches.sort(key=lambda match: match.distance)  # 더 신뢰도 높은 매칭이 앞에 오도록 거리 기준으로 정렬한다.

left_points = np.float32([left_keypoints[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)  # 기준 이미지 좌표를 호모그래피 계산 형식으로 정리한다.
right_points = np.float32([right_keypoints[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)  # 대상 이미지 좌표도 같은 형식으로 정리한다.
# 요구사항: cv.findHomography()를 사용하여 호모그래피 행렬을 계산
# glsxm: cv.findHomography()에서 cv.RANSAC을 사용하면 이상점(Outlier) 영향을 줄일 수 있음
homography_matrix, inlier_mask = cv.findHomography(right_points, left_points, cv.RANSAC, 5.0)  # RANSAC을 사용해 이상점을 줄이며 img2를 img1 기준으로 정렬할 호모그래피를 계산한다.

left_height, left_width = left_image.shape[:2]  # 기준 이미지의 높이와 너비를 구한다.
right_height, right_width = right_image.shape[:2]  # 대상 이미지의 높이와 너비를 구한다.
left_corners = np.float32([[0, 0], [0, left_height], [left_width, left_height], [left_width, 0]]).reshape(-1, 1, 2)  # 기준 이미지의 네 모서리 좌표를 만든다.
right_corners = np.float32([[0, 0], [0, right_height], [right_width, right_height], [right_width, 0]]).reshape(-1, 1, 2)  # 대상 이미지의 네 모서리 좌표를 만든다.
warped_right_corners = cv.perspectiveTransform(right_corners, homography_matrix)  # 호모그래피를 적용했을 때 img2의 모서리가 어디로 이동하는지 계산한다.
all_corners = np.concatenate((left_corners, warped_right_corners), axis=0)  # 두 이미지의 모든 모서리를 한 배열에 모은다.
x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)  # 전체 파노라마의 최소 x, y 좌표를 계산한다.
x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)  # 전체 파노라마의 최대 x, y 좌표를 계산한다.

translation_x = max(0, -x_min)  # 음수 x 좌표가 있으면 오른쪽으로 이동시키기 위한 보정값을 계산한다.
translation_y = max(0, -y_min)  # 음수 y 좌표가 있으면 아래쪽으로 이동시키기 위한 보정값을 계산한다.
translation_matrix = np.array(  # 좌표계를 양수 범위로 옮기기 위한 3x3 평행이동 행렬을 만든다.
    [[1.0, 0.0, float(translation_x)], [0.0, 1.0, float(translation_y)], [0.0, 0.0, 1.0]],  # x축과 y축 이동량을 반영한 동차 좌표 행렬을 구성한다.
    dtype=np.float64,  # 호모그래피와 곱하기 위해 float64 형식으로 저장한다.
)

panorama_width = max(int(x_max - x_min), left_width + right_width)  # PDF 힌트를 반영해 최소한 두 이미지 너비를 합친 크기 이상의 파노라마 폭을 사용한다.
panorama_height = max(int(y_max - y_min), max(left_height, right_height))  # 세로 방향 이동까지 포함하면서 최소한 원본 높이 이상이 되도록 파노라마 높이를 정한다.
# 요구사항: cv.warpPerspective()를 사용하여 한 이미지를 변환하여 다른 이미지와 정렬
# 힌트: cv.warpPerspective()를 사용할 때 출력 크기를 두 이미지를 합친 파노라마 크기 (w1+w2, max(h1,h2))로 설정
warped_image = cv.warpPerspective(right_image, translation_matrix @ homography_matrix, (panorama_width, panorama_height))  # img2를 img1 좌표계 기준 파노라마 캔버스로 변환한다.
warped_image[translation_y:translation_y + left_height, translation_x:translation_x + left_width] = left_image  # 기준 이미지인 img1을 같은 캔버스 위에 덮어써 정렬 결과를 완성한다.    

non_black_mask = cv.cvtColor(warped_image, cv.COLOR_BGR2GRAY) > 0  # 정합 결과에서 실제 이미지가 존재하는 영역만 True로 표시한다.
non_black_points = np.column_stack(np.where(non_black_mask))  # 검은 여백이 아닌 모든 픽셀의 좌표를 행렬 형태로 모은다.
if non_black_points.size > 0:  # 실제 이미지 영역이 하나라도 존재하는지 확인한다.
    top_row, left_col = non_black_points.min(axis=0)  # 유효 영역의 가장 위쪽과 가장 왼쪽 좌표를 계산한다.
    bottom_row, right_col = non_black_points.max(axis=0)  # 유효 영역의 가장 아래쪽과 가장 오른쪽 좌표를 계산한다.
    warped_image = warped_image[top_row:bottom_row + 1, left_col:right_col + 1]  # 계산된 경계만 남기도록 검은 여백을 잘라낸다.

matches_mask = inlier_mask.ravel().tolist()  # RANSAC이 정상 대응점으로 인정한 인라이어 여부를 리스트로 변환한다.
matches_to_draw = good_matches[:80]  # 매칭 선이 너무 많아지지 않도록 상위 80개 좋은 매칭만 시각화에 사용한다.
mask_to_draw = matches_mask[:80]  # 선택한 매칭 수에 맞춰 인라이어 마스크도 같은 길이로 자른다.
matching_result = cv.drawMatches(  # 특징점 매칭 결과를 한 장의 이미지로 시각화한다.
    left_image,  # 기준 이미지인 img1을 전달한다.
    left_keypoints,  # 기준 이미지의 특징점 목록을 전달한다.
    right_image,  # 대상 이미지인 img2를 전달한다.
    right_keypoints,  # 대상 이미지의 특징점 목록을 전달한다.
    matches_to_draw,  # 좋은 매칭 중 시각화용으로 고른 매칭 목록을 전달한다.
    None,  # 새로운 결과 이미지를 생성하도록 None을 전달한다.
    matchesMask=mask_to_draw,  # RANSAC 인라이어로 판별된 매칭만 선으로 표시한다.
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,  # 매칭되지 않은 단독 특징점은 그리지 않도록 설정한다.
)

rgb_warped_image = cv.cvtColor(warped_image, cv.COLOR_BGR2RGB)  # Matplotlib 표시를 위해 정합 결과 이미지를 RGB로 변환한다.
rgb_matching_result = cv.cvtColor(matching_result, cv.COLOR_BGR2RGB)  # 매칭 시각화 이미지도 RGB로 변환한다.
# 요구사항: 변환된 이미지(Warped Image)와 특징점 매칭 결과(Matching Result)를 나란히 출력
figure, axes = plt.subplots(1, 2, figsize=(20, 8))  # 정합 결과와 매칭 결과를 나란히 배치할 도화지를 만든다.
axes[0].imshow(rgb_warped_image)
axes[0].set_title("Warped Image") 
axes[1].imshow(rgb_matching_result)
axes[1].set_title(f"Matching Result ({int(inlier_mask.sum())} inliers)")
figure.tight_layout()

plt.show()
plt.close(figure) 
```

- **호모그래피 계산**

`cv.findHomography(..., cv.RANSAC, 5.0)`를 사용해 이상점의 영향을 줄이면서 `img2`를 `img1` 좌표계에 맞추는 변환 행렬을 계산했습니다.

- **평행이동 보정**

원근 변환 후 좌표가 음수가 될 수 있어, 모서리 좌표의 최소값을 기준으로 추가 평행이동 행렬을 곱해 파노라마 캔버스 안으로 옮겼습니다.

- **검은 여백 자동 제거**

`warpPerspective()` 결과 주변에는 빈 캔버스가 남기 쉬워서, 0이 아닌 픽셀 좌표 범위를 계산해 최종 파노라마를 자동으로 크롭했습니다.

- **인라이어 중심 매칭 시각화**

RANSAC이 정상 대응점으로 인정한 인라이어만 `drawMatches()`에 표시해, 실제 정합에 기여한 매칭만 볼 수 있도록 구성했습니다.
