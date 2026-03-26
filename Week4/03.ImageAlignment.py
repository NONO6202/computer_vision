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
    if len(pair) < 2:  # 최근접 이웃 후보가 두 개보다 적은 경우는 건너닫는다.
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

# --- [비교용 1단계: 단순 덮어쓰기 및 크롭] ---
warped_simple = warped_image.copy()  # 블렌딩 전 상태를 보존하기 위해 복사본을 만든다.
warped_simple[translation_y:translation_y + left_height, translation_x:translation_x + left_width] = left_image  # 기준 이미지인 img1을 단순하게 덮어쓴다.

# 단순 덮어쓰기 결과에서 검은 여백 자동 크롭
mask_simple = cv.cvtColor(warped_simple, cv.COLOR_BGR2GRAY) > 0  # 실제 이미지가 존재하는 영역만 True로 표시한다.
coords_simple = np.column_stack(np.where(mask_simple))  # 픽셀 좌표를 모은다.
if coords_simple.size > 0:
    top_s, left_s = coords_simple.min(axis=0)  # 유효 영역의 경계를 찾는다.
    bottom_s, right_s = coords_simple.max(axis=0)
    warped_simple_cropped = warped_simple[top_s:bottom_s + 1, left_s:right_s + 1]  # 검은 여백을 잘라낸다.
else:
    warped_simple_cropped = warped_simple  # 이미지 영역이 없으면 원본을 유지한다.

# --- [비교용 2단계: 거리 변환 기반 알파 블렌딩 적용 및 크롭] ---
warped_blended = warped_image.copy()  # warped_image는 이미 warped_right_corners까지 투시 변환된 상태이다.

left_canvas = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)  # 왼쪽 이미지를 캔버스 크기에 맞게 배치할 빈 배열을 만든다.
left_canvas[translation_y:translation_y + left_height, translation_x:translation_x + left_width] = left_image  # img1을 제자리에 넣는다.

left_mask = np.zeros((panorama_height, panorama_width), dtype=np.uint8)  # 왼쪽 이미지의 유효 영역 마스크를 생성한다.
left_mask[translation_y:translation_y + left_height, translation_x:translation_x + left_width] = 255  # img1 자리를 흰색으로 칠한다.

right_mask_base = np.full((right_height, right_width), 255, dtype=np.uint8)  # 오른쪽 이미지 원본 크기만큼 255로 채워진 마스크를 만든다.
right_mask = cv.warpPerspective(right_mask_base, translation_matrix @ homography_matrix, (panorama_width, panorama_height))  # 마스크에도 동일한 투시 변환을 적용한다.

left_dist = cv.distanceTransform(left_mask, cv.DIST_L2, 3)  # 거리 변환을 계산한다.
right_dist = cv.distanceTransform(right_mask, cv.DIST_L2, 3)

weight_sum = left_dist + right_dist + 1e-6  # 분모가 0이 되는 것을 방지하기 위해 작은 값을 더해 거리 합을 구한다.
alpha_left = left_dist / weight_sum  # 왼쪽 가중치(0.0~1.0)를 계산한다.
alpha_right = right_dist / weight_sum  # 오른쪽 가중치(0.0~1.0)를 계산한다.

alpha_left = np.repeat(alpha_left[:, :, np.newaxis], 3, axis=2)  # 컬러 채널과 곱하기 위해 차원을 확장한다.
alpha_right = np.repeat(alpha_right[:, :, np.newaxis], 3, axis=2)

warped_blended = (left_canvas * alpha_left + warped_blended * alpha_right).astype(np.uint8)  # 가중치를 곱하고 더해서 부드럽게 블렌딩된 파노라마를 만든다.

# 블렌딩 결과에서 검은 여백 자동 크롭
mask_blended = cv.cvtColor(warped_blended, cv.COLOR_BGR2GRAY) > 0  # 이미지가 존재하는 영역만 True로 표시한다.
coords_blended = np.column_stack(np.where(mask_blended))  # 좌표를 모은다.
if coords_blended.size > 0:
    top_b, left_b = coords_blended.min(axis=0)  # 유효 경계를 찾는다.
    bottom_b, right_b = coords_blended.max(axis=0)
    warped_blended_cropped = warped_blended[top_b:bottom_b + 1, left_b:right_b + 1]  # 검은 여백을 잘라낸다.
else:
    warped_blended_cropped = warped_blended  # 이미지 영역이 없으면 원본을 유지한다.

# --- [시각화 데이터 준비] ---
matches_mask = inlier_mask.ravel().tolist()  # RANSAC 인라이어 여부를 리스트로 변환한다.
matches_to_draw = good_matches[:80]  # 상위 80개 좋은 매칭만 시각화에 사용한다.
mask_to_draw = matches_mask[:80]  # 동일한 개수로 인라이어 마스크를 자른다.
matching_result = cv.drawMatches(  # 특징점 매칭 결과를 시각화한다.
    left_image, left_keypoints, right_image, right_keypoints,
    matches_to_draw, None, matchesMask=mask_to_draw,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

rgb_matching_result = cv.cvtColor(matching_result, cv.COLOR_BGR2RGB)  # RGB로 변환한다.
rgb_simple_cropped = cv.cvtColor(warped_simple_cropped, cv.COLOR_BGR2RGB)  # RGB로 변환한다.
rgb_blended_cropped = cv.cvtColor(warped_blended_cropped, cv.COLOR_BGR2RGB)  # RGB로 변환한다.

# 요구사항: 변환된 이미지(Warped Image)와 특징점 매칭 결과(Matching Result)를 나란히 출력
figure, axes = plt.subplots(3, 1, figsize=(9, 12)) 
axes[0].imshow(rgb_matching_result)
axes[0].set_title(f"Matching Result ({int(inlier_mask.sum())} inliers)") 
axes[1].imshow(rgb_simple_cropped)
axes[1].set_title("Cropped Panorama (Unblended - Harsh boundary)")
axes[2].imshow(rgb_blended_cropped)
axes[2].set_title("Cropped Panorama (Blended - Smooth boundary)")
figure.tight_layout()

plt.show()
plt.close(figure)