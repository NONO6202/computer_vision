import cv2
import numpy as np
import glob

# 체크보드 내부 코너 개수
CHECKERBOARD = (9, 6)

# 체크보드 한 칸 실제 크기 (mm)
square_size = 25.0

# 코너 정밀화 조건
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 실제 좌표 생성
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# 저장할 좌표
objpoints = [] # 3D 실제 좌표
imgpoints = [] # 2D 이미지 좌표

images = glob.glob("Week2/images/calibration_images/left*.jpg")

img_size = None

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 코너 검출은 그레이스케일에서 수행
    if img_size is None:
        img_size = gray.shape[::-1] # (width, height)

    # 이미지에서 체스보드 코너 위치 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # 코너를 모두 성공적으로 찾은 경우에만 리스트에 추가
    if ret == True:
        objpoints.append(objp) # 실제 좌표 추가
        # 찾아낸 코너 픽셀 좌표를 서브픽셀 단위로 더욱 정교하게 다듬음
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2) # 정밀화된 2D 이미지 좌표 추가

# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------
# 3D 실제 좌표와 2D 이미지 좌표 매칭을 통해 카메라 파라미터 계산
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
# 3D 실제 좌표의 점이 2D 이미지 plane으로 맵핑되는 원리는 Homogeneous Coordinates(동차 좌표계)를 사용한 행렬 곱으로 표현됩니다.
# 수식: s[x, y, 1]^T = K * [R|t] * [X, Y, Z, 1]^T
# (회전 R, 평행이동 t)

print("카메라 내부 파라미터 행렬 K:")
print(K) # 카메라 내부 파라미터 (focal length, principal point)

print("\왜곡 계수:")
print(dist) # 왜곡 계수 (방사 왜곡, 접선 왜곡)
print(f"- 방사 왜곡: {dist[0][0:3]}") # 렌즈 중심에서 멀어질 수록 직선이 휘어지는 현상
print(f"- 접선 왜곡: {dist[0][3:5]}") # 렌즈와 이미지 센서가 완전히 평행하지 않을 때 발생

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------
if len(images) > 0:
    sample_img = cv2.imread(images[0])
    
    # 계산된 K와 dist를 이용하여 원본 이미지의 렌즈 왜곡을 펴줌(보정) 
    undistorted_img = cv2.undistort(sample_img, K, dist, None, K)
    
    # 결과를 화면에 출력하여 비교
    cv2.imshow('Original', sample_img)
    cv2.imshow('Undistorted', undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
