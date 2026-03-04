# Week 1

---

## 이미지 불러오기 및 그레이스케일 변환

- OpenCV를 사용하여 이미지를 불러오고 화면에 출력
- 원본 이미지와 그레이스케일로 변환된 이미지를 나란히 표시

<img width="1199" height="430" alt="image" src="https://github.com/user-attachments/assets/9c74f070-b5bb-4e53-a405-c0b39e93d172" />

```OpenCV1.py
import cv2 as cv
import numpy as np

# 요구사항1: 이미지 로드
img = cv.imread('Week1/girl_laughing.jpg')

# 이미지 크기를 600*400으로 조정
img = cv.resize(img, dsize=(600, 400))

# 요구사항2: 이미지를 그레이스케일로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# np.hstack()을 위해 그레이스케일 이미지 차원 변경 (1채널 -> 3채널)
gray_3ch = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

# 요구사항3: 원본 이미지와 그레이스케일 이미지를 가로로 연결하여 출력
combined = np.hstack((img, gray_3ch))

# 요구사항4: 결과를 화면에 표시하고, 아무 키나 누르면 창 닫기
cv.imshow('OpenCV1', combined)q
cv.waitKey(0)
# 모든 창 닫기
cv.destroyAllWindows()
```
**변형 & 스케일링**

<img width="681" height="270" alt="image" src="https://github.com/user-attachments/assets/38795041-0c59-4cfe-88df-5ca90c1ee520" />

**차원 불일치 주의**

<img width="768" height="277" alt="무제" src="https://github.com/user-attachments/assets/6cb98829-b540-4220-85e5-647c69ee278e" />

---

## 페인팅 붓 크기 조절 기능 추가

- 마우스 입력으로 이미지 위에 붓질
- 키보드 입력을 이용해 붓의 크기를 조절하는 기능 추가

<img width="1440" height="989" alt="image" src="https://github.com/user-attachments/assets/bd4cd16f-c8ed-41ac-8e27-982270885ce4" />

```OpenCV2.py
import cv2 as cv

# 요구사항1: 초기 붓 크기 5
brush_size = 5 
img = cv.imread('Week1/girl_laughing.jpg')
drawing = False

# 마우스 이벤트 처리 함수
def paint_brush(event, x, y, flags, param):
    global drawing, brush_size, color, img
    
    # 요구사항4: "좌클릭=파란색", 우클릭=빨간색, 드래그로 연속 그리기
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        color = (255, 0, 0)  # 좌클릭: 파란색
        cv.circle(img, (x, y), brush_size, color, -1)
        
    # 요구사항4: 좌클릭=파란색, "우클릭=빨간색", 드래그로 연속 그리기
    elif event == cv.EVENT_RBUTTONDOWN:
        drawing = True
        color = (0, 0, 255)  # 우클릭: 빨간색
        cv.circle(img, (x, y), brush_size, color, -1)
        
    # 요구사항4: 좌클릭=파란색, 우클릭=빨간색, "드래그로 연속 그리기"
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:  # 드래그로 연속 그리기
            cv.circle(img, (x, y), brush_size, color, -1)
            
    # 마우스 동시에 누르면 계속 그리려지는 오류 방지
    elif event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP:
        drawing = False

cv.namedWindow('OpenCV2')
cv.setMouseCallback('OpenCV2', paint_brush)

# Key 입력은 루프 안에서 처리
while True:
    cv.imshow('OpenCV2', img)
    # cv.waitKey(1)로 받은 값을 이용해 +, -, q를 구분
    key = cv.waitKey(1)
    
    if key == ord('q'): # 요구사항5: q 키를 누르면 영상 창이 종료
        break
    elif key == ord('+'): # 요구사항2: + 입력 시 크기 1 증가
        brush_size = min(15, brush_size + 1) # 요구사항3: 최대 15로 제한
    elif key == ord('-'): # 요구사항2: - 입력 시 크기 1 감소
        brush_size = max(1, brush_size - 1) # 요구사항3: 최소 1로 제한

# 모든 창 닫기
cv.destroyAllWindows()
```

---

## 마우스로 영역 선택 및 ROI(관심영역) 추출

- 이미지를 불러오고 사용자가 마우스로 클릭하고 드래그하여 관심영역(ROI)을 선택
- 선택한 영역만 따로 저장하거나 표시

<img width="1387" height="975" alt="image" src="https://github.com/user-attachments/assets/17d717f2-26ea-46ff-87e9-53944dc54079" />

```OpenCV3.py
import cv2 as cv
import numpy as np

# 요구사항1: "이미지를 불러오고" 화면에 출력
original_img = cv.imread('Week1/soccer.jpg')
img = original_img.copy()

drawing = False
roi_extracted = False

# 마우스 이벤트를 처리하는 함수
def select_roi(event, x, y, flags, param):
    global drawing, start_x, start_y, end_x, end_y, img, original_img, roi_extracted
    
    # 요구사항3: "사용자가 클릭한 시작점"에서 드래그하여 사각형을 그리며 영역을 선택
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y # 클릭한 시작점
        
    # 요구사항3: 사용자가 클릭한 시작점에서 "드래그하여 사각형을 그리며 영역을 선택"
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            img = original_img.copy() # 드래그 시 이전 사각형 잔상 제거
            # 함수로 드래그 중인 영역을 시각화
            cv.rectangle(img, (start_x, start_y), (x, y), (0, 255, 0), 2)
            
    # 요구사항4: 마우스를 놓으면 해당 영역을 잘라내서 별도의 창에 출력
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y
        cv.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        
        # 드래그 방향에 상관없이 슬라이싱이 가능하도록 좌표 정렬
        x1, x2 = min(start_x, end_x), max(start_x, end_x)
        y1, y2 = min(start_y, end_y), max(start_y, end_y)
        
        if x1 != x2 and y1 != y2:
            roi = original_img[y1:y2, x1:x2]
            # 마우스를 놓으면 해당 영역을 잘라내서 별도의 창에 출력
            cv.imshow('ROI', roi) 
            roi_extracted = True

cv.namedWindow('OpenCV3')
# 요구사항2: cv.setMouseCallback()을 사용하여 마우스 이벤트를 처리
cv.setMouseCallback('OpenCV3', select_roi)

while True:
    cv.imshow('OpenCV3', img)
    key = cv.waitKey(1)
    
    if key == ord('r'): # 요구사항5: r키를 누르면 영역 선택을 리셋하고 처음부터 다시 선택
        img = original_img.copy()
        roi_extracted = False
        if cv.getWindowProperty('ROI', cv.WND_PROP_VISIBLE) >= 1:
            cv.destroyWindow('ROI')
            
    elif key == ord('s'): # 요구사항6: s 키를 누르면 선택한 영역을 이미지 파일로 저장
        if roi_extracted:
            x1, x2 = min(start_x, end_x), max(start_x, end_x)
            y1, y2 = min(start_y, end_y), max(start_y, end_y)
            roi = original_img[y1:y2, x1:x2]
            # cv.imwrite()를 사용하여 이미지를 저장
            cv.imwrite('ROI.jpg', roi) 
            
    elif key == ord('q'): # 종료 키 설정
        break

# 모든 창 닫기
cv.destroyAllWindows()
```
**드래그 시 이전 사각형 중첩으로 잔상 주의**

![image](https://github.com/user-attachments/assets/df6b61b4-cab9-4f7f-abf0-c848821b7c6f)

