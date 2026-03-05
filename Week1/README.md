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
- **이미지 로드 및 리사이징**

`cv.imread()`를 통해 원본 이미지를 BGR로 불러온후 `cv.resize()`를 통해 이미지를 600*400 픽셀로 조정하여 보기 좋게했습니다.

- **색상 공간 변환**

`cv.cvtColor(img, cv.COLOR_BGR2GRAY)`를 통해 BGR인 3채널 이미지를 Gratscale인 1채널 이미지로 변환했습니다.

- **차원 불일치 주의 및 병합**

컬러 이미지(3차원 배열)와 흑백 이미지(2차원 배열)을 `np.hstack()`을 통해 가로로 이어 붙이려면 두 배열의 차원이 동일해야 합니다.
`gray_3ch = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)`를 통해 1채널 흑백 이미지를 시각적인 변화 없이 3채널 데이터로 늘린후 병합했습니다.

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
- **콜백 함수 등록**

`cv.setMouseCallback('OpenCV2', paint_brush)`를 통해 특정 창에서 발생하는 마우스 이벤트를 paint_brush에 전달합니다.

- **마우스 상태 관리**

글로벌 변수 drawing을 boolean으로 토글하여, 버튼을 누른채로 이동(`EVENT_MOUSEMOVE`) 할 때만 그림이 그려지도록 상태를 제어합니다.
좌클릭(`EVENT_LBUTTONDOWN`) 파란색, 우클릭(`EVENT_RBUTTONDOWN`) 빨간색으로 색상 변수를 할당했습니다.

- **동시클릭 오류방지**

마우스 좌우 버튼을 동시에 누르거나 떼는 상황에서 발생할 수 있는 연속 그리기 논리 오류를 방지하기 위해 `EVENT_LBUTTONUP` 또는 `EVENT_RBUTTONUP` 발생 시 즉각적으로 drawing 상태를 해제했습니다.

- **실시간 키보드 입력 감지**

`while True` 무한 루프 내에서 `cv.waitKey(1)`을 호출하여 1밀리초마다 키 입력을 확인하여, 
+와 - 키 입력에 따라 brush_size를 1~15 범위 내에서 제한적(`min`, `max` 함수 사용)으로 증감시킵니다.

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
- **드래그 잔상 제거**

마우스 를 드래그하는 동안(`EVENT_MOUSEMOVE`) `cv.rectangle`을 반복 호출하면 기존에 그려진 사각형 위에 계속 중첩되어 잔상이 남습니다.
이를 방지하기 위해 드래그 이벤트가 발생할 때마다 `img = original_img.copy()`를 호출하여 도화지를 초기화 한 후 사각형을 새로 그립니다.
`img = original_img.copy()`를 누락시 드래그 궤적을 따라 사각형이 누적되는 시각적 결함이 발생합니다.

![image](https://github.com/user-attachments/assets/df6b61b4-cab9-4f7f-abf0-c848821b7c6f)

- **좌표 정렬**

사용자가 우하단에서 좌상단으로 역방향 드래그를 할 경우, 시작 좌표 값이 종료 좌표 값이 종료 좌표 값보다 커지게 되어 슬라이싱(`img[y1:y2, x1:x2]`)시 에러가 발생하거나 빈 배열이 반환 됩니다.
이를 해결하기 위해 `min()`과 `max()`함수를 활용하여 항상 x1,y1이 좌상단을 x2,y2가 우하단을 가리키도록 정렬합니다.

**`min()`, `max()`가 없으면:**
콜백함수가 cv.imshow 줄에서 내부적인 에러를 뱉고 작동을 멈추기 때문에 새로운 ROI 창은 열리지 않습니다.
하지만 메인 스레드 자체는 오류 없이 살아있으므로 원본 이미지가 있는 창은 닫히지 않고 정상적으로 유지됩니다.

- **상태 제어 및 윈도우 관리**

r 키를 누르면 저장된 ROI를 초기화하고 별도로 생성된 'ROI' 창이 열려있는지 확인(`cv.getWindowProperty`)한 후 닫습니다. s 키 입력 시 `cv.imwrite()`를 통해 슬라이싱 된 배열 데이터를 실제 이미지 파일로 저장합니다.