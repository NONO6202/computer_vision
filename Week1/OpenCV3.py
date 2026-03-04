import cv2 as cv
import numpy as np

original_img = cv.imread('Week1/soccer.jpg')
img = original_img.copy()

drawing = False
start_x, start_y = -1, -1
end_x, end_y = -1, -1
roi_extracted = False

# 마우스 이벤트를 처리하는 함수
def select_roi(event, x, y, flags, param):
    global drawing, start_x, start_y, end_x, end_y, img, original_img, roi_extracted
    
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y # 클릭한 시작점
        
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            img = original_img.copy() # 드래그 시 이전 사각형 잔상 제거
            # cv.rectangle() 함수로 드래그 중인 영역을 시각화
            cv.rectangle(img, (start_x, start_y), (x, y), (0, 255, 0), 2)
            
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y
        cv.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        
        # 드래그 방향에 상관없이 슬라이싱이 가능하도록 좌표 정렬
        x1, x2 = min(start_x, end_x), max(start_x, end_x)
        y1, y2 = min(start_y, end_y), max(start_y, end_y)
        
        if x1 != x2 and y1 != y2:
            # ROI 추출은 numpy 슬라이싱을 사용
            roi = original_img[y1:y2, x1:x2]
            # 마우스를 놓으면 해당 영역을 잘라내서 별도의 창에 출력
            cv.imshow('ROI', roi) 
            roi_extracted = True

cv.namedWindow('OpenCV3')
cv.setMouseCallback('OpenCV3', select_roi)

while True:
    cv.imshow('OpenCV3', img)
    key = cv.waitKey(1)
    
    if key == ord('r'): # r키를 누르면 영역 선택을 리셋하고 처음부터 다시 선택
        img = original_img.copy()
        roi_extracted = False
        if cv.getWindowProperty('ROI', cv.WND_PROP_VISIBLE) >= 1:
            cv.destroyWindow('ROI')
            
    elif key == ord('s'): # s 키를 누르면 선택한 영역을 이미지 파일로 저장
        if roi_extracted:
            x1, x2 = min(start_x, end_x), max(start_x, end_x)
            y1, y2 = min(start_y, end_y), max(start_y, end_y)
            roi = original_img[y1:y2, x1:x2]
            # cv.imwrite()를 사용하여 이미지를 저장
            cv.imwrite('ROI.jpg', roi) 
            
    elif key == ord('q'): # 종료 키 설정
        break

cv.destroyAllWindows()