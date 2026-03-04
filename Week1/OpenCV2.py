import cv2 as cv

# 초기 붓 크기 5
brush_size = 5 
img = cv.imread('Week1/girl_laughing.jpg')
drawing = False

# 마우스 이벤트 처리 함수
def paint_brush(event, x, y, flags, param):
    global drawing, brush_size, color, img
    
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        color = (255, 0, 0)  # 좌클릭: 파란색
        cv.circle(img, (x, y), brush_size, color, -1)
        
    elif event == cv.EVENT_RBUTTONDOWN:
        drawing = True
        color = (0, 0, 255)  # 우클릭: 빨간색
        cv.circle(img, (x, y), brush_size, color, -1)
        
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:  # 드래그로 연속 그리기
            cv.circle(img, (x, y), brush_size, color, -1)
            
    elif event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP:
        drawing = False

cv.namedWindow('OpenCV2')
cv.setMouseCallback('OpenCV2', paint_brush)

# Key 입력은 루프 안에서 처리
while True:
    cv.imshow('OpenCV2', img)
    # cv.waitKey(1)로 받은 값을 이용해 +, -, q를 구분
    key = cv.waitKey(1)
    
    if key == ord('q'): # q 키를 누르면 영상 창이 종료
        break
    elif key == ord('+'): # + 입력 시 크기 1 증가
        brush_size = min(15, brush_size + 1) # 최대 15로 제한
    elif key == ord('-'): # - 입력 시 크기 1 감소
        brush_size = max(1, brush_size - 1) # 최소 1로 제한

cv.destroyAllWindows()