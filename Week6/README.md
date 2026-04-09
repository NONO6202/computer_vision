# Week 6

`yolov3.weights`를 따로 L06에 저장해야합니다. 

---

## 1. SORT 알고리즘을 활용한 다중 객체 추적기 구현

- `YOLOv3`로 각 프레임의 객체를 검출
- 검출된 경계상자를 입력으로 받아 SORT 방식으로 객체를 추적
- 각 객체에 고유 ID를 부여하고 비디오 프레임 위에 실시간으로 표시

```01.SORTTracking.py
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment

INPUT_VIDEO = "Week6/L06/slow_traffic_small.mp4"
YOLO_CFG = "Week6/L06/yolov3.cfg"
YOLO_WEIGHTS = "Week6/L06/yolov3.weights" 
OUTPUT_VIDEO = "Week6/outputs/01_sort_tracking.mp4"
CONF_THRESHOLD = 0.5  # 객체 검출 confidence threshold를 설정한다.
NMS_THRESHOLD = 0.4  # NMS threshold를 설정한다.
MAX_AGE = 12  # 검출이 잠시 끊겨도 트랙을 유지할 최대 프레임 수를 설정한다.
MIN_HITS = 3  # 안정적인 트랙으로 인정할 최소 매칭 횟수를 설정한다.
MAX_FRAMES = 0  # 디버깅용 최대 처리 프레임 수를 설정한다. 0이면 전체 프레임을 처리한다.
NO_DISPLAY = False

COCO_CLASSES = ["person","bicycle","car","motorbike","aeroplane","bus","train","truck",]

TRACK_TARGET_IDS = {0, 1, 2, 3, 5, 7}  # 사람, 자전거, 자동차, 오토바이, 버스, 트럭만 추적 대상으로 사용한다.

def get_output_layer_names(network):
    layer_names = network.getLayerNames()  # 네트워크에 포함된 전체 레이어 이름 목록을 가져온다.
    output_indices = network.getUnconnectedOutLayers()  # 출력 레이어의 인덱스를 가져온다.
    return [layer_names[index - 1] for index in output_indices.flatten()]  # 1-based 인덱스를 0-based로 바꿔 출력 레이어 이름 목록을 만든다.


def compute_iou(box_a, box_b):
    x_left = max(box_a[0], box_b[0])  # 두 박스가 겹치는 영역의 왼쪽 x좌표를 계산한다.
    y_top = max(box_a[1], box_b[1])  # 두 박스가 겹치는 영역의 위쪽 y좌표를 계산한다.
    x_right = min(box_a[2], box_b[2])  # 두 박스가 겹치는 영역의 오른쪽 x좌표를 계산한다.
    y_bottom = min(box_a[3], box_b[3])  # 두 박스가 겹치는 영역의 아래쪽 y좌표를 계산한다.

    if x_right <= x_left or y_bottom <= y_top:  # 겹치는 면적이 없으면 IoU는 0으로 처리한다.
        return 0.0

    intersection_area = float((x_right - x_left) * (y_bottom - y_top))  # 겹치는 영역의 면적을 계산한다.
    area_a = float(max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1]))  # 첫 번째 박스 면적을 계산한다.
    area_b = float(max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1]))  # 두 번째 박스 면적을 계산한다.
    union_area = area_a + area_b - intersection_area  # 합집합 면적을 계산한다.

    if union_area <= 0.0:  # 분모가 0이 되는 예외 상황을 방지한다.
        return 0.0
    return intersection_area / union_area  # 교집합 면적을 합집합 면적으로 나눠 IoU를 반환한다.


class KalmanBoxTracker:
    next_track_id = 0  # 새 트랙이 생성될 때마다 고유 ID를 부여하기 위한 클래스 변수를 만든다.

    def __init__(self, bbox, class_id, confidence):
        self.kalman = cv.KalmanFilter(8, 4)  # x1, y1, x2, y2와 각 속도를 포함하는 8차원 칼만 필터를 만든다.
        self.kalman.transitionMatrix = np.array(  # 위치와 속도의 선형 상태 전이 행렬을 설정한다.
            [
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.kalman.measurementMatrix = np.array(  # 측정값은 x1, y1, x2, y2 네 좌표만 직접 관측하도록 측정 행렬을 설정한다.
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        self.kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2  # 상태 예측 오차를 작게 두어 박스가 급격히 튀지 않도록 만든다.
        self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1  # 검출 박스의 측정 노이즈를 반영하도록 설정한다.
        self.kalman.errorCovPost = np.eye(8, dtype=np.float32)  # 초기 오차 공분산을 단위행렬로 설정한다.

        initial_state = np.array([[bbox[0]], [bbox[1]], [bbox[2]], [bbox[3]], [0], [0], [0], [0]], dtype=np.float32)  # 초기 상태를 현재 검출 박스 기준으로 설정한다.
        self.kalman.statePost = initial_state  # 필터의 사후 상태를 초기화한다.

        self.track_id = KalmanBoxTracker.next_track_id  # 현재 트랙에 사용할 고유 ID를 가져온다.
        KalmanBoxTracker.next_track_id += 1  # 다음 트랙을 위해 고유 ID를 1 증가시킨다.

        self.class_id = class_id  # 현재 트랙의 클래스 ID를 저장한다.
        self.confidence = confidence  # 현재 트랙에 연결된 최근 검출 confidence를 저장한다.
        self.hits = 1  # 생성 직후 한 번은 매칭된 것으로 간주한다.
        self.age = 0  # 트랙이 살아온 프레임 수를 0으로 초기화한다.
        self.time_since_update = 0  # 마지막 업데이트 이후 지난 프레임 수를 0으로 초기화한다.

    def predict(self):
        prediction = self.kalman.predict()  # 칼만 필터로 다음 프레임의 박스 좌표를 예측한다.
        self.age += 1  # 트랙이 한 프레임 더 유지되었음을 기록한다.
        self.time_since_update += 1  # 업데이트 없이 지난 프레임 수도 1 증가시킨다.
        return prediction[:4].reshape(-1)  # 예측된 x1, y1, x2, y2 좌표만 1차원 배열로 반환한다.

    def update(self, bbox, class_id, confidence):
        measurement = np.array([[bbox[0]], [bbox[1]], [bbox[2]], [bbox[3]]], dtype=np.float32)  # 관측된 검출 박스를 측정 벡터 형식으로 만든다.
        self.kalman.correct(measurement)  # 칼만 필터에 새 측정값을 넣어 상태를 보정한다.
        self.class_id = class_id  # 최신 검출 결과의 클래스 ID로 갱신한다.
        self.confidence = confidence  # 최신 검출 confidence로 갱신한다.
        self.hits += 1  # 성공적으로 다시 매칭된 횟수를 1 증가시킨다.
        self.time_since_update = 0  # 방금 업데이트했으므로 미갱신 프레임 수를 0으로 되돌린다.

    def get_state(self):
        state = self.kalman.statePost[:4].reshape(-1)  # 현재 칼만 필터 상태에서 x1, y1, x2, y2만 꺼낸다.
        state[0] = min(state[0], state[2] - 1)  # x1이 x2보다 커지지 않도록 보정한다.
        state[1] = min(state[1], state[3] - 1)  # y1이 y2보다 커지지 않도록 보정한다.
        return state


def associate_detections_to_tracks(detections, trackers, iou_threshold):
    if len(trackers) == 0:  # 기존 트랙이 하나도 없으면 모든 검출을 새 트랙으로 만들어야 한다.
        return [], list(range(len(detections))), []  # 매칭 없음, 모든 검출 unmatched, 트랙 unmatched 없음으로 반환한다.

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)  # 검출과 트랙 사이 IoU를 저장할 행렬을 만든다.
    for detection_index, detection in enumerate(detections):  # 각 검출 박스를 하나씩 순회한다.
        for tracker_index, tracker_box in enumerate(trackers):  # 각 예측 트랙 박스를 하나씩 순회한다.
            iou_matrix[detection_index, tracker_index] = compute_iou(detection[:4], tracker_box)  # 각 쌍의 IoU를 계산해 행렬에 저장한다.

    matched_detection_indices, matched_tracker_indices = linear_sum_assignment(-iou_matrix)  # IoU가 최대가 되도록 헝가리안 알고리즘으로 최적 매칭을 찾는다.

    matches = []  # 최종적으로 인정된 매칭 쌍을 저장할 리스트를 만든다.
    unmatched_detections = set(range(len(detections)))  # 아직 매칭되지 않은 모든 검출 인덱스를 집합으로 만든다.
    unmatched_trackers = set(range(len(trackers)))  # 아직 매칭되지 않은 모든 트랙 인덱스를 집합으로 만든다.

    for detection_index, tracker_index in zip(matched_detection_indices, matched_tracker_indices):  # 헝가리안 알고리즘이 제안한 각 매칭 쌍을 순회한다.
        if iou_matrix[detection_index, tracker_index] < iou_threshold:  # IoU가 너무 낮으면 잘못된 연관으로 보고 버린다.
            continue  # 임계값 미만인 쌍은 실제 매칭으로 인정하지 않는다.
        matches.append((detection_index, tracker_index))  # 유효한 매칭 쌍은 최종 매칭 리스트에 추가한다.
        unmatched_detections.discard(detection_index)  # 매칭에 사용된 검출 인덱스는 unmatched 집합에서 제거한다.
        unmatched_trackers.discard(tracker_index)  # 매칭에 사용된 트랙 인덱스도 unmatched 집합에서 제거한다.

    return matches, sorted(unmatched_detections), sorted(unmatched_trackers)  # 최종 매칭과 미매칭 검출/트랙 인덱스를 반환한다.


def detect_objects(frame, network, output_layer_names, confidence_threshold, nms_threshold):
    frame_height, frame_width = frame.shape[:2]  # 현재 프레임의 높이와 너비를 가져온다.
    blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)  # YOLOv3 입력 형식에 맞게 프레임을 정규화된 blob으로 변환한다.
    network.setInput(blob)  # 생성한 blob을 네트워크 입력으로 설정한다.
    outputs = network.forward(output_layer_names)  # YOLOv3의 출력 레이어 결과를 한 번에 추론한다.

    boxes = []  # NMS 전에 후보 박스 좌표를 저장할 리스트를 만든다.
    confidences = []  # 각 후보 박스의 confidence를 저장할 리스트를 만든다.
    class_ids = []  # 각 후보 박스의 클래스 ID를 저장할 리스트를 만든다.

    for output in outputs:  # 출력 레이어별 결과를 하나씩 순회한다.
        for detection in output:  # 각 detection 벡터를 하나씩 순회한다.
            scores = detection[5:]  # detection 벡터의 5번 이후 값은 클래스별 score이므로 따로 꺼낸다.
            class_id = int(np.argmax(scores))  # 가장 높은 score를 가진 클래스 ID를 고른다.
            confidence = float(scores[class_id])  # 선택된 클래스의 score를 confidence로 사용한다.

            if class_id not in TRACK_TARGET_IDS:  # 교통 영상에서 추적할 대상 클래스가 아니면 건너뛴다.
                continue
            if confidence < confidence_threshold:  # confidence가 임계값보다 낮으면 잡음 검출로 보고 제외한다.
                continue

            center_x = int(detection[0] * frame_width)  # 정규화된 중심 x좌표를 실제 프레임 좌표로 변환한다.
            center_y = int(detection[1] * frame_height)  # 정규화된 중심 y좌표를 실제 프레임 좌표로 변환한다.
            width = int(detection[2] * frame_width)  # 정규화된 박스 너비를 실제 픽셀 크기로 변환한다.
            height = int(detection[3] * frame_height)  # 정규화된 박스 높이를 실제 픽셀 크기로 변환한다.
            x = max(0, center_x - width // 2)  # 중심 좌표를 좌상단 x좌표로 바꾼다.
            y = max(0, center_y - height // 2)  # 중심 좌표를 좌상단 y좌표로 바꾼다.

            boxes.append([x, y, width, height])  # NMS에 넣기 위해 좌상단 좌표와 크기 형식으로 박스를 저장한다.
            confidences.append(confidence)  # 해당 박스의 confidence도 함께 저장한다.
            class_ids.append(class_id)  # 해당 박스의 클래스 ID도 함께 저장한다.

    kept_indices = cv.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)  # 중복되는 후보 박스를 NMS로 제거한다.

    detections = []  # 최종적으로 사용할 검출 박스 목록을 저장할 리스트를 만든다.
    if len(kept_indices) == 0:  # NMS 이후 남은 박스가 없으면 빈 리스트를 반환한다.
        return detections

    for index in kept_indices.flatten():  # NMS를 통과한 각 박스 인덱스를 하나씩 순회한다.
        x, y, width, height = boxes[index]  # 해당 박스의 좌상단 좌표와 크기를 꺼낸다.
        detections.append([x, y, x + width, y + height, confidences[index], class_ids[index]])  # 추적기가 쓰기 좋은 x1,y1,x2,y2 형식으로 바꿔 저장한다.
    return detections


def draw_tracks(frame, trackers, min_hits):
    for tracker in trackers:  # 현재 살아 있는 모든 트랙을 하나씩 순회한다.
        if tracker.hits < min_hits and tracker.time_since_update > 0:  # 아직 충분히 안정화되지 않았고 이번 프레임에 매칭도 실패한 트랙은 표시하지 않는다.
            continue

        x1, y1, x2, y2 = tracker.get_state().astype(int)  # 현재 트랙 상태를 정수 박스 좌표로 변환한다.
        x1 = max(0, x1)  # 프레임 바깥쪽으로 나간 좌표를 왼쪽 경계 안으로 보정한다.
        y1 = max(0, y1)  # 프레임 바깥쪽으로 나간 좌표를 위쪽 경계 안으로 보정한다.
        x2 = min(frame.shape[1] - 1, x2)  # 오른쪽 좌표가 프레임을 넘지 않도록 보정한다.
        y2 = min(frame.shape[0] - 1, y2)  # 아래쪽 좌표가 프레임을 넘지 않도록 보정한다.

        class_name = COCO_CLASSES[tracker.class_id]  # 클래스 ID를 COCO 클래스 이름으로 변환한다.
        color_seed = (37 * tracker.track_id) % 255  # 트랙 ID마다 다른 색이 나오도록 간단한 시드를 만든다.
        color = (int((color_seed + 80) % 255), int((color_seed + 160) % 255), int((color_seed + 220) % 255))  # 시드 기반으로 BGR 색상을 만든다.

        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # 추적 중인 객체의 경계상자를 화면에 그린다.
        label = f"ID {tracker.track_id} | {class_name}"  # 트랙 ID와 클래스 이름을 함께 표시할 문자열을 만든다.
        cv.putText(frame, label, (x1, max(20, y1 - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)  # 경계상자 위에 트랙 ID와 클래스 이름을 그린다.


def main():
    input_path = Path(INPUT_VIDEO)
    cfg_path = Path(YOLO_CFG)
    # 요구사항: 객체 검출기 구현: YOLOv3와 같은 사전 훈련된 객체 검출 모델을 사용하여 각 프레임에서 객체를 검출합니다.
    weights_path = Path(YOLO_WEIGHTS)
    output_path = Path(OUTPUT_VIDEO)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    network = cv.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))  # YOLOv3 cfg와 weights 파일을 사용해 DNN 네트워크를 로드한다.
    output_layer_names = get_output_layer_names(network)  # YOLOv3 추론에 사용할 출력 레이어 이름 목록을 가져온다.
    capture = cv.VideoCapture(str(input_path))  # 입력 비디오 파일을 열어 프레임을 순차적으로 읽을 준비를 한다.

    fps = capture.get(cv.CAP_PROP_FPS) or 30.0  # 입력 비디오의 FPS
    frame_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))  # 입력 비디오 프레임의 너비
    frame_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))  # 입력 비디오 프레임의 높이
    writer = cv.VideoWriter(str(output_path), cv.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    trackers = []  # 현재 활성 상태인 SORT 트랙 객체 목록을 저장할 리스트를 만든다.
    frame_index = 0  # 몇 번째 프레임을 처리 중인지 기록할 카운터를 0으로 초기화한다.

    while True:
        success, frame = capture.read()  # 입력 비디오에서 다음 프레임을 한 장 읽는다.
        if not success:  # 더 이상 읽을 프레임이 없으면 반복을 종료한다.
            break

        frame_index += 1  # 프레임을 하나 읽었으므로 프레임 번호를 1 증가시킨다.
        if MAX_FRAMES > 0 and frame_index > MAX_FRAMES:  # 디버깅용 최대 프레임 수를 넘기면 처리 루프를 중단한다.
            break

        detections = detect_objects(frame, network, output_layer_names, CONF_THRESHOLD, NMS_THRESHOLD)  # 현재 프레임에서 YOLOv3로 객체를 검출한다.
        predicted_boxes = [tracker.predict() for tracker in trackers]  # 기존 트랙들의 다음 위치를 칼만 필터로 먼저 예측한다.

        matches, unmatched_detections, unmatched_trackers = associate_detections_to_tracks(detections, predicted_boxes, 0.3)  # IoU 기반 헝가리안 알고리즘으로 검출과 기존 트랙을 연관시킨다.

        for detection_index, tracker_index in matches:  # 성공적으로 매칭된 검출-트랙 쌍을 하나씩 순회한다.
            detection = detections[detection_index]  # 현재 매칭된 검출 결과를 꺼낸다.
            trackers[tracker_index].update(detection[:4], detection[5], detection[4])  # 검출 박스로 트랙 상태를 보정하고 최신 클래스/확률도 갱신한다.

        # 요구사항: mathworks.comSORT 추적기 초기화: 검출된 객체의 경계 상자를 입력으로 받아 SORT 추적기를 초기화합니다.
        for detection_index in unmatched_detections:  # 어떤 트랙과도 연결되지 않은 새 검출을 하나씩 순회한다.
            detection = detections[detection_index]  # 새로운 객체로 보이는 검출 결과를 꺼낸다.
            trackers.append(KalmanBoxTracker(detection[:4], detection[5], detection[4]))  # 새 검출마다 새로운 SORT 트랙을 생성해 목록에 추가한다.

        trackers = [tracker for index, tracker in enumerate(trackers) if tracker.time_since_update <= MAX_AGE or index not in unmatched_trackers]  # 너무 오래 업데이트되지 않은 트랙은 제거해 추적 안정성을 유지한다.
        trackers = [tracker for tracker in trackers if tracker.time_since_update <= MAX_AGE]  # 최종적으로 max_age를 넘긴 트랙은 목록에서 완전히 제거한다.

        # 요구사항: 결과 시각화: 추적된 각 객체에 고유 ID를 부여하고, 해당 ID와 경계 상자를 비디오 프레임에 표시하여 실시간으로 출력합니다.
        draw_tracks(frame, trackers, MIN_HITS)  # 현재 프레임 위에 안정적으로 유지 중인 트랙의 ID와 경계상자를 그린다.
        cv.putText(frame, f"Frame: {frame_index}", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)  # 좌측 상단에 현재 프레임 번호를 표시한다.
        cv.putText(frame, f"Active Tracks: {len(trackers)}", (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)  # 좌측 상단에 현재 활성 트랙 수를 함께 표시한다.

        writer.write(frame)
        if not NO_DISPLAY:
            cv.imshow("SORT Tracking", frame)
            key = cv.waitKey(1) & 0xFF
            if key == 27:  # ESC 키
                break

    capture.release()
    writer.release()
    if not NO_DISPLAY:
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
```

- **YOLOv3 객체 검출**

`cv.dnn.readNetFromDarknet()`으로 `yolov3.cfg`, `yolov3.weights`를 로드하고, 각 프레임마다 차량/사람 계열 클래스만 검출하도록 구성했습니다.

- **SORT 추적**

외부 `filterpy` 없이 `cv.KalmanFilter`와 `scipy.optimize.linear_sum_assignment()`를 사용해 SORT의 핵심 흐름인 예측, 데이터 연관, 트랙 갱신을 구현했습니다.

- **상단 설정값 기반 실행**

`python3 Week6/01.SORTTracking.py`

디버깅용으로 일부 프레임만 처리하려면 파일 상단의 `MAX_FRAMES` 값을 `120`처럼 바꿉니다.

창 없이 저장만 하려면 파일 상단의 `NO_DISPLAY = True`로 바꾼 뒤 실행합니다.

---

## 2. Mediapipe를 활용한 얼굴 랜드마크 추출 및 시각화

- `Mediapipe FaceMesh` 또는 `FaceLandmarker` fallback으로 얼굴 랜드마크를 검출
- OpenCV를 사용해 MP4 입력 영상을 읽음
- 검출된 얼굴 랜드마크 468개를 점으로 표시
- `ESC` 키를 누르면 프로그램 종료

```02.FaceMesh.py
# 요구사항: OpenCV를 사용하여 웹캠으로부터 실시간 영상을 캡처합니다.
# 웹캠이 없는 관계로 영상으로 대체하였습니다.

from pathlib import Path
import cv2 as cv

SOURCE = "Week6/L06/face.mp4"  # 기본 입력은 face.mp4로 설정한다.
OUTPUT = "Week6/outputs/02_facemesh_video.mp4"  # 처리 결과를 저장할 출력 비디오 경로를 설정한다.
MODEL_PATH = "Week6/L06/face_landmarker.task"  # Face Landmarker 모델 파일 경로를 설정한다.
MAX_NUM_FACES = 1  # 한 프레임에서 동시에 추적할 최대 얼굴 수를 설정한다.
MAX_FRAMES = 0  # 디버깅용 최대 처리 프레임 수를 설정한다. 0이면 전체 프레임을 처리한다.
NO_DISPLAY = False  # OpenCV 창을 띄우지 않고 결과만 저장하려면 True로 바꾼다.

def create_face_mesh_detector(mp):
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):  # 예전 mediapipe 패키지처럼 solutions.face_mesh가 존재하는지 확인한다.
        return "solutions", mp.solutions.face_mesh.FaceMesh(  # 기존 FaceMesh API를 사용할 수 있으면 그대로 검출기를 생성한다.
            static_image_mode=False,  # 실시간 스트림 모드로 동작하도록 설정한다.
            max_num_faces=MAX_NUM_FACES,  # 한 프레임에서 추적할 최대 얼굴 수를 설정값으로 반영한다.
            refine_landmarks=True,  # 눈과 입 주변 랜드마크를 더 세밀하게 보정하도록 설정한다.
            min_detection_confidence=0.5,  # 얼굴 검출 confidence 임계값을 0.5
            min_tracking_confidence=0.5,  # 얼굴 추적 confidence 임계값도 0.5
        )

    model_path = Path(MODEL_PATH)  # 최신 mediapipe Tasks API에서 사용할 모델 경로를 Path 객체로 변환한다.

    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))  # .task 모델 파일을 읽기 위한 BaseOptions 객체를 만든다.
    options = mp.tasks.vision.FaceLandmarkerOptions(  # 최신 mediapipe Tasks API용 FaceLandmarker 옵션을 구성한다.
        base_options=base_options,  # 모델 파일 경로를 포함한 BaseOptions를 전달한다.
        running_mode=mp.tasks.vision.RunningMode.VIDEO,  # 비디오 프레임 스트림 처리용 VIDEO 모드로 동작하도록 설정한다.
        num_faces=MAX_NUM_FACES,  # 한 프레임에서 추적할 최대 얼굴 수를 설정값으로 반영한다.
        min_face_detection_confidence=0.5,  # 얼굴 검출 confidence 임계값
        min_face_presence_confidence=0.5,  # 얼굴 존재 confidence 임계값
        min_tracking_confidence=0.5,  # 얼굴 추적 confidence 임계값
    )
    return "tasks", mp.tasks.vision.FaceLandmarker.create_from_options(options)  # Tasks API 기반 검출기를 생성해 반환한다.


def main():
    import mediapipe as mp

    video_source = Path(SOURCE)
    capture = cv.VideoCapture(str(video_source))  # MP4 파일을 열어 프레임을 순차적으로 읽을 준비를 한다.

    frame_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH)) or 640  # 프레임 너비를 가져오고 실패하면 640을 기본값으로 사용한다.
    frame_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)) or 480  # 프레임 높이를 가져오고 실패하면 480을 기본값으로 사용한다.
    fps = capture.get(cv.CAP_PROP_FPS) or 30.0  # 입력 소스 FPS를 가져오고 실패하면 30FPS를 기본값으로 사용한다.

    writer = None  # 출력 비디오 저장기를 기본적으로 사용하지 않도록 None으로 초기화한다.
    if OUTPUT:  # 출력 경로가 비어 있지 않은 경우에만 저장기를 생성한다.
        output_path = Path(OUTPUT)  # 출력 경로를 Path 객체로 변환한다.
        output_path.parent.mkdir(parents=True, exist_ok=True)  # 출력 폴더가 없으면 자동으로 생성한다.
        writer = cv.VideoWriter(str(output_path), cv.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))  # 처리 결과를 저장할 MP4 비디오 작성기를 만든다.

    # 요구사항: Mediapipe의 FaceMesh 모듈을 사용하여 얼굴 랜드마크 검출기를 초기화합니다.
    detector_type, face_mesh = create_face_mesh_detector(mp)  # 현재 설치된 mediapipe 형태에 맞는 얼굴 랜드마크 검출기를 생성한다.
    with face_mesh:  # 검출기를 컨텍스트로 열어 종료 시 자원을 자동으로 정리한다.
        frame_index = 0  # 몇 번째 프레임을 처리 중인지 기록할 카운터를 0으로 초기화한다.
        while True:  # 입력이 끝날 때까지 프레임 단위 처리를 반복한다.
            success, frame = capture.read()  # 웹캠 또는 mp4에서 다음 프레임을 한 장 읽는다.
            if not success:  # 더 이상 읽을 프레임이 없으면 반복을 종료한다.
                break
            frame_index += 1  # 프레임을 하나 읽었으므로 프레임 번호를 1 증가시킨다.
            if MAX_FRAMES > 0 and frame_index > MAX_FRAMES:  # 디버깅용 최대 프레임 수를 넘기면 처리 루프를 중단한다.
                break

            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Mediapipe는 RGB 입력을 사용하므로 현재 프레임을 RGB로 변환한다.
            timestamp_ms = int((frame_index / fps) * 1000.0)  # VIDEO 모드용 시간 정보를 밀리초 단위로 계산한다.

            if detector_type == "solutions":  # 예전 FaceMesh API를 사용 중인지 확인한다.
                results = face_mesh.process(rgb_frame)  # 기존 FaceMesh API로 현재 프레임의 얼굴 랜드마크를 추론한다.
                face_landmarks_list = results.multi_face_landmarks or []  # 기존 API 결과에서 랜드마크 목록을 꺼내고 없으면 빈 리스트를 사용한다.
                iterable_landmarks = [face_landmarks.landmark for face_landmarks in face_landmarks_list]  # 각 얼굴의 landmark 리스트만 뽑아 동일한 후처리 형식으로 맞춘다.
            else:  # 최신 Tasks API를 사용 중이면 VIDEO 모드 추론을 수행한다.
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)  # Tasks API가 요구하는 mp.Image 객체를 현재 RGB 프레임으로 생성한다.
                results = face_mesh.detect_for_video(mp_image, timestamp_ms)  # VIDEO 모드 FaceLandmarker로 현재 프레임의 얼굴 랜드마크를 추론한다.
                iterable_landmarks = results.face_landmarks or []  # Tasks API 결과에서 얼굴 랜드마크 목록을 꺼내고 없으면 빈 리스트를 사용한다.

            face_count = 0  # 현재 프레임에서 검출된 얼굴 수를 기록할 카운터를 0으로 초기화한다.
            if iterable_landmarks:  # 검출된 얼굴 랜드마크가 하나 이상 존재하는지 확인한다.
                face_count = len(iterable_landmarks)  # 현재 프레임에서 검출된 얼굴 수를 저장한다.
                for landmarks in iterable_landmarks:  # 각 얼굴의 랜드마크 집합을 하나씩 순회한다.
                    for landmark in landmarks:  # 각 랜드마크의 정규화 좌표를 하나씩 순회한다.
                        x = int(landmark.x * frame_width)  # 정규화된 x좌표를 현재 프레임의 픽셀 좌표로 변환한다.
                        y = int(landmark.y * frame_height)  # 정규화된 y좌표를 현재 프레임의 픽셀 좌표로 변환한다.
                        if 0 <= x < frame_width and 0 <= y < frame_height:  # 좌표가 프레임 범위를 벗어나지 않는지 확인한다.
                            # 요구사항: 검출된 얼굴 랜드마크를 실시간 영상에 점으로 표시합니다.
                            cv.circle(frame, (x, y), 1, (0, 255, 0), -1)  # 각 랜드마크를 초록색 점으로 프레임 위에 그린다.

            source_label = video_source.name
            cv.putText(frame, f"Source: {source_label}", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)  # 좌측 상단에 입력 소스 정보를 표시한다.
            cv.putText(frame, f"Faces: {face_count}", (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)  # 좌측 상단에 현재 검출된 얼굴 수를 함께 표시한다.
            cv.putText(frame, f"Backend: {detector_type}", (20, 90), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)  # 좌측 상단에 현재 사용 중인 Mediapipe 백엔드를 표시한다.

            if writer is not None:
                writer.write(frame)

            if not NO_DISPLAY:
                cv.imshow("FaceMesh", frame)
                key = cv.waitKey(1) & 0xFF
                # 요구사항: ESC 키를 누르면 프로그램이 종료되도록 설정합니다.
                if key == 27:
                    break

    capture.release()
    if writer is not None:
        writer.release()
    if not NO_DISPLAY:
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
```

- **MP4 입력 전용**

웹캠 입력 코드는 제거했고, 파일 상단의 `SOURCE` 값을 `"Week6/L06/face.mp4"`처럼 바꿔 다른 MP4 파일에 그대로 적용할 수 있게 맞췄습니다.

- **최신 mediapipe fallback**

현재 설치된 `mediapipe 0.10.33`에서는 예전 `mp.solutions.face_mesh`가 없어서, 코드가 자동으로 `Tasks API`의 `FaceLandmarker`와 `face_landmarker.task` 모델 파일을 사용하도록 fallback 하게 만들었습니다.

- **정규화 좌표를 픽셀 좌표로 변환**

FaceMesh가 반환하는 랜드마크는 `0~1` 범위의 정규화 좌표이므로, 프레임의 `width`, `height`를 곱해 OpenCV가 그릴 수 있는 픽셀 좌표로 바꿨습니다.

- **상단 설정값 기반 실행**

`python3 Week6/02.FaceMesh.py`

다른 MP4를 쓰려면 파일 상단의 `SOURCE` 값을 원하는 경로로 바꿉니다.

창 없이 저장만 하려면 파일 상단의 `NO_DISPLAY = True`로 바꾸고, 필요하면 `MAX_FRAMES = 120`처럼 제한을 둔 뒤 실행합니다.

---

## 실행 메모

- 기본 `python3`가 `Python 3.13`인 환경에서는 `mediapipe`가 `tensorflow/protobuf` 관련 충돌로 비정상 종료될 수 있습니다.
- FaceMesh는 `.venv-week6` 같은 `Python 3.12` 가상환경에서 실행하는 것을 기준으로 정리했습니다.
- `SOURCE = "Week6/L06/face.mp4"`와 `OUTPUT = "Week6/outputs/02_facemesh_video.mp4"` 기준 결과 비디오는 `02_facemesh_video.mp4`로 저장됩니다.
