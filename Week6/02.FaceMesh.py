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
