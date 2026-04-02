from pathlib import Path  # 운영체제와 상관없이 경로를 다루기 위해 Path를 불러온다.

print("tensorflow.keras 버전 스크립트를 시작합니다.", flush=True)  # TensorFlow import 이전까지는 정상 실행되는지 확인하기 위해 시작 메시지를 즉시 출력한다.

import matplotlib.pyplot as plt  # 학습 곡선과 예측 결과를 시각화하기 위해 Matplotlib를 불러온다.
import numpy as np  # 배열 전처리와 예측 결과 계산을 위해 NumPy를 불러온다.
import tensorflow as tf  # 사용자가 요청한 tensorflow.keras 버전을 그대로 재현하기 위해 TensorFlow를 불러온다.
from tensorflow.keras import Sequential  # PDF 힌트에 맞춰 Sequential 모델을 사용하기 위해 불러온다.
from tensorflow.keras.datasets import mnist  # PDF 요구사항에 맞춰 tensorflow.keras.datasets의 MNIST를 사용한다.
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input  # 완전연결 신경망을 구성할 레이어들을 불러온다.

print(f"TensorFlow version: {tf.__version__}", flush=True)  # import가 성공한 환경에서는 TensorFlow 버전을 즉시 출력한다.

week5_dir = Path(__file__).resolve().parent  # 현재 스크립트가 들어 있는 Week5 폴더 경로를 구한다.
output_dir = week5_dir / "outputs"  # 결과 이미지와 요약 파일을 저장할 출력 폴더 경로를 만든다.
output_dir.mkdir(parents=True, exist_ok=True)  # 출력 폴더가 없으면 자동으로 생성한다.

# 요구사항: MNIST 데이터셋을 로드
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # TensorFlow Keras API로 MNIST 학습 세트와 테스트 세트를 메모리로 불러온다.

train_limit = 15000  # CPU 환경에서도 적당한 시간 안에 학습이 끝나도록 학습 데이터 일부만 사용한다.
test_limit = 3000  # 평가와 예측 시각화에 사용할 테스트 데이터 수도 제한한다.
x_train = x_train[:train_limit].astype("float32") / 255.0  # 픽셀 값을 0~1 범위로 정규화해 학습이 안정되도록 만든다.
y_train = y_train[:train_limit]  # 학습용 정답 라벨도 같은 개수만 남긴다.
x_test = x_test[:test_limit].astype("float32") / 255.0  # 테스트 이미지도 같은 방식으로 정규화한다.
y_test = y_test[:test_limit]  # 테스트 라벨도 같은 개수만 남긴다.

# 요구사항: 간단한 신경망 모델을 구축
model = Sequential(  # PDF의 힌트에 맞춰 tensorflow.keras.Sequential 기반 완전연결 신경망을 구성한다.
    [
        Input(shape=(28, 28)),  # 손글씨 숫자 이미지는 28x28 흑백 이미지이므로 입력 크기를 이에 맞춘다.
        Flatten(),  # 2차원 이미지를 1차원 벡터로 펼쳐 Dense 레이어에 전달할 수 있게 만든다.
        Dense(256, activation="relu"),  # 첫 번째 은닉층에서 비선형 특징을 학습한다.
        Dropout(0.2),  # 일부 뉴런을 무작위로 끄어 과적합을 완화한다.
        Dense(128, activation="relu"),  # 두 번째 은닉층에서 더 추상적인 숫자 특징을 학습한다.
        Dense(10, activation="softmax"),  # 숫자 0~9의 확률을 출력하도록 마지막 층을 구성한다.
    ]
)  # tensorflow.keras.Sequential 모델 정의를 마친다.

model.compile(  # 학습 방법과 손실 함수, 평가 지표를 설정한다.
    optimizer="adam",  # 기본 성능이 좋은 Adam 옵티마이저를 사용한다.
    loss="sparse_categorical_crossentropy",  # 정답 라벨이 정수 형태이므로 sparse categorical crossentropy를 사용한다.
    metrics=["accuracy"],  # 분류 정확도를 함께 기록해 학습 성능을 확인한다.
)  # 모델 컴파일을 마친다.

# 요구사항: 모델을 훈련시키고 정확도를 평가
history = model.fit(  # 학습 데이터를 사용해 모델 파라미터를 업데이트한다.
    x_train,  # 정규화한 학습 이미지를 입력으로 사용한다.
    y_train,  # 학습 이미지에 대응하는 정답 라벨을 전달한다.
    epochs=4,  # CPU 환경을 고려해 4회 반복 학습한다.
    batch_size=128,  # 한 번에 128장씩 묶어 학습해 속도와 안정성의 균형을 맞춘다.
    validation_split=0.1,  # 학습 데이터의 10%를 검증용으로 떼어 과적합 여부를 함께 확인한다.
    verbose=2,  # 학습 로그를 간결한 형태로 출력한다.
)  # 모델 학습을 마친다.

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)  # 학습이 끝난 모델을 테스트 세트에서 평가한다.

sample_count = 10  # 예측 결과를 보여 줄 테스트 샘플 개수를 10개로 정한다.
sample_images = x_test[:sample_count]  # 시각화에 사용할 첫 10개 테스트 이미지를 가져온다.
sample_labels = y_test[:sample_count]  # 각 테스트 이미지의 실제 정답 라벨도 함께 가져온다.
sample_probabilities = model.predict(sample_images, verbose=0)  # 샘플 이미지에 대한 클래스별 확률을 계산한다.
sample_predictions = np.argmax(sample_probabilities, axis=1)  # 가장 높은 확률을 가진 클래스를 최종 예측 숫자로 선택한다.

summary_path = output_dir / "01_mnist_tensorflow_summary.txt"  # 학습 결과를 텍스트로 저장할 파일 경로를 만든다.
with summary_path.open("w", encoding="utf-8") as summary_file:  # 결과 요약 파일을 UTF-8 인코딩으로 연다.
    summary_file.write("MNIST tensorflow.keras 분류기 학습 결과\n")  # 파일 첫 줄에 제목을 기록한다.
    summary_file.write(f"- 학습 샘플 수: {len(x_train)}\n")  # 사용한 학습 데이터 개수를 기록한다.
    summary_file.write(f"- 테스트 샘플 수: {len(x_test)}\n")  # 사용한 테스트 데이터 개수를 기록한다.
    summary_file.write(f"- 테스트 손실: {test_loss:.4f}\n")  # 테스트 손실 값을 소수점 넷째 자리까지 기록한다.
    summary_file.write(f"- 테스트 정확도: {test_accuracy:.4f}\n")  # 테스트 정확도를 소수점 넷째 자리까지 기록한다.

figure = plt.figure(figsize=(18, 10))  # 학습 곡선과 샘플 예측을 함께 담을 큰 도화지를 만든다.
grid = figure.add_gridspec(3, 5)  # 상단은 학습 곡선, 하단 두 줄은 샘플 예측 이미지를 배치하기 위한 격자를 만든다.

accuracy_axis = figure.add_subplot(grid[0, :3])  # 정확도 곡선을 그릴 축을 첫 번째 행 왼쪽에 만든다.
accuracy_axis.plot(history.history["accuracy"], label="Train Accuracy")  # 에폭별 학습 정확도 변화를 선 그래프로 그린다.
accuracy_axis.plot(history.history["val_accuracy"], label="Validation Accuracy")  # 에폭별 검증 정확도 변화를 함께 그린다.
accuracy_axis.set_title("MNIST Accuracy (tensorflow.keras)")  # 정확도 그래프 제목을 설정한다.
accuracy_axis.set_xlabel("Epoch")  # x축이 에폭 번호임을 표시한다.
accuracy_axis.set_ylabel("Accuracy")  # y축이 정확도 값임을 표시한다.
accuracy_axis.legend()  # 학습 정확도와 검증 정확도를 구분할 범례를 표시한다.

loss_axis = figure.add_subplot(grid[0, 3:])  # 손실 곡선을 그릴 축을 첫 번째 행 오른쪽에 만든다.
loss_axis.plot(history.history["loss"], label="Train Loss")  # 에폭별 학습 손실 변화를 선 그래프로 그린다.
loss_axis.plot(history.history["val_loss"], label="Validation Loss")  # 에폭별 검증 손실 변화를 함께 그린다.
loss_axis.set_title("MNIST Loss (tensorflow.keras)")  # 손실 그래프 제목을 설정한다.
loss_axis.set_xlabel("Epoch")  # x축이 에폭 번호임을 표시한다.
loss_axis.set_ylabel("Loss")  # y축이 손실 값임을 표시한다.
loss_axis.legend()  # 학습 손실과 검증 손실을 구분할 범례를 표시한다.

for index in range(sample_count):  # 선택한 10개 샘플 이미지를 하나씩 순회한다.
    sample_axis = figure.add_subplot(grid[1 + index // 5, index % 5])  # 두 번째와 세 번째 행에 샘플 이미지를 순서대로 배치한다.
    sample_axis.imshow(sample_images[index], cmap="gray")  # 흑백 숫자 이미지를 회색조로 표시한다.
    sample_axis.set_title(f"T:{sample_labels[index]} / P:{sample_predictions[index]}")  # 실제 숫자와 예측 숫자를 제목으로 표시한다.
    sample_axis.axis("off")  # 샘플 이미지 주변의 좌표축 눈금을 숨긴다.

figure.suptitle(f"MNIST tensorflow.keras Result - Test Accuracy: {test_accuracy:.4f}", fontsize=16)  # 전체 figure 상단에 최종 테스트 정확도를 표시한다.
figure.tight_layout()  # 각 그래프와 이미지가 겹치지 않도록 레이아웃을 정리한다.

output_path = output_dir / "01_mnist_tensorflow_result.png"  # 최종 시각화 이미지를 저장할 파일 경로를 만든다.
figure.savefig(output_path, dpi=200, bbox_inches="tight")  # 화면에 보이는 결과를 PNG 파일로 저장한다.
if "agg" not in plt.get_backend().lower():  # 비대화형 백엔드가 아닐 때만 화면 표시를 시도한다.
    plt.show()  # 사용자가 일반 환경에서 실행했을 때 결과 창이 화면에 나타나도록 한다.
plt.close(figure)  # 스크립트 종료 전에 figure 자원을 정리한다.
