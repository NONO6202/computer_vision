import os
from pathlib import Path

# 현재 로컬 환경에서는 TensorFlow가 비정상 종료되어 Keras torch 백엔드를 사용
os.environ.setdefault("KERAS_BACKEND", "torch") 

import keras
# Dense, Flatten 레이어를 사용하기 위함
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist

week5_dir = Path(__file__).resolve().parent
output_dir = week5_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# 요구사항: MNIST 데이터셋을 로드
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # MNIST 학습 세트와 테스트 세트를 메모리로 불러온다.

dataset_len = int(len(x_train) / 10)
# 요구사항: 데이터를 훈련 세트와 테스트 세트로 분할
train_limit =  dataset_len * 9 
test_limit = dataset_len
x_train = x_train[:train_limit].astype("float32") / 255.0  # 픽셀 값을 0~1 범위로 정규화해 학습이 안정되도록 만든다.
y_train = y_train[:train_limit]
x_test = x_test[:test_limit].astype("float32") / 255.0
y_test = y_test[:test_limit] 

# 요구사항: 간단한 신경망 모델을 구축
# 힌트: Sequential 모델과 Dense 레이어를 활용하여 신경망을 구성
# PDF의 힌트에 맞춰 Sequential 기반 완전연결 신경망을 구성한다.
model = keras.Sequential(
    [
        # 힌트:  손글씨 숫자 이미지는 28x28 픽셀 크기의 흑백 이미지
        layers.Input(shape=(28, 28)),  # 손글씨 숫자 이미지는 28x28 흑백 이미지이므로 입력 크기를 이에 맞춤
        layers.Flatten(),  # 2차원 이미지를 1차원 벡터로 펼쳐 Dense 레이어에 전달할 수 있게 만듦
        layers.Dense(256, activation="relu"),  # 첫 번째 은닉층에서 비선형 특징을 학습
        layers.Dense(128, activation="relu"),  # 두 번째 은닉층에서 더 추상적인 숫자 특징을 학습
        layers.Dropout(0.1),  # dropout으로 과적합을 완화
        layers.Dense(10, activation="softmax"),  # 0~9로 softmax를 취함
    ]
)
# 학습 방법과 손실 함수, 평가 지표를 설정한다.
model.compile( 
    optimizer="adam",  # Adam 옵티마이저를 사용
    loss="sparse_categorical_crossentropy",  # 정답 라벨이 정수 형태이므로 sparse categorical crossentropy를 사용
    metrics=["accuracy"],  # 분류 정확도를 함께 기록해 학습 성능을 확인
) 

# 요구사항: 모델을 훈련시키고 정확도를 평가
# 학습 데이터를 사용해 모델 파라미터를 업데이트한다.
history = model.fit(
    x_train,
    y_train,
    epochs=10,  # 5회 반복 학습한다.
    batch_size=512,  # 한 번에 1024장 배치로 묶는다.
    validation_split=0.1,  # 학습 데이터의 0.1을 검증용으로 과적합 여부를 확인
)
# 학습이 끝난 모델을 테스트 세트에서 평가한다.
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)  

sample_count = 10
sample_images = x_test[:sample_count] 
sample_labels = y_test[:sample_count]
sample_probabilities = model.predict(sample_images, verbose=0)  # 샘플 이미지에 대한 클래스별 확률을 계산
sample_predictions = np.argmax(sample_probabilities, axis=1)  # 가장 높은 확률을 가진 클래스를 최종 예측 숫자로 선택

print("MNIST 분류기 학습 결과\n")
print(f"- 학습 샘플 수: {len(x_train)}\n") 
print(f"- 테스트 샘플 수: {len(x_test)}\n")
print(f"- 테스트 손실: {test_loss:.4f}\n")
print(f"- 테스트 정확도: {test_accuracy:.4f}\n")

figure = plt.figure(figsize=(18, 10))
grid = figure.add_gridspec(3, 5)

accuracy_axis = figure.add_subplot(grid[0, :3])
accuracy_axis.plot(history.history["accuracy"], label="Train Accuracy")
accuracy_axis.plot(history.history["val_accuracy"], label="Validation Accuracy") 
accuracy_axis.set_title("MNIST Accuracy")
accuracy_axis.set_xlabel("Epoch")
accuracy_axis.set_ylabel("Accuracy")
accuracy_axis.legend() 

loss_axis = figure.add_subplot(grid[0, 3:])
loss_axis.plot(history.history["loss"], label="Train Loss")
loss_axis.plot(history.history["val_loss"], label="Validation Loss")
loss_axis.set_title("MNIST Loss")
loss_axis.set_xlabel("Epoch")
loss_axis.set_ylabel("Loss") 
loss_axis.legend()

for index in range(sample_count):
    sample_axis = figure.add_subplot(grid[1 + index // 5, index % 5])
    sample_axis.imshow(sample_images[index], cmap="gray")
    sample_axis.set_title(f"T:{sample_labels[index]} / P:{sample_predictions[index]}")

figure.suptitle(f"MNIST Classifier Result - Test Accuracy: {test_accuracy:.4f}", fontsize=16) 
figure.tight_layout() 

output_path = output_dir / "01_mnist_result.png"
figure.savefig(output_path)
plt.show()
plt.close(figure)
