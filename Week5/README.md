# Week 5

---

## 1. 간단한 이미지 분류기 구현

- MNIST 데이터셋을 로드하여 손글씨 숫자 분류 모델을 학습
- `Sequential`과 `Dense` 레이어를 사용해 간단한 신경망을 구성
- 모델을 훈련시키고 테스트 정확도를 평가한 뒤 예측 결과를 시각화

> 참고: PDF 힌트는 `tensorflow.keras` 기준이지만, 현재 로컬 Python 3.13 환경에서는 TensorFlow가 비정상 종료되어 동일한 Keras API를 `torch backend`로 실행하도록 구성했습니다.

<img width="817" height="411" alt="image" src="https://github.com/user-attachments/assets/afdbcf60-7961-41d3-b74a-c8714f84d2c1" />
<img width="1800" height="1000" alt="01_mnist_result" src="https://github.com/user-attachments/assets/fd18451a-ef0c-4eee-a469-3b26afbe3bf3" />

```01.MNISTClassifier.py
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
    batch_size=512,  # 한 번에 512장 배치로 묶는다.
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
```

- **MNIST 데이터 전처리**

손글씨 숫자 이미지는 `28x28` 흑백 이미지이므로, `float32`로 바꾼 뒤 `255.0`으로 나누어 0~1 범위로 정규화했습니다.

- **간단한 완전연결 신경망**

PDF 힌트대로 `Sequential` 모델과 `Dense` 레이어를 사용했고, 입력 이미지는 `Flatten()`으로 1차원 벡터로 펼친 뒤 분류하도록 구성했습니다.

---

## 2. CIFAR-10 데이터셋을 활용한 CNN 모델 구축

- CIFAR-10 데이터셋을 로드하고 픽셀 값을 0~1 범위로 정규화
- `Conv2D`, `MaxPooling2D`, `Dense` 레이어로 CNN 모델을 구성
- 모델 성능을 평가하고 `dog.jpg`에 대한 예측 결과를 출력

<img width="820" height="442" alt="image" src="https://github.com/user-attachments/assets/42579efe-c5f9-41d0-9a41-6378c37f5831" />
<img width="3581" height="1970" alt="02_cifar10_result" src="https://github.com/user-attachments/assets/c4da6125-b0a7-4d5a-b419-db575f8bbadc" />


```02.CIFAR10CNN.py
import os
from pathlib import Path
# 현재 로컬 환경에서는 TensorFlow가 비정상 종료되어 Keras torch 백엔드를 사용
os.environ.setdefault("KERAS_BACKEND", "torch")

import cv2 as cv
import keras
# Conv2D, MaxPooling2D, Dense 등의 레이어를 사용하기 위해 layers를 불러온다.
from keras import layers 
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras.datasets import cifar10

# CIFAR-10의 클래스 번호를 사람이 읽을 수 있는 이름으로 대응시킨다.
class_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

week5_dir = Path(__file__).resolve().parent
dog_image_path = week5_dir / "dog.jpg"
output_dir = week5_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# 요구사항: CIFAR-10 데이터셋을 로드
(x_train, y_train), (x_test, y_test) = cifar10.load_data()  # CIFAR-10 학습 세트와 테스트 세트를 메모리로 불러온다.

dataset_len = int(len(x_train) / 10)
# train, test dataset을 분류한다.
train_limit =  dataset_len * 9 
test_limit = dataset_len
# 힌트: 데이터 전처리 시 픽셀 값을 0~1 범위로 정규화하면 모델의 수렴이 빨라질 수 있음
x_train = x_train[:train_limit].astype("float32") / 255.0  # 요구사항에 맞춰 픽셀 값을 0~1 범위로 정규화한다.
y_train = y_train[:train_limit].reshape(-1)  # CIFAR 라벨은 (N,1) 형태라서 1차원으로 펼쳐 손실 함수에 맞춘다.
x_test = x_test[:test_limit].astype("float32") / 255.0 
y_test = y_test[:test_limit].reshape(-1) 

# 요구사항: CNN 모델을 설계하고 훈련
# 힌트: Conv2D, MaxPooling2D, Flatten, Dense 레이어를 활용하여 CNN을 구성
# PDF의 힌트에 맞춰 Conv2D와 MaxPooling2D 기반 CNN을 구성한다.
model = keras.Sequential(
    [
        layers.Input(shape=(32, 32, 3)),  # CIFAR-10 이미지는 32x32 RGB 이미지이므로 입력 크기를 이에 맞춘다.
        layers.Conv2D(32, (3, 3), activation="relu",),  # 첫 번째 합성곱층에서 저수준 특징을 추출
        layers.MaxPooling2D((2, 2)),  # 공간 해상도를 절반으로 줄여 계산량을 낮추고 중요한 특징만 남김
        layers.Conv2D(128, (3, 3), activation="relu",),  # 두 번째 합성곱층에서 더 복잡한 시각 패턴을 학습
        layers.MaxPooling2D((2, 2)), 
        layers.Flatten(),  # 합성곱 특징 맵을 1차원 벡터로 펼쳐 Dense 층에 전달
        layers.Dense(128, activation="relu"),  # 완전연결층에서 분류에 필요한 특징 조합을 학습
        layers.Dropout(0.1),  # dropout으로 과적합을 완화
        layers.Dense(10, activation="softmax"),  # CIFAR-10의 10개 클래스를 확률로 출력
    ]
)

# 학습 방법과 손실 함수, 평가 지표를 설정한다.
model.compile( 
optimizer="adam",  # Adam 옵티마이저를 사용
    loss="sparse_categorical_crossentropy",  # 정답 라벨이 정수 형태이므로 sparse categorical crossentropy를 사용
    metrics=["accuracy"],  # 분류 정확도를 함께 기록해 학습 성능을 확인
)

# 학습 데이터를 사용해 모델 파라미터를 업데이트한다.
history = model.fit(
    x_train,
    y_train,
    epochs=5,  # 5회 반복 학습한다.
    batch_size=512,  # 한 번에 512장 배치로 묶는다.
    validation_split=0.1,  # 학습 데이터의 0.1을 검증용으로 과적합 여부를 확인
)
# 학습이 끝난 모델을 테스트 세트에서 평가한다.
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)  

dog_image_bgr = cv.imread(str(dog_image_path))  # 예측에 사용할 dog.jpg 이미지를 BGR 형식으로 읽어 온다.

dog_image_rgb = cv.cvtColor(dog_image_bgr, cv.COLOR_BGR2RGB)  # Matplotlib에서 올바른 색으로 보이도록 RGB로 변환한다.
dog_image_resized = cv.resize(dog_image_rgb, (32, 32))  # CIFAR-10 입력 크기에 맞추기 위해 dog 이미지를 32x32로 축소한다.
dog_image_input = dog_image_resized.astype("float32") / 255.0  # 학습 데이터와 동일한 분포를 맞추기 위해 0~1 범위로 정규화한다.
dog_image_input = np.expand_dims(dog_image_input, axis=0)  # 모델 예측 입력 형식에 맞게 배치 차원을 하나 추가한다.

# 요구사항: 테스트 이미지(dog.jpg)에 대한 예측을 수행
dog_probabilities = model.predict(dog_image_input, verbose=0)[0]  # dog 이미지 한 장에 대한 클래스별 예측 확률을 계산한다.
predicted_index = int(np.argmax(dog_probabilities))  # 가장 높은 확률을 가진 클래스 번호를 최종 예측 결과로 선택한다.
predicted_label = class_names[predicted_index]  # 예측된 클래스 번호를 사람이 읽을 수 있는 클래스 이름으로 변환한다.
top_indices = np.argsort(dog_probabilities)[::-1][:5]  # 확률이 높은 상위 5개 클래스를 내림차순으로 고른다.
top_labels = [class_names[index] for index in top_indices]  # 상위 5개 클래스 번호를 이름으로 변환한다.
top_probabilities = dog_probabilities[top_indices]  # 상위 5개 클래스의 확률 값만 따로 추린다.

print("CIFAR-10 CNN 학습 결과\n")
print(f"- 학습 샘플 수: {len(x_train)}\n")
print(f"- 테스트 샘플 수: {len(x_test)}\n")
print(f"- 테스트 손실: {test_loss:.4f}\n")
print(f"- 테스트 정확도: {test_accuracy:.4f}\n")
print(f"- dog.jpg 예측 클래스: {predicted_label}\n")
for rank, index in enumerate(top_indices, start=1):
    print(f"  {rank}. {class_names[index]}: {dog_probabilities[index]:.4f}\n")

figure = plt.figure(figsize=(18, 10))
grid = figure.add_gridspec(2, 2) 

accuracy_axis = figure.add_subplot(grid[0, 0])
accuracy_axis.plot(history.history["accuracy"], label="Train Accuracy")
accuracy_axis.plot(history.history["val_accuracy"], label="Validation Accuracy")
accuracy_axis.set_title("CIFAR-10 CNN Accuracy")
accuracy_axis.set_xlabel("Epoch")
accuracy_axis.set_ylabel("Accuracy")
accuracy_axis.legend()

loss_axis = figure.add_subplot(grid[0, 1])
loss_axis.plot(history.history["loss"], label="Train Loss")
loss_axis.plot(history.history["val_loss"], label="Validation Loss")
loss_axis.set_title("CIFAR-10 CNN Loss")
loss_axis.set_xlabel("Epoch")
loss_axis.set_ylabel("Loss")
loss_axis.legend()

dog_axis = figure.add_subplot(grid[1, 0])
dog_axis.imshow(dog_image_rgb)
dog_axis.set_title(f"dog.jpg Prediction: {predicted_label}")

probability_axis = figure.add_subplot(grid[1, 1])
probability_axis.bar(top_labels, top_probabilities, color="steelblue")
probability_axis.set_title("Top-5 Prediction Probabilities")
probability_axis.set_ylabel("Probability")
probability_axis.set_ylim(0.0, 1.0)

figure.suptitle(f"CIFAR-10 CNN Result - Test Accuracy: {test_accuracy:.4f}", fontsize=16)
figure.tight_layout()

output_path = output_dir / "01_mnist_result.png"
figure.savefig(output_path)
plt.show()
plt.close(figure)
```

- **CIFAR-10 정규화**

RGB 픽셀 값을 `255.0`으로 나누어 0~1 범위로 맞추면 학습 초반 손실이 안정적으로 줄어드는 데 도움이 됩니다.

- **CNN 구조**

PDF 힌트대로 `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense` 레이어를 사용해 간단한 합성곱 신경망을 구성했습니다.

