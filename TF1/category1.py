# STEP 1: 필요한 모듈 import
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# STEP 2: 데이터 전처리
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=float)

# STEP 3: 모델의 정의 (modeling)
model = Sequential([
    Dense(1, input_shape=[1]),
])

# STEP 4: 모델의 생성 (compile)
model.compile(optimizer='sgd', loss='mse')

# STEP 5: 학습 (fit)
model.fit(xs, ys, epochs=1200, verbose=0)

# 검증 (실제 시험 볼 때는 이 과정은 필요가 없음)
# 16.000046
model.predict([10.0])