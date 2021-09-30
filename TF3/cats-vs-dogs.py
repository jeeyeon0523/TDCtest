# ======================================================================
# There are 5 questions in this test with increasing difficulty from 1-5
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score much less
# than your Category 5 question.
# ======================================================================
#
# Computer Vision with CNNs
#
# For this exercise you will build a cats v dogs classifier
# using the Cats v Dogs dataset from TFDS.
# Be sure to use the final layer as shown
#     (Dense, 2 neurons, softmax activation)
#
# The testing infrastructre will resize all images to 224x224
# with 3 bytes of color depth. Make sure your input layer trains
# images to that specification, or the tests will fail.
#
# Make sure your output layer is exactly as specified here, or the
# tests will fail.

# =========== 합격 기준 가이드라인 공유 ============= #
# val_loss 기준에 맞춰 주시는 것이 훨씬 더 중요 #
# val_loss 보다 조금 높아도 상관없음. (언저리까지 OK) #
# =================================================== #
# 문제명: Category 3 - cats vs dogs
# val_loss: 0.3158
# val_acc: 0.8665
# =================================================== #
# =================================================== #


import tensorflow_datasets as tfds
import tensorflow as tf

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import VGG16

dataset_name = 'cats_vs_dogs'

# 기존 코드 대신 사용
train_dataset = tfds.load(name=dataset_name, split='train[:80%]')
valid_dataset = tfds.load(name=dataset_name, split='train[80%:]')


def preprocess(data):
    # YOUR CODE HERE
    x = data['image']
    y = data['label']
    x = x / 255
    x = tf.image.resize(x, size=(224, 224))

    return x, y


def solution_model():
    batch_size = 32
    train_data = train_dataset.map(preprocess).batch(batch_size)
    valid_data = valid_dataset.map(preprocess).batch(batch_size)

    transfer_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    transfer_model.trainable = False

    model = Sequential([
        transfer_model,
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dense(128, activation='relu'),

        # YOUR CODE HERE, BUT MAKE SURE YOUR LAST LAYER HAS 2 NEURONS ACTIVATED BY SOFTMAX
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.summary()

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

    checkpoint_path = "my_checkpoint.ckpt"
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                 save_weights_only=True,
                                 save_best_only=True,
                                 monitor='val_loss',
                                 verbose=1)

    model.fit(train_data,
              validation_data=(valid_data),
              epochs=20,
              callbacks=[checkpoint],
              )

    model.load_weights(checkpoint_path)

    return model


# Note that you'll need to save your model as a .h5 like this
# This .h5 will be uploaded to the testing infrastructure
# and a score will be returned to you
if __name__ == '__main__':
    model = solution_model()
    model.save("TF3-cats-vs-dogs.h5")
