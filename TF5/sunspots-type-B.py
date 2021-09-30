# QUESTION
#
# For this task you will need to train a neural network
# to predict sunspot activity using the Sunspots.csv dataset.
# Your neural network must  have an MAE
# of 0.12 or less on the normalized dataset for top marks.
# Code for normalizing the data is provided and should not be changed.
# At the bottom of this file, we provide  some testing
# code should you want to check your model.

# Note: Do not use lambda layers in your model, they are not supported
# on the grading infrastructure.

# =========== 합격 기준 가이드라인 공유 ============= #
# val_loss 기준에 맞춰 주시는 것이 훨씬 더 중요 #
# val_loss 보다 조금 높아도 상관없음. (언저리까지 OK) #
# =================================================== #
# 문제명: Category 5 - sunspots type B (NO Lambda)
# val_loss: 상관없음
# val_mae: 0.1121
# =================================================== #
# =================================================== #


import csv
import tensorflow as tf
import numpy as np
import urllib

from tensorflow.keras.layers import Dense, LSTM, Lambda, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import Huber


# DO NOT CHANGE THIS CODE
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/Sunspots.csv'
    urllib.request.urlretrieve(url, 'sunspots.csv')

    time_step = []
    sunspots = []

    with open('sunspots.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            sunspots.append(float(row[2]))
            time_step.append(int(row[0]))

    series = np.array(sunspots)
    time = np.array(time_step)

    # DO NOT CHANGE THIS CODE
    # This is the normalization function
    min = np.min(series)
    max = np.max(series)
    series -= min
    series /= max
    time = np.array(time_step)

    # The data should be split into training and validation sets at time step 3000
    # DO NOT CHANGE THIS CODE
    split_time = 3000

    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]

    # DO NOT CHANGE THIS CODE
    window_size = 30
    batch_size = 32
    shuffle_buffer_size = 1000

    train_set = windowed_dataset(x_train, window_size=window_size, batch_size=batch_size,
                                 shuffle_buffer=shuffle_buffer_size)
    validation_set = windowed_dataset(x_valid, window_size=window_size, batch_size=batch_size,
                                      shuffle_buffer=shuffle_buffer_size)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(60, kernel_size=5,
                               padding="causal",
                               activation="relu",
                               input_shape=[None, 1]),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),

        # YOUR CODE HERE. Whatever your first layer is, the input shape will be [None,1] when using the Windowed_dataset above, depending on the layer type chosen
        tf.keras.layers.Dense(1)
    ])

    # YOUR CODE HERE TO COMPILE AND TRAIN THE
    model.summary()

    optimizer = SGD(learning_rate=0.0002, momentum=0.9)
    loss = Huber()

    model.compile(loss=loss, optimizer=optimizer, metrics=["mae"])

    checkpoint_path = 'tmp_checkpoint.ckpt'
    checkpoint = ModelCheckpoint(checkpoint_path,
                                 save_weights_only=True,
                                 save_best_only=True,
                                 monitor='val_mae',
                                 verbose=1)

    epochs = 100

    model.fit(train_set,
              validation_data=(validation_set),
              epochs=epochs,
              callbacks=[checkpoint],
              )

    model.load_weights(checkpoint_path)

    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, this .h5 model will be
# sent to the testing infrastructure for scoring.

# You must use the Submit and Test button to submit your model
# at least once in each category before you finally submit your exam.

if __name__ == '__main__':
    model = solution_model()
    model.save("TF5-sunspots-type-B.h5")

# THIS CODE IS USED IN THE TESTER FOR FORECASTING. IF YOU WANT TO TEST YOUR MODEL
# BEFORE UPLOADING YOU CAN DO IT WITH THIS
# def model_forecast(model, series, window_size):
#    ds = tf.data.Dataset.from_tensor_slices(series)
#    ds = ds.window(window_size, shift=1, drop_remainder=True)
#    ds = ds.flat_map(lambda w: w.batch(window_size))
#    ds = ds.batch(32).prefetch(1)
#    forecast = model.predict(ds)
#    return forecast


# window_size = # YOUR CODE HERE
# rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
# rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]

# result = tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()

## To get the maximum score, your model must have an MAE OF .12 or less.
## When you Submit and Test your model, the grading infrastructure
## converts the MAE of your model to a score from 0 to 5 as follows:

# test_val = 100 * result
# score = math.ceil(17 - test_val)
# if score > 5:
#    score = 5

# print(score)