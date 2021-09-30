# ======================================================================
# There are 5 questions in this test with increasing difficulty from 1-5
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score much less
# than your Category 5 question.
# ======================================================================
#
# NLP QUESTION
#
# For this task you will build a classifier for the sarcasm dataset
# The classifier should have a final layer with 1 neuron activated by sigmoid as shown
# It will be tested against a number of sentences that the network hasn't previously seen
# And you will be scored on whether sarcasm was correctly detected in those sentences

# =========== 합격 기준 가이드라인 공유 ============= #
# val_loss 기준에 맞춰 주시는 것이 훨씬 더 중요 #
# val_loss 보다 조금 높아도 상관없음. (언저리까지 OK) #
# =================================================== #
# 문제명: Category 4 - sarcasm
# val_loss: 0.3650
# val_acc: 0.83
# =================================================== #
# =================================================== #


import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint


def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, 'sarcasm.json')

    # DO NOT CHANGE THIS CODE OR THE TESTS MAY NOT WORK
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE

    # 1. json파일 로드
    with open('sarcasm.json') as f:
        datas = json.load(f)

    # 2. 데이터넷 구성(sentences, labels)
    for data in datas:
        sentences.append(data['headline'])
        labels.append(data['is_sarcastic'])

    # 3. Train/Vaild Set 분리
    train_sentences = sentences[:training_size]
    train_labels = labels[:training_size]

    validation_sentences = sentences[training_size:]
    validation_labels = labels[training_size:]

    # 4-1) Tokenizer 정의
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    # 4-2) Tokenizer로 학습시킬 문장에 대한 토큰화 진행
    tokenizer.fit_on_texts(train_sentences)
    # 4-3) 문장(sentences)을 토큰으로 변경
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    validation_sequences = tokenizer.texts_to_sequences(validation_sentences)

    # 5. 시퀀스의 길이 맞추기
    train_padded = pad_sequences(train_sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type)
    validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type,
                                      truncating=trunc_type)

    # 6. label값을 numpy array로 변환
    train_labels = np.array(train_labels)
    validation_labels = np.array(validation_labels)

    # 7. 모델정의
    model = tf.keras.Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),  # 차원을 줄여줌
        Bidirectional(LSTM(64, return_sequences=True)),  # 겹쳐진부분
        Bidirectional(LSTM(64)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        # YOUR CODE HERE. KEEP THIS OUTPUT LAYER INTACT OR TESTS MAY FAIL
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.summary()

    # 8. 컴파일
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    # 9. 체크포인트 생성
    checkpoint_path = 'my_checkpoint.ckpt'
    checkpoint = ModelCheckpoint(checkpoint_path,
                                 save_weights_only=True,
                                 save_best_only=True,
                                 monitor='val_loss',
                                 verbose=1)

    # 10. 학습
    epochs = 10
    model.fit(train_padded, train_labels,
              validation_data=(validation_padded, validation_labels),
              callbacks=[checkpoint],
              epochs=epochs)

    # 11. load weights
    model.load_weights(checkpoint_path)

    return model


# Note that you'll need to save your model as a .h5 like this
# This .h5 will be uploaded to the testing infrastructure
# and a score will be returned to you
if __name__ == '__main__':
    model = solution_model()
    model.save("TF4-sarcasm.h5")

# val_loss : 0.37374 합격했다는 후기 확인
