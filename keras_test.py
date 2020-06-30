#!/usr/local/bin/python3

import numpy as np
from keras.utils import to_categorical
from keras import models, layers
from keras.preprocessing.text import one_hot
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

y_true = np.array([[0, 1, 1], [1, 0, 0]])
y_pred = np.array([[0, 1, 0], [1, 1, 0]])
print(f1_score(y_true, y_pred, average='macro'))
print(f1_score(y_true, y_pred, average='micro'))

# MAXLEN = 5

# model = models.Sequential()
# model.add(layers.Embedding(100, 100,input_length=MAXLEN))
# model.add(layers.Bidirectional(layers.LSTM(128)))
# model.add(layers.Dropout(0.2))
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(28, activation='sigmoid'))
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# print(model.summary())
# print('trained embedding shape:', model.layers[0].get_weights()[0].shape)

# pred = model.predict(input)
# pred = np.round(model.predict(input))
