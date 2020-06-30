#!/usr/local/bin/python3
import os
import sys
from keras import models
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.metrics import f1_score
import pickle, json
import utils
from keras import optimizers
import keras.backend as K

en_train_dir = './CLEAN_DATA/en_train.tsv'
en_dev_dir = './CLEAN_DATA/en_dev.tsv'
en_test_dir = './CLEAN_DATA/en_test.tsv'
en_train_texts, en_train_labels = utils.load_data(en_train_dir)
en_dev_texts, en_dev_labels = utils.load_data(en_dev_dir)
en_test_texts, en_test_labels = utils.load_data(en_test_dir)

de_train_dir = './CLEAN_DATA/de_train.tsv'
de_dev_dir = './CLEAN_DATA/de_dev.tsv'
de_test_dir = './CLEAN_DATA/de_dev.tsv'
de_train_texts, de_train_labels = utils.load_data(de_train_dir)
de_dev_texts, de_dev_labels = utils.load_data(de_dev_dir)
de_test_texts, de_test_labels = utils.load_data(de_test_dir)

es_train_dir = './CLEAN_DATA/es_train.tsv'
es_dev_dir = './CLEAN_DATA/es_dev.tsv'
es_test_dir = './CLEAN_DATA/es_test.tsv'
es_train_texts, es_train_labels = utils.load_data(es_train_dir)
es_dev_texts, es_dev_labels = utils.load_data(es_dev_dir)
es_test_texts, es_test_labels = utils.load_data(es_test_dir)

hu_train_dir = './CLEAN_DATA/hu_train.tsv'
hu_dev_dir = './CLEAN_DATA/hu_dev.tsv'
hu_test_dir = './CLEAN_DATA/hu_test.tsv'
hu_train_texts, hu_train_labels = utils.load_data(hu_train_dir)
hu_dev_texts, hu_dev_labels = utils.load_data(hu_dev_dir)
hu_test_texts, hu_test_labels = utils.load_data(hu_test_dir)

sk_train_dir = './CLEAN_DATA/sk_train.tsv'
sk_dev_dir = './CLEAN_DATA/sk_dev.tsv'
sk_test_dir = './CLEAN_DATA/sk_test.tsv'
sk_train_texts, sk_train_labels = utils.load_data(sk_train_dir)
sk_dev_texts, sk_dev_labels = utils.load_data(sk_dev_dir)
sk_test_texts, sk_test_labels = utils.load_data(sk_test_dir)

sv_train_dir = './CLEAN_DATA/sv_train.tsv'
sv_dev_dir = './CLEAN_DATA/sv_dev.tsv'
sv_test_dir = './CLEAN_DATA/sv_test.tsv'
sv_train_texts, sv_train_labels = utils.load_data(sv_train_dir)
sv_dev_texts, sv_dev_labels = utils.load_data(sv_dev_dir)
sv_test_texts, sv_test_labels = utils.load_data(sv_test_dir)

it_train_dir = './CLEAN_DATA/it_train.tsv'
it_dev_dir = './CLEAN_DATA/it_dev.tsv'
it_test_dir = './CLEAN_DATA/it_test.tsv'
it_train_texts, it_train_labels = utils.load_data(it_train_dir)
it_dev_texts, it_dev_labels = utils.load_data(it_dev_dir)
it_test_texts, it_test_labels = utils.load_data(it_test_dir)

pt_train_dir = './CLEAN_DATA/pt_train.tsv'
pt_dev_dir = './CLEAN_DATA/pt_dev.tsv'
pt_test_dir = './CLEAN_DATA/pt_test.tsv'
pt_train_texts, pt_train_labels = utils.load_data(pt_train_dir)
pt_dev_texts, pt_dev_labels = utils.load_data(pt_dev_dir)
pt_test_texts, pt_test_labels = utils.load_data(pt_test_dir)

# MAX_WORDS = 30000
MAXLEN = 25

tokenizer = Tokenizer()
tokenizer.fit_on_texts(en_train_texts + en_dev_texts + en_test_texts + de_train_texts + de_dev_texts + de_test_texts + es_train_texts + es_dev_texts + es_test_texts + hu_train_texts + hu_dev_texts + hu_test_texts + sk_train_texts + sk_dev_texts + sk_test_texts + sv_train_texts + sv_dev_texts + sv_test_texts + it_train_texts + it_dev_texts + it_test_texts + pt_train_texts + pt_dev_texts + pt_test_texts)

vocab_size = len(tokenizer.word_index) + 1  # +UNK
print('unique tokens in tokenizer: ' + str(vocab_size - 1))

print('transforming into vectors...')
en_train_sequences = tokenizer.texts_to_sequences(en_train_texts)
en_dev_sequences = tokenizer.texts_to_sequences(en_dev_texts)
en_test_sequences = tokenizer.texts_to_sequences(en_test_texts)

de_train_sequences = tokenizer.texts_to_sequences(de_train_texts)
de_dev_sequences = tokenizer.texts_to_sequences(de_dev_texts)
de_test_sequences = tokenizer.texts_to_sequences(de_test_texts)

es_train_sequences = tokenizer.texts_to_sequences(es_train_texts)
es_dev_sequences = tokenizer.texts_to_sequences(es_dev_texts)
es_test_sequences = tokenizer.texts_to_sequences(es_test_texts)

hu_train_sequences = tokenizer.texts_to_sequences(hu_train_texts)
hu_dev_sequences = tokenizer.texts_to_sequences(hu_dev_texts)
hu_test_sequences = tokenizer.texts_to_sequences(hu_test_texts)

sk_train_sequences = tokenizer.texts_to_sequences(sk_train_texts)
sk_dev_sequences = tokenizer.texts_to_sequences(sk_dev_texts)
sk_test_sequences = tokenizer.texts_to_sequences(sk_test_texts)

sv_train_sequences = tokenizer.texts_to_sequences(sv_train_texts)
sv_dev_sequences = tokenizer.texts_to_sequences(sv_dev_texts)
sv_test_sequences = tokenizer.texts_to_sequences(sv_test_texts)

it_train_sequences = tokenizer.texts_to_sequences(it_train_texts)
it_dev_sequences = tokenizer.texts_to_sequences(it_dev_texts)
it_test_sequences = tokenizer.texts_to_sequences(it_test_texts)

pt_train_sequences = tokenizer.texts_to_sequences(pt_train_texts)
pt_dev_sequences = tokenizer.texts_to_sequences(pt_dev_texts)
pt_test_sequences = tokenizer.texts_to_sequences(pt_test_texts)

print('padding to ' + str(MAXLEN) + ' words each...')
en_train_data = pad_sequences(en_train_sequences, maxlen=MAXLEN)
en_dev_data = pad_sequences(en_dev_sequences, maxlen=MAXLEN)
en_test_data = pad_sequences(en_test_sequences, maxlen=MAXLEN)

de_train_data = pad_sequences(de_train_sequences, maxlen=MAXLEN)
de_dev_data = pad_sequences(de_dev_sequences, maxlen=MAXLEN)
de_test_data = pad_sequences(de_test_sequences, maxlen=MAXLEN)

es_train_data = pad_sequences(es_train_sequences, maxlen=MAXLEN)
es_dev_data = pad_sequences(es_dev_sequences, maxlen=MAXLEN)
es_test_data = pad_sequences(es_test_sequences, maxlen=MAXLEN)

hu_train_data = pad_sequences(hu_train_sequences, maxlen=MAXLEN)
hu_dev_data = pad_sequences(hu_dev_sequences, maxlen=MAXLEN)
hu_test_data = pad_sequences(hu_test_sequences, maxlen=MAXLEN)

sk_train_data = pad_sequences(sk_train_sequences, maxlen=MAXLEN)
sk_dev_data = pad_sequences(sk_dev_sequences, maxlen=MAXLEN)
sk_test_data = pad_sequences(sk_test_sequences, maxlen=MAXLEN)

sv_train_data = pad_sequences(sv_train_sequences, maxlen=MAXLEN)
sv_dev_data = pad_sequences(sv_dev_sequences, maxlen=MAXLEN)
sv_test_data = pad_sequences(sv_test_sequences, maxlen=MAXLEN)

it_train_data = pad_sequences(it_train_sequences, maxlen=MAXLEN)
it_dev_data = pad_sequences(it_dev_sequences, maxlen=MAXLEN)
it_test_data = pad_sequences(it_test_sequences, maxlen=MAXLEN)

pt_train_data = pad_sequences(pt_train_sequences, maxlen=MAXLEN)
pt_dev_data = pad_sequences(pt_dev_sequences, maxlen=MAXLEN)
pt_test_data = pad_sequences(pt_test_sequences, maxlen=MAXLEN)

print('processing labels...')
en_train_labels = np.asarray(en_train_labels)
en_dev_labels = np.asarray(en_dev_labels)
en_test_labels = np.asarray(en_test_labels)

de_train_labels = np.asarray(de_train_labels)
de_dev_labels = np.asarray(de_dev_labels)
de_test_labels = np.asarray(de_test_labels)

es_train_labels = np.asarray(es_train_labels)
es_dev_labels = np.asarray(es_dev_labels)
es_test_labels = np.asarray(es_test_labels)

hu_train_labels = np.asarray(hu_train_labels)
hu_dev_labels = np.asarray(hu_dev_labels)
hu_test_labels = np.asarray(hu_test_labels)

sk_train_labels = np.asarray(sk_train_labels)
sk_dev_labels = np.asarray(sk_dev_labels)
sk_test_labels = np.asarray(sk_test_labels)

sv_train_labels = np.asarray(sv_train_labels)
sv_dev_labels = np.asarray(sv_dev_labels)
sv_test_labels = np.asarray(sv_test_labels)

it_train_labels = np.asarray(it_train_labels)
it_dev_labels = np.asarray(it_dev_labels)
it_test_labels = np.asarray(it_test_labels)

pt_train_labels = np.asarray(pt_train_labels)
pt_dev_labels = np.asarray(pt_dev_labels)
pt_test_labels = np.asarray(pt_test_labels)

print('en train data tensor shape = ', en_train_data.shape)
print('en train label tensor shape = ', en_train_labels.shape)
print('en dev data tensor shape = ', en_dev_data.shape)
print('en dev label tensor shape = ', en_dev_labels.shape)
print('en test data tensor shape = ', en_test_data.shape)
print('en test label tensor shape = ', en_test_labels.shape)

print('de train data tensor shape = ', de_train_data.shape)
print('de train label tensor shape = ', de_train_labels.shape)
print('de dev data tensor shape = ', de_dev_data.shape)
print('de dev label tensor shape = ', de_dev_labels.shape)
print('de test data tensor shape = ', de_test_data.shape)
print('de test label tensor shape = ', de_test_labels.shape)

print('es train data tensor shape = ', es_train_data.shape)
print('es train label tensor shape = ', es_train_labels.shape)
print('es dev data tensor shape = ', es_dev_data.shape)
print('es dev label tensor shape = ', es_dev_labels.shape)
print('es test data tensor shape = ', es_test_data.shape)
print('es test label tensor shape = ', es_test_labels.shape)

print('hu train data tensor shape = ', hu_train_data.shape)
print('hu train label tensor shape = ', hu_train_labels.shape)
print('hu dev data tensor shape = ', hu_dev_data.shape)
print('hu dev label tensor shape = ', hu_dev_labels.shape)
print('hu test data tensor shape = ', hu_test_data.shape)
print('hu test label tensor shape = ', hu_test_labels.shape)

print('sk train data tensor shape = ', sk_train_data.shape)
print('sk train label tensor shape = ', sk_train_labels.shape)
print('sk dev data tensor shape = ', sk_dev_data.shape)
print('sk dev label tensor shape = ', sk_dev_labels.shape)
print('sk test data tensor shape = ', sk_test_data.shape)
print('sk test label tensor shape = ', sk_test_labels.shape)

print('sv train data tensor shape = ', sv_train_data.shape)
print('sv train label tensor shape = ', sv_train_labels.shape)
print('sv dev data tensor shape = ', sv_dev_data.shape)
print('sv dev label tensor shape = ', sv_dev_labels.shape)
print('sv test data tensor shape = ', sv_test_data.shape)
print('sv test label tensor shape = ', sv_test_labels.shape)

print('it train data tensor shape = ', it_train_data.shape)
print('it train label tensor shape = ', it_train_labels.shape)
print('it dev data tensor shape = ', it_dev_data.shape)
print('it dev label tensor shape = ', it_dev_labels.shape)
print('it test data tensor shape = ', it_test_data.shape)
print('it test label tensor shape = ', it_test_labels.shape)

print('pt train data tensor shape = ', pt_train_data.shape)
print('pt train label tensor shape = ', pt_train_labels.shape)
print('pt dev data tensor shape = ', pt_dev_data.shape)
print('pt dev label tensor shape = ', pt_dev_labels.shape)
print('pt test data tensor shape = ', pt_test_data.shape)
print('pt test label tensor shape = ', pt_test_labels.shape)

en_train_data, en_train_labels = utils.shuffle(en_train_data, en_train_labels)
en_dev_data, en_dev_labels = utils.shuffle(en_dev_data, en_dev_labels)
en_test_data, en_test_labels = utils.shuffle(en_test_data, en_test_labels)

de_train_data, de_train_labels = utils.shuffle(de_train_data, de_train_labels)
de_dev_data, de_dev_labels = utils.shuffle(de_dev_data, de_dev_labels)
de_test_data, de_test_labels = utils.shuffle(de_test_data, de_test_labels)

es_train_data, es_train_labels = utils.shuffle(es_train_data, es_train_labels)
es_dev_data, es_dev_labels = utils.shuffle(es_dev_data, es_dev_labels)
es_test_data, es_test_labels = utils.shuffle(es_test_data, es_test_labels)

hu_train_data, hu_train_labels = utils.shuffle(hu_train_data, hu_train_labels)
hu_dev_data, hu_dev_labels = utils.shuffle(hu_dev_data, hu_dev_labels)
hu_test_data, hu_test_labels = utils.shuffle(hu_test_data, hu_test_labels)

sk_train_data, sk_train_labels = utils.shuffle(sk_train_data, sk_train_labels)
sk_dev_data, sk_dev_labels = utils.shuffle(sk_dev_data, sk_dev_labels)
sk_test_data, sk_test_labels = utils.shuffle(sk_test_data, sk_test_labels)

sv_train_data, sv_train_labels = utils.shuffle(sv_train_data, sv_train_labels)
sv_dev_data, sv_dev_labels = utils.shuffle(sv_dev_data, sv_dev_labels)
sv_test_data, sv_test_labels = utils.shuffle(sv_test_data, sv_test_labels)

it_train_data, it_train_labels = utils.shuffle(it_train_data, it_train_labels)
it_dev_data, it_dev_labels = utils.shuffle(it_dev_data, it_dev_labels)
it_test_data, it_test_labels = utils.shuffle(it_test_data, it_test_labels)

pt_train_data, pt_train_labels = utils.shuffle(pt_train_data, pt_train_labels)
pt_dev_data, pt_dev_labels = utils.shuffle(pt_dev_data, pt_dev_labels)
pt_test_data, pt_test_labels = utils.shuffle(pt_test_data, pt_test_labels)

x_train = en_train_data
# x_train = np.concatenate((en_train_data, de_train_data, es_train_data, hu_train_data, sk_train_data, sv_train_data, it_train_data, pt_train_data))
y_train = en_train_labels
# y_train = np.concatenate((en_train_labels, de_train_labels, es_train_labels, hu_train_labels, sk_train_labels, sv_train_labels, it_train_labels, pt_train_labels))

x_test_en = en_test_data
y_test_en = en_test_labels
x_test_de = de_test_data
y_test_de = de_test_labels
x_test_es = es_test_data
y_test_es = es_test_labels
x_test_hu = hu_test_data
y_test_hu = hu_test_labels
x_test_sk = sk_test_data
y_test_sk = sk_test_labels
x_test_sv = sv_test_data
y_test_sv = sv_test_labels
x_test_it = it_test_data
y_test_it = it_test_labels
x_test_pt = pt_test_data
y_test_pt = pt_test_labels

# tests
print(x_train[:3])
print(x_test_en[:3])

EMBEDDING_DIM = 300
embeddings_index = utils.load_embs_2_dict('EN_DE_ES_HU_SK_SV_IT_PT.txt', dim=EMBEDDING_DIM)
embedding_matrix = utils.build_emb_matrix(num_embedding_vocab=vocab_size, embedding_dim=EMBEDDING_DIM, word_index=tokenizer.word_index, embeddings_index=embeddings_index)

global_en_mic_train = 0
global_de_mic_train = 0
global_en_mac_train = 0
global_de_mac_train = 0
global_en_mic_tune = 0
global_de_mic_tune = 0
global_en_mac_tune = 0
global_de_mac_tune = 0
num_iterations = 8

for i in range(num_iterations):
    print('training iteration:', i + 1)

    # build model
    model = models.Sequential()
    # model.add(layers.Embedding(vocab_size, EMBEDDING_DIM, input_length=MAXLEN))
    model.add(layers.Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False, input_length=MAXLEN))
    # model.add(layers.Conv1D(128, 3, padding='valid', activation='relu'))
    # model.add(layers.MaxPooling1D())
    # model.add(layers.Flatten())
    model.add(layers.Bidirectional(layers.LSTM(128)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(28, activation='sigmoid'))
    # Adam = optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())
    print(K.eval(model.optimizer.lr))
    es = EarlyStopping(monitor='val_loss', mode='auto', min_delta=0, patience=5, restore_best_weights=True, verbose=1)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='auto', verbose=1, save_best_only=True)
    model.fit(x_train, y_train, validation_data=(en_dev_data, en_dev_labels), batch_size=64, epochs=1000, shuffle=True, callbacks=[es, mc])
    # print('trained embedding shape:', model.layers[0].get_weights()[0].shape)

    gold_en = y_test_en
    predicted_en = np.round(model.predict(x_test_en))
    gold_de = y_test_de
    predicted_de = np.round(model.predict(x_test_de))

    en_mic, de_mic, en_mac, de_mac = utils.test_evaluation(gold_en, predicted_en, gold_de, predicted_de)
    global_en_mic_train += en_mic
    global_de_mic_train += de_mic
    global_en_mac_train += en_mac
    global_de_mac_train += de_mac

    # de fine-tuning
    FINETUNE = True
    if FINETUNE:
        print('train:', de_train_dir)
        print('dev:', de_dev_dir)
        model = models.load_model('best_model.h5', compile=True)
        # model.layers[0].trainable = True
        # Adam = optimizers.Adam(learning_rate=0.0001)
        # model.compile(optimizer=Adam, loss='binary_crossentropy', metrics=['acc'])
        print(model.summary())
        print(K.eval(model.optimizer.lr))
        es = EarlyStopping(monitor='val_loss', mode='auto', min_delta=0, patience=5, restore_best_weights=True, verbose=1)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='auto', verbose=1, save_best_only=True, save_weights_only=False)
        model.fit(de_train_data, de_train_labels, validation_data=(de_dev_data, de_dev_labels), batch_size=64, epochs=1000, shuffle=True, callbacks=[es, mc])

        gold_en = y_test_en
        predicted_en = np.round(model.predict(x_test_en))
        gold_de = y_test_de
        predicted_de = np.round(model.predict(x_test_de))

        en_mic, de_mic, en_mac, de_mac = utils.test_evaluation(gold_en, predicted_en, gold_de, predicted_de)
        global_en_mic_tune += en_mic
        global_de_mic_tune += de_mic
        global_en_mac_tune += en_mac
        global_de_mac_tune += de_mac

print()
print('AVG OF', num_iterations, 'TRAIN-ITERATIONS')
en_micro_train = round( (global_en_mic_train/num_iterations), 4)
de_micro_train = round( (global_de_mic_train/num_iterations), 4)
en_macro_train = round( (global_en_mac_train/num_iterations), 4)
de_macro_train = round( (global_de_mac_train/num_iterations), 4)
print('{0: <10}'.format('En-micro') + '\t' + '{0: <10}'.format('De-micro') + '\t' + '{0: <10}'.format('En-macro') + '\t' + '{0: <10}'.format('De-macro'))
print('{0: <10}'.format(en_micro_train) + '\t' + '{0: <10}'.format(de_micro_train) + '\t' + '{0: <10}'.format(en_macro_train) + '\t' + '{0: <10}'.format(de_macro_train))

if FINETUNE:
    print('AVG OF', num_iterations, 'TUNE-ITERATIONS')
    en_micro_tune = round( (global_en_mic_tune/num_iterations), 4)
    de_micro_tune = round( (global_de_mic_tune/num_iterations), 4)
    en_macro_tune = round( (global_en_mac_tune/num_iterations), 4)
    de_macro_tune = round( (global_de_mac_tune/num_iterations), 4)
    print('{0: <10}'.format('En-micro') + '\t' + '{0: <10}'.format('De-micro') + '\t' + '{0: <10}'.format('En-macro') + '\t' + '{0: <10}'.format('De-macro'))
    print('{0: <10}'.format(en_micro_tune) + '\t' + '{0: <10}'.format(de_micro_tune) + '\t' + '{0: <10}'.format(en_macro_tune) + '\t' + '{0: <10}'.format(de_macro_tune))

# plot results
# utils.plot(history)