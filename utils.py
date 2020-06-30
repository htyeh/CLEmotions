import string
import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.utils import to_categorical
from sklearn.metrics import f1_score

def load_data(file_path, scaling=1):
    texts = []
    labels = []
    print('loading ' + file_path + '...')
    with open(file_path) as f:
        lines = f.readlines()
        num_samples = len(lines) * scaling
        for line in lines[:num_samples]:
            text, classes, id = [item.strip() for item in line.split('\t')]
            texts.append(text)
            classes = np.array(classes.split(','))
            categorical = to_categorical(classes, num_classes=28)
            one_hot = categorical.sum(axis=0)
            labels.append(one_hot)
    return texts, labels


def vectorize(texts, labels, max_words, maxlen):
    print('transforming into vectors...')
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    vocab_size = len(tokenizer.word_index) + 1      # UNK
    print('unique tokens found: ' + str(vocab_size - 1) + ', using most frequent ' + str(max_words))
    print('padding to ' + str(maxlen) + ' words each...')
    data = pad_sequences(sequences, maxlen=maxlen)  # all reviews cut/padded to MAXLEN (default at the front)
    labels = np.asarray(labels)
    print('data tensor shape = ', data.shape)
    print('label tensor shape = ', labels.shape)
    return data, labels, vocab_size
# usage: data, labels, vocab_size = vectorize(texts, labels, MAX_WORDS, MAXLEN)

def shuffle(data, labels):
    print('shuffling...')
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    return data, labels

def load_embs_2_dict(path, dim=100):
    print('loading ' + path)
    embeddings_index = {}
    with open(path) as f:
        for line in tqdm.tqdm(f.readlines(), desc="Loading", unit='embedding'):
            values = line.split()
            if len(values[1:]) == dim:
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
    print('found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def build_emb_matrix(num_embedding_vocab, embedding_dim, word_index, embeddings_index):
    print('building embedding matrix with %s words...' % num_embedding_vocab)
    embedding_matrix = np.zeros((num_embedding_vocab, embedding_dim))
    for word, i in word_index.items():
        if i >= num_embedding_vocab:        # leave out if word too rare
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:    # otherwise embedding = all 0
            embedding_matrix[i] = embedding_vector
    print('loaded pre-trained embeddings shape = ', embedding_matrix.shape)
    return embedding_matrix

def save_embs_2_file(model, emblayer_index, word_index, path='trained_embs.txt'):
    emb_weights = model.layers[emblayer_index].get_weights()[0]
    word2emb = {word:emb_weights[index] for word, index in word_index.items()}
    vocab_size = len(word2emb)  # NOT the same as vocab_size in main: does not include UNK
    embedding_dim = len(emb_weights[0])
    with open(path, 'w') as output:
        output.write(str(vocab_size) + ' ' + str(embedding_dim) + '\n')
        for word, emb in word2emb.items():
            output.write(word + ' ' + ' '.join([str(item) for item in emb]) + '\n')

def test_evaluation(gold_en, predicted_en, gold_de, predicted_de):
    # gold = y_test; predicted = model.predict(x_test)
    print('sample en gold:', gold_en[:10])
    print('sample en pred:', predicted_en[:10])
    print('sample de gold:', gold_de[:10])
    print('sample de pred:', predicted_de[:10])
    en_micro = round(f1_score(gold_en, predicted_en, average='micro'), 4)
    de_micro = round(f1_score(gold_de, predicted_de, average='micro'), 4)
    en_macro = round(f1_score(gold_en, predicted_en, average='macro'), 4)
    de_macro = round(f1_score(gold_de, predicted_de, average='macro'), 4)
    print('{0: <10}'.format('En-micro') + '\t' + '{0: <10}'.format('De-micro') + '\t' + '{0: <10}'.format('En-macro') + '\t' + '{0: <10}'.format('De-macro'))
    print('{0: <10}'.format(en_micro) + '\t' + '{0: <10}'.format(de_micro) + '\t' + '{0: <10}'.format(en_macro) + '\t' + '{0: <10}'.format(de_macro))
    return en_micro, de_micro, en_macro, de_macro

# def multiling_evaluation(gold_en, predicted_en, gold_de, predicted_de):
def multiling_evaluation(gold_list, pred_list):
    # gold = y_test; predicted = model.predict(x_test)
    # print('sample en gold:', gold_en[:30])
    # print('sample en pred:', predicted_en[:30])
    # print('sample de gold:', gold_de[:30])
    # print('sample de pred:', predicted_de[:30])
    micro_list = []
    macro_list = []
    for gold, pred in zip(gold_list, pred_list):
        micro_score = round(f1_score(gold, pred, average='micro'), 4)
        macro_score = round(f1_score(gold, pred, average='macro'), 4)
        micro_list.append(micro_score)
        macro_list.append(macro_score)
    print('micro F1s', micro_list)
    print('macro F1s', macro_list)
    return micro_list, macro_list



def list_layers(model):
    print('listing layers...')
    for i, layer in enumerate(model.layers):
        print(i, layer)

def plot(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# def f1(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

# test_evaluation (old)
# def test_evaluation(gold, predicted):
#     # gold = y_test; predicted = model.predict(x_test)
#     print('sample gold:', gold[:30])
#     print('sample pred:', predicted[:30])
#     true_positives = {0: 0, 1: 0, 2: 0}
#     false_positives = {0: 0, 1: 0, 2: 0}
#     false_negatives = {0: 0, 1: 0, 2: 0}
#     for i, pred_label in enumerate(predicted):
#         if pred_label == gold[i]:
#             true_positives[pred_label] += 1
#         else:
#             false_positives[pred_label] += 1
#             false_negatives[gold[i]] += 1
#     # macro param.
#     precisions = {}
#     recalls = {}
#     f1s = {}
#     for i in [0, 1, 2]:
#         try:
#             precision = true_positives[i] / (true_positives[i] + false_positives[i])
#             recall = true_positives[i] / (true_positives[i] + false_negatives[i])
#         except ZeroDivisionError:
#             precision = 0.00001
#             recall = 0.00001
#         precisions[i] = precision
#         recalls[i] = recall
#         f1s[i] = 2 * (precision * recall) / (precision + recall)
#     macro_precision = sum(precisions.values()) / 3
#     macro_recall = sum(recalls.values()) / 3
#     macro_f1 = sum(f1s.values()) / 3
#     # micro param.
#     try:
#         micro_precision = sum(true_positives.values()) / (sum(true_positives.values()) + sum(false_positives.values()))
#         micro_recall = sum(true_positives.values()) / (sum(true_positives.values()) + sum(false_negatives.values()))
#     except ZeroDivisionError:
#         micro_precision = 0.00001
#         micro_recall = 0.00001
#     micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
#     # print('macro precision:', macro_precision)
#     # print('macro recall:', macro_recall)
#     print('macro F1:', macro_f1)
#     # print('micro precision:', micro_precision)
#     # print('micro recall:', micro_recall)
#     print('micro F1:', micro_f1)