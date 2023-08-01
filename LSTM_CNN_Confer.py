import time

import matplotlib
import nltk
import openpyxl
import os
import regex
import pandas as pd

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.layers import Bidirectional, TimeDistributed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from gensim.models import Word2Vec
from math import log
from nltk.tokenize import word_tokenize

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from matplotlib import pyplot
#from networkx.drawing.tests.test_pylab import plt
from wordcloud import WordCloud

#matplotlib.use('tkagg')

from nltk import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
import numpy as np

from numpy import array


from tensorflow.python.client._pywrap_tf_session import *
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import GlobalMaxPool1D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import f1_score

#
# ###Nltk abstract_words tokenize...........................................................................
# with open('FGFSJournal.txt', 'rt', encoding='UTF8') as file:
#     all_abstract = []
#     for line in file:
#         if '<abstract>' in line:
#             abstract = line.split('</abstract>')[0].split('<abstract>')[-1]
#             # sentences = nltk.sent_tokenize(abstract)
#             #abstract = ''.join(i for i in abstract if not i.isdigit())
#             #abstract = regex.sub('[^\w\d\s]+', '', abstract)
#             abstract = nltk.word_tokenize(abstract)
#             stop_words = set(stopwords.words('english'))
#             filtered_sentence_abstract = [w.lower() for w in abstract if
#                                           w.lower() not in punctuation and w.lower() not in stop_words]
#             tagged_list = nltk.pos_tag(filtered_sentence_abstract)
#             nouns_list = [t[0] for t in tagged_list if t[-1] == 'NN']
#             lm = WordNetLemmatizer()
#             singluar_form = [lm.lemmatize(w, pos='v') for w in nouns_list]
#             all_abstract.append(singluar_form)
#
# #print(len(all_abstract))
# # for i in all_abstract:
# #     print(i)
# #print(max(map(len, all_abstract)))
# #
# # ###Nltk filtered_sentence_keywords_words tokenize...........................................................................
# # with open('FGFSJournal.txt', 'rt', encoding='UTF8') as file:
# #     all_keywords = []
# #     for line in file:
# #         if '<keywords>' in line:
# #             keywords = line.split('</keywords>')[0].split('<keywords>')[-1]
# #             digits = ''.join(i for i in keywords if not i.isdigit())
# #             punchs = regex.sub('[^\w\d\s]+', '', digits)
# #             #abstract = nltk.sent_tokenize(punchs)
# #             tokenize = nltk.word_tokenize(punchs)
# #             stop_words = set(stopwords.words('english'))
# #             filtered_sentence_keywords = [w.lower() for w in tokenize if
# #                                           w.lower() not in punctuation and w.lower() not in stop_words]
# #             tagged_list = nltk.pos_tag(filtered_sentence_keywords)
# #             nouns_list = [t[0] for t in tagged_list if t[-1] == 'NN']
# #             lm = WordNetLemmatizer()
# #             singluar_form = [lm.lemmatize(w, pos='v') for w in nouns_list]
# #             all_keywords.append(abstract)
# #
# # #print(len(all_keywords))
# #
# # ##====================================FGFS_labels====================================
# # select_words = ['system',  'network', 'approach', 'time', 'cloud', 'information', 'process', 'problem', 'service', 'security']
# # key_labels = []
# # for i in range(0, 3961):
# #     count = 0
# #     for j in range(0, len(select_words)):
# #         if select_words[j] in all_keywords[i]:
# #             count += 1
# #     if count >= 2:
# #         key_labels.append(1)
# #     else:
# #         key_labels.append(0)
# #
# #
# # #print("FGFS_labels:", FGFS_labels)
# #
# # print()
# # token = Tokenizer()  # create the tokenizer
# # token.fit_on_texts(all_keywords)  # fit the tokenizer on the documents
# #
# # max_length = 154
# #
# # key_train, key_test = train_test_split(all_keywords, test_size=0.30, random_state=1)
# #
# # test = token.texts_to_sequences(key_train)
# # key_x_test = pad_sequences(test, maxlen=max_length, padding='post')
# # key_y_test = np.array(key_labels)
#
#
# ##===============================pre-traind word2vec data==========================================
# CBOW_embeddings = Word2Vec(sentences=all_abstract, vector_size=100, window=5, min_count=0, sg=0)
# CBOW_embeddings.wv.save_word2vec_format('CBOW_Pre-trained_word2Vec.txt', binary=False)
#
#
# ##====================================Prepairinng FGFS test data====================================
# with open('TestData_FGFS.txt', 'rt', encoding='UTF8') as file:
#     test_FGFS = []
#     for line in file:
#         if '<abstract>' in line:
#             abstract = line.split('</abstract>')[0].split('<abstract>')[-1]
#             abstract = ''.join(i for i in abstract if not i.isdigit())
#             abstract = regex.sub('[^\w\d\s]+', '', abstract)
#             abstract = nltk.word_tokenize(abstract)
#             stop_words = set(stopwords.words('english'))
#             filtered_sentence_abstract = [w.lower() for w in abstract if
#                                           w.lower() not in punctuation and w.lower() not in stop_words]
#             tagged_list = nltk.pos_tag(filtered_sentence_abstract)
#             nouns_list = [t[0] for t in tagged_list if t[1] == 'NN']
#             lm = WordNetLemmatizer()
#             singluar_form = [lm.lemmatize(w, pos='v') for w in nouns_list]
#             test_FGFS.append(singluar_form)
#
#
# # print(max(map(len, test_FGFS)))
# # print("Test data from FGFS:", test_FGFS)
#
#
#
# ##====================================FGFS_labels====================================
# select_words = ['system',  'network', 'approach', 'time', 'cloud', 'information', 'process', 'problem', 'service', 'security']
# FGFS_labels = []
# for i in range(0, 1132):
#     count = 0
#     for j in range(0, len(select_words)):
#         if select_words[j] in test_FGFS[i]:
#             count += 1
#     if count >= 1:
#         FGFS_labels.append(1)
#     else:
#         FGFS_labels.append(0)
#
#
# #print("FGFS_labels:", FGFS_labels)
#
# print()
# token = Tokenizer()  # create the tokenizer
# token.fit_on_texts(test_FGFS)  # fit the tokenizer on the documents
#
# word_index = token.word_index
# #print('unique words: {}'.format(len(word_index)))
#
# max_length = max(len(l) for l in all_abstract)
#
# test = token.texts_to_sequences(test_FGFS)
# x_test = pad_sequences(test, maxlen=max_length, padding='post')
# y_test = np.asarray(FGFS_labels).astype('float32').reshape((-1, 1)) #np.array(FGFS_labels)
#
#
# #====================================Prepairinng AWR test data====================================
# with open('TestData_AWR.txt', 'rt', encoding='UTF8') as file:
#     test_AWR = []
#     for line in file:
#         if '<abstract>' in line:
#             abstract = line.split('</abstract>')[0].split('<abstract>')[-1]
#             abstract = ''.join(i for i in abstract if not i.isdigit())
#             abstract = regex.sub('https?:\/\/.*[\r\n]*', '', abstract) ## URL 제거
#             abstract = regex.sub('[^\w\d\s]+', '', abstract)
#             abstract = nltk.word_tokenize(abstract)
#             stop_words = set(stopwords.words('english'))
#             filtered_sentence_abstract = [w.lower() for w in abstract if
#                                           w.lower() not in punctuation and w.lower() not in stop_words]
#             tagged_list = nltk.pos_tag(filtered_sentence_abstract)
#             nouns_list = [t[0] for t in tagged_list if t[-1] == 'NN']
#             lm = WordNetLemmatizer()
#             singluar_form = [lm.lemmatize(w, pos='v') for w in nouns_list]
#             test_AWR.append(singluar_form)
#
#
# #print(max(map(len, test_AWR)))
# #print()
# #print("Test data from AWR:", test_AWR)
#
#
# select_words = ['system',  'network', 'approach', 'time', 'cloud', 'information', 'process', 'problem', 'service', 'security']
# AWR_labels = []
# for i in range(0, 100):
#     count = 0
#     for j in range(0, len(select_words)):
#         if select_words[j] in test_AWR[i]:
#             count += 1
#     if count >=2:
#         AWR_labels.append(1)
#     else:
#         AWR_labels.append(0)
# #print("AWR_labels:", AWR_labels)
#
#
# print()
# token = Tokenizer()  # create the tokenizer
# token.fit_on_texts(test_AWR)  # fit the tokenizer on the documents
#
# word_index = token.word_index
# #print('unique words: {}'.format(len(word_index)))
#
#
# max_length = 153
#
# test1 = token.texts_to_sequences(test_AWR)
# x_test_1 = pad_sequences(test1, maxlen=max_length, padding='post')
# y_test_1 = np.asarray(AWR_labels).astype('float32').reshape((-1, 1))# np.array(AWR_labels)
# #print("AWR test label:", y_test_1)
# #
#
#
#
# ##abstract CNN training...................................................
# print("create the tokenizer")
# token = Tokenizer()  # create the tokenizer
# token.fit_on_texts(all_abstract)  # fit the tokenizer on the documents
# #print("Total words:", len(token.word_index))
#
#
# word_index = token.word_index
# #print('unique words: {}'.format(len(word_index)))
#
#
#
# # # print()
# vocab_size = len(token.word_index) + 1  # define vocabulary size (largest integer value)
# #print('Vocabulary size: %d' % vocab_size)
#
#
# #max_length = 259
#
# max_length = max(len(l) for l in all_abstract) # 모든 샘플에서 길이가 가장 긴 샘플의 길이 출력
# #print('샘플의 최대 길이 : {}'.format(max_length))
#
# train, valid = train_test_split(all_abstract, test_size=0.25, random_state=1)
#
# #print("train", len(train))
# #print("valid", len(valid))
#
#
#
# ##====================================train_labels====================================
# select_words = ['system',  'network', 'approach', 'time', 'cloud', 'information', 'process', 'problem', 'service', 'security']
# train_labels = []
# for i in range(0, 3395):
#     count = 0
#     for j in range(0, len(select_words)):
#         if select_words[j] in all_abstract[i]:
#             count += 1
#     if count >=1:
#         train_labels.append(1)
#     else:
#         train_labels.append(0)
#
#
# ##====================================validation labels====================================
# select_words = ['system',  'network', 'approach', 'time', 'cloud', 'information', 'process', 'problem', 'service', 'security']
# valid_labels = []
# for i in range(0, 1132):
#
#     count = 0
#     for j in range(0, len(select_words)):
#         if select_words[j] in valid[i]:
#             count += 1
#     if count >=2:
#         valid_labels.append(1)
#     else:
#         valid_labels.append(0)
#
# #print(valid_labels)
#
#
#
# train_data = token.texts_to_sequences(train)
# valid_data = token.texts_to_sequences(valid)
# # print("integer incode:", data)
# # print("length: ", len(data))
#
#
# X_train = pad_sequences(train_data, maxlen=max_length, padding='post')
# #y_train = np.array(train_labels)
# y_train = np.asarray(train_labels).astype('float32').reshape((-1, 1))
# #print(len(X_train))
#
# X_valid = pad_sequences(valid_data, max_length, padding='post')
# #y_valid = np.array(valid_labels)
# y_valid = np.asarray(valid_labels).astype('float32').reshape((-1, 1))
#
#
#
# ##======================CNNs model with word2vec=====================================
# embedding_index = {}
# list_v = []
# file = open('CBOW_Pre-trained_word2Vec.txt', 'rt', encoding='UTF8')
# line = file.readline()
# totalWords, numOfFeatures = line.split()
# print(totalWords, numOfFeatures)
# for line in file:
#     values = line.split()
#     list_v.append(values)
#     word = values[0]
#     coefs = array(values[1:], dtype='float64')
#     embedding_index[word] = coefs
# #file.close()
#
# print('Found %s word vectors.' % len(embedding_index))
# df_values = pd.DataFrame(list_v)
# print(df_values, "\n")
#
#
# embedding_matrix1 = np.array([[0 for col in range(100)] for row in range(10588)])
#
# for word, i in token.word_index.items():
#     embedding_vector = embedding_index.get(word)
#     if embedding_vector is not None:
#         if( i == 100):
#             print(i,"번째 완료")
#         for j in range(0, 100):
#            embedding_matrix1[i][j] = embedding_vector[j]
#         #print(i,"번째 완료")
#
# #print("embedding_matrix1:", embedding_matrix1)
#
#
# #=======================CBOW_CNNs model using LSTM ====================================================
# embedding_dim = 100
# pooling = 2
# dropout = 0.5
# filters = 128
# epochs = 100
# batch_sizes = 128
#
#
# ##===================LSTM model for Text Classification=====================================================
# lstm_model = Sequential()
# lstm_model.add(Embedding(vocab_size, 100, weights=[embedding_matrix1], input_length=max_length, trainable=True))
# lstm_model.add(LSTM(100, return_sequences=False))
# lstm_model.add(Dropout(0.5))
# lstm_model.add(Dense(1, activation='sigmoid'))
# lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# print(lstm_model.summary())
# file_path = "CBOW_LSTM.h5py"
# checkpoint = ModelCheckpoint(filepath=file_path, monitor='acc', verbose=1, save_best_only=True, mode='max', save_weights_only=True)
# callbacks_list = [checkpoint]
# CBOW_LSTM = lstm_model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=epochs, batch_size=batch_sizes, verbose=1,
#                callbacks=callbacks_list)
# CBOW_LSTM_Train = lstm_model.evaluate(X_train, y_train, verbose=0)
# print(('CBOW_GRU_Train_Score: %f' % (CBOW_LSTM_Train[1] * 100)))
# CBOW_LSTM_FGFS_Test = lstm_model.evaluate(x_test, y_test, verbose=0)
# print(('CBOW_LSTM_FGFS_Test Accuracy: %f' % (CBOW_LSTM_FGFS_Test[1]*100)))
# CBOW_LSTM_AWR_Test = lstm_model.evaluate(x_test_1, y_test_1, verbose=0)
# print(('CBOW_LSTM_AWR_Test Accuracy: %f' % (CBOW_LSTM_AWR_Test[1]*100)))
#
#
# ##====================F_score CBOW_lstm==========================================
# # predict probabilities for test set
# CBOW_lstm_yhat_probs = lstm_model.predict(x_test, verbose=0)
# # reduce to 1d array
# CBOW_lstm_yhat_probs = CBOW_lstm_yhat_probs[:, 0]
#
#
# # accuracy: (tp + tn) / (p + n)
# CBOW_lstm_accuracy = accuracy_score(y_test, np.round(abs(CBOW_lstm_yhat_probs)))
# print('CBOW_lstm_accuracy: %f' % CBOW_lstm_accuracy)
# # precision tp / (tp + fp)
# CBOW_lstm_precision = precision_score(y_test, np.round(abs(CBOW_lstm_yhat_probs)))
# print('CBOW_lstm_precision: %f' % CBOW_lstm_precision)
# # recall: tp / (tp + fn)
# CBOW_lstm_recall = recall_score(y_test, np.round(abs(CBOW_lstm_yhat_probs)))
# print('CBOW_lstm_recall: %f' % CBOW_lstm_recall)
# # f1: 2 tp / (2 tp + fp + fn)
# CBOW_lstm_f1 = f1_score(y_test, np.round(abs(CBOW_lstm_yhat_probs)))
# print('CBOW_lstm_f1: %f' % CBOW_lstm_f1)
#
#
# ##===============================================CBOW GRU model===================================================
# gru_model = Sequential()
# gru_model.add(Embedding(vocab_size, 100, weights=[embedding_matrix1], input_length=max_length, trainable=True))
# gru_model.add(GRU(100, return_sequences=False))
# gru_model.add(Dropout(0.5))
# gru_model.add(Dense(1, activation='sigmoid'))
# gru_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# print(gru_model.summary())
# file_path = "CBOW_GRU.h5py"
# checkpoint = ModelCheckpoint(filepath=file_path, monitor='acc', verbose=1, save_best_only=True, mode='max',
#                              save_weights_only=True)
# callbacks_list = [checkpoint]
# CBOW_GRU = gru_model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=epochs, batch_size=batch_sizes,
#                          verbose=1,
#                        callbacks=callbacks_list)
# CBOW_GRU_Train = gru_model.evaluate(X_train, y_train, verbose=0)
# print(('CBOW_GRU_Train_Score: %f' % (CBOW_GRU_Train[1] * 100)))
# CBOW_GRU_FGFS_Test = gru_model.evaluate(x_test, y_test, verbose=0)
# print(('CBOW_GRU_FGFS_Test Accuracy: %f' % (CBOW_GRU_FGFS_Test[1]*100)))
# CBOW_GRU_AWR_Test = gru_model.evaluate(x_test_1, y_test_1, verbose=0)
# print(('CBOW_GRU_AWR_Test Accuracy: %f' % (CBOW_GRU_AWR_Test[1]*100)))
#
#
# ##====================F_score CBOW_gru==========================================
# # predict probabilities for test set
# CBOW_gru_yhat_probs = gru_model.predict(x_test, verbose=0)
# # reduce to 1d array
# CBOW_gru_yhat_probs = CBOW_gru_yhat_probs[:, 0]
#
#
# # accuracy: (tp + tn) / (p + n)
# CBOW_gru_accuracy = accuracy_score(y_test, np.round(abs(CBOW_gru_yhat_probs)))
# print('CBOW_gru_accuracy: %f' % CBOW_gru_accuracy)
# # precision tp / (tp + fp)
# CBOW_gru_precision = precision_score(y_test, np.round(abs(CBOW_gru_yhat_probs)))
# print('CBOW_gru_precision: %f' % CBOW_gru_precision)
# # recall: tp / (tp + fn)
# CBOW_gru_recall = recall_score(y_test, np.round(abs(CBOW_gru_yhat_probs)))
# print('CBOW_gru_recall: %f' % CBOW_gru_recall)
# # f1: 2 tp / (2 tp + fp + fn)
# CBOW_gru_f1 = f1_score(y_test, np.round(abs(CBOW_gru_yhat_probs)))
# print('CBOW_gru_f1: %f' % CBOW_gru_f1)
#
#
# #=======================CBOW_CNNs model using LSTM ====================================================
# cbow_cnn_lstm_model = Sequential()
# cbow_cnn_lstm_model.add(Embedding(vocab_size, 100, weights=[embedding_matrix1], input_length=max_length,trainable=True))
# cbow_cnn_lstm_model.add(Conv1D(filters=filters, kernel_size=2, padding='same', activation='relu'))
# cbow_cnn_lstm_model.add(MaxPooling1D(pool_size=pooling))
# cbow_cnn_lstm_model.add(LSTM(100, return_sequences=True, recurrent_dropout=dropout))
# cbow_cnn_lstm_model.add(Conv1D(filters=filters, kernel_size=3, padding='same', activation='relu'))
# cbow_cnn_lstm_model.add(MaxPooling1D(pool_size=pooling))
# cbow_cnn_lstm_model.add(LSTM(100, return_sequences=True, recurrent_dropout=dropout))
# cbow_cnn_lstm_model.add(Conv1D(filters=filters, kernel_size=4, padding='same', activation='relu'))
# cbow_cnn_lstm_model.add(MaxPooling1D(pool_size=pooling))
# cbow_cnn_lstm_model.add(LSTM(100, return_sequences=True, recurrent_dropout=dropout))
# cbow_cnn_lstm_model.add(Dense(10, activation='relu'))
# cbow_cnn_lstm_model.add(Dense(1, activation='sigmoid'))
# cbow_cnn_lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# print(cbow_cnn_lstm_model.summary())
#
# file_path = "CBOW_CNN_LSTM.h5py"
# checkpoint = ModelCheckpoint(filepath=file_path, monitor='acc', verbose=1, save_best_only=True, mode='max',
#                              save_weights_only=True)
# callbacks_list = [checkpoint]
# CBOW_CNN_LSTM = cbow_cnn_lstm_model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=epochs,
#                                         batch_size=batch_sizes, verbose=1, callbacks=callbacks_list)
# CBOW_CNN_LSTM_Train = cbow_cnn_lstm_model.evaluate(X_train, y_train, verbose=0)
# print(('CBOW_CNN_LSTM_Train_Score: %f' % (CBOW_CNN_LSTM_Train[1] * 100)))
# CBOW_CNN_LSTM_FGFS_Test = cbow_cnn_lstm_model.evaluate(x_test, y_test, verbose=0)
# print(('CBOW_CNN_LSTM_FGFS_Test Accuracy: %f' % (CBOW_CNN_LSTM_FGFS_Test[1]*100)))
# CBOW_CNN_LSTM_AWR_Test = cbow_cnn_lstm_model.evaluate(x_test_1, y_test_1, verbose=0)
# print(('CBOW_CNN_LSTM_AWR_Test Accuracy: %f' % (CBOW_CNN_LSTM_AWR_Test[1]*100)))
#
#
# ##====================F_score CBOW_cnn_lstm==========================================
# # predict probabilities for test set
# CBOW_cnn_lstm_probs = cbow_cnn_lstm_model.predict(x_test, verbose=0)
# # reduce to 1d array
# CBOW_cnn_lstm_probs = CBOW_cnn_lstm_probs[:, 0]
#
#
# # accuracy: (tp + tn) / (p + n)
# CBOW_cnn_lstm_accuracy = accuracy_score(y_test, np.round(abs(CBOW_cnn_lstm_probs)))
# print('CBOW_cnn_lstm_accuracy: %f' % CBOW_cnn_lstm_accuracy)
# # precision tp / (tp + fp)
# CBOW_cnn_lstm_precision = precision_score(y_test, np.round(abs(CBOW_cnn_lstm_probs)))
# print('CBOW_cnn_lstm_precision: %f' % CBOW_cnn_lstm_precision)
# # recall: tp / (tp + fn)
# CBOW_cnn_lstm_recall = recall_score(y_test, np.round(abs(CBOW_cnn_lstm_probs)))
# print('CBOW_cnn_lstm_recall: %f' % CBOW_cnn_lstm_recall)
# # f1: 2 tp / (2 tp + fp + fn)
# CBOW_cnn_lstm_f1 = f1_score(y_test, np.round(abs(CBOW_cnn_lstm_probs)))
# print('CBOW_cnn_lstm_f1: %f' % CBOW_cnn_lstm_f1)
#
#
# #=======================CBOW_CNNs model using GRU====================================================
# CBOW_CNN_GRU_model = Sequential()
# CBOW_CNN_GRU_model.add(Embedding(vocab_size, 100, weights=[embedding_matrix1], input_length=max_length)) #, trainable=True
# CBOW_CNN_GRU_model.add(Conv1D(filters=filters, kernel_size=2, padding='same', activation='relu'))
# CBOW_CNN_GRU_model.add(MaxPooling1D(pool_size=pooling))
# CBOW_CNN_GRU_model.add(GRU(100, return_sequences=True, recurrent_dropout=dropout))
# CBOW_CNN_GRU_model.add(Conv1D(filters=filters, kernel_size=3, padding='same', activation='relu'))
# CBOW_CNN_GRU_model.add(MaxPooling1D(pool_size=pooling))
# CBOW_CNN_GRU_model.add(GRU(100, return_sequences=True, recurrent_dropout=dropout))
# CBOW_CNN_GRU_model.add(Conv1D(filters=filters, kernel_size=4, padding='same', activation='relu'))
# CBOW_CNN_GRU_model.add(MaxPooling1D(pool_size=pooling))
# CBOW_CNN_GRU_model.add(GRU(100, return_sequences=True, recurrent_dropout=dropout))
# CBOW_CNN_GRU_model.add(Dense(10, activation='relu'))
# CBOW_CNN_GRU_model.add(Dense(1, activation='sigmoid'))
# CBOW_CNN_GRU_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# print(CBOW_CNN_GRU_model.summary())
#
# file_path = "CBOW_CNN_GRU.h5py"
# checkpoint = ModelCheckpoint(filepath=file_path, monitor='acc', verbose=1, save_best_only=True, mode='max',
#                              save_weights_only=True)
# callbacks_list = [checkpoint]
# CBOW_CNN_GRU = CBOW_CNN_GRU_model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=epochs, batch_size=batch_sizes,
#                                    verbose=1,
#                        callbacks=callbacks_list)
# CBOW_CNN_GRU_Trian = CBOW_CNN_GRU_model.evaluate(X_train, y_train, verbose=0)
# print(('Trian Accuracy: %f' % (CBOW_CNN_GRU_Trian[1] * 100)))
# CBOW_CNN_GRU_FGFS_Test = CBOW_CNN_GRU_model.evaluate(x_test, y_test, verbose=0)
# print(('CBOW_CNN_GRU_FGFS_Test Accuracy: %f' % (CBOW_CNN_GRU_FGFS_Test[1]*100)))
# CBOW_CNN_GRU_AWR_Test = CBOW_CNN_GRU_model.evaluate(x_test_1, y_test_1, verbose=0)
# print(('CBOW_CNN_GRU_AWR_Test Accuracy: %f' % (CBOW_CNN_GRU_AWR_Test[1]*100)))
#
#
# ##====================F_score CBOW_cnn_gru==========================================
# # predict probabilities for test set
# CBOW_cnn_gru_probs = CBOW_CNN_GRU_model.predict(x_test, verbose=0)
# # reduce to 1d array
# CBOW_cnn_gru_probs = CBOW_cnn_gru_probs[:, 0]
#
#
# # accuracy: (tp + tn) / (p + n)
# CBOW_cnn_gru_accuracy = accuracy_score(y_test, np.round(abs(CBOW_cnn_gru_probs)))
# print('CBOW_cnn_gru_accuracy: %f' % CBOW_cnn_gru_accuracy)
# # precision tp / (tp + fp)
# CBOW_cnn_gru_precision = precision_score(y_test, np.round(abs(CBOW_cnn_gru_probs)))
# print('CBOW_cnn_gru_precision: %f' % CBOW_cnn_gru_precision)
# # recall: tp / (tp + fn)
# CBOW_cnn_gru_recall = recall_score(y_test, np.round(abs(CBOW_cnn_gru_probs)))
# print('CBOW_cnn_gru_recall: %f' % CBOW_cnn_gru_recall)
# # f1: 2 tp / (2 tp + fp + fn)
# CBOW_cnn_gru_f1 = f1_score(y_test, np.round(abs(CBOW_cnn_gru_probs)))
# print('CBOW_cnn_gru_f1: %f' % CBOW_cnn_gru_f1)
#
#
#
#
# ##=======================CBOW Data visualization====================================================
# pyplot.plot(CBOW_LSTM.history['acc'], color='b',  linewidth='1.7', label='CBOW_LSTM') #linestyle='solid',
# pyplot.plot(CBOW_GRU.history['acc'],  color='y',  linewidth='1.7', label='CBOW_GRU') #linestyle='dashed',
# pyplot.plot(CBOW_CNN_LSTM.history['acc'],  color='g',  linewidth='1.7', label='CBOW_CNN_LSTM') #linestyle='dashdot',
# pyplot.plot(CBOW_CNN_GRU.history['acc'],  color='r',  linewidth='1.7', label="CBOW_CNN_GRU") #linestyle='dotted',
# pyplot.title('CBOW Word Embedding Models')
# pyplot.xlabel('Epochs')
# #pyplot.ylim(0, 1)
# pyplot.ylabel('Accuracy')
# pyplot.grid(True)
# pyplot.legend(loc='center right')  # ['CBOW_LSTM', 'CBOW_GRU', 'CBOW_CNN_LSTM', "CBOW_CNN_GRU", 'CBOW_CNN_BiLSTM', "CBOW_CNN_BiGRU"],
# pyplot.savefig('COnfer_CBOW_Embdding.png')
# #pyplot.show()
#
#
#
#
#
# ##========================pre-traind word2vec data===============================================
# sg_embeddings = Word2Vec(sentences=all_abstract, vector_size=100, window=5, min_count=0, sg=1)
# sg_embeddings.wv.save_word2vec_format('Sg_Pre-trained_word2Vec.txt', binary=False)
#
#
# ##========================abstract CNN training==================================================
# print("create the tokenizer")
# token = Tokenizer()  # create the tokenizer
# token.fit_on_texts(all_abstract)  # fit the tokenizer on the documents
# print("Total words:", len(token.word_index))
#
#
# word_index = token.word_index
# #print('unique words: {}'.format(len(word_index)))
#
# ##print()
# vocab_size = len(token.word_index) + 1  # define vocabulary size (largest integer value)
# #print('Vocabulary size: %d' % vocab_size)
#
#
# #max_length = 259
#
# max_length = max(len(l) for l in all_abstract) # 모든 샘플에서 길이가 가장 긴 샘플의 길이 출력
# #print('샘플의 최대 길이 : {}'.format(max_length))
#
# train, valid = train_test_split(all_abstract, test_size=0.25, random_state=1)
#
# #print("train", len(train))
# #print("valid", len(valid))
#
#
# ##====================================train_labels====================================
# select_words = ['system',  'network', 'approach', 'time', 'cloud', 'information', 'process', 'problem', 'service', 'security']
# Sg_train_labels = []
# for i in range(0, 3395):
#     count = 0
#     for j in range(0, len(select_words)):
#         if select_words[j] in all_abstract[i]:
#             count += 1
#     if count >=1:
#         Sg_train_labels.append(1)
#     else:
#         Sg_train_labels.append(0)
#
#
# ###====================================validation labels====================================
# select_words = ['system',  'network', 'approach', 'time', 'cloud', 'information', 'process', 'problem', 'service', 'security']
# Sg_valid_labels = []
# for i in range(0, 1132):
#
#     count = 0
#     for j in range(0, len(select_words)):
#         if select_words[j] in valid[i]:
#             count += 1
#     if count >=2:
#         Sg_valid_labels.append(1)
#     else:
#         Sg_valid_labels.append(0)
# #print(valid_labels)
#
#
# Sg_train_data = token.texts_to_sequences(train)
# Sg_valid_data = token.texts_to_sequences(valid)
# # print("integer incode:", data)
# # print("length: ", len(data))
#
# Sg_X_train = pad_sequences(Sg_train_data, maxlen=max_length, padding='post')
# Sg_y_train = np.asarray(Sg_train_labels).astype('float32').reshape((-1, 1)) #np.array(Sg_train_labels)
# #print(len(X_train))
#
# Sg_X_valid = pad_sequences(Sg_valid_data, max_length, padding='post')
# Sg_y_valid = np.asarray(Sg_valid_labels).astype('float32').reshape((-1, 1)) #np.array(
#     #Sg_valid_labels)
#
#
#
# ###====================================Sg_CNNs model with word2vec====================================
# sg_embedding_index = {}
# list_v = []
# file = open('Sg_Pre-trained_word2Vec.txt', 'rt', encoding='UTF8')
# line = file.readline()
# totalWords, numOfFeatures = line.split()
# print(totalWords, numOfFeatures)
# for line in file:
#     values = line.split()
#     list_v.append(values)
#     word = values[0]
#     coefs = array(values[1:], dtype='float64')
#     sg_embedding_index[word] = coefs
#
#
# print('Found %s word vectors.' % len(sg_embedding_index))
# df_values = pd.DataFrame(list_v)
# print(df_values, "\n")
#
# sg_embedding_matrix1 = np.array([[0 for col in range(100)] for row in range(10588)])
# for word, i in token.word_index.items():
#     # try:
#     embedding_vector = sg_embedding_index.get(word)
#     if embedding_vector is not None:
#         if( i == 100):
#             print(i,"번째 완료")
#         for j in range(0, 100):
#            sg_embedding_matrix1[i][j] = embedding_vector[j]
#         #print(i,"번째 완료")
#
# #print("sg_embedding_matrix1:", sg_embedding_matrix1)
#
#
# #=======================Sg_CNNs model using LSTM ====================================================
# # embedding_dim = 100
# # pooling = 2
# # dropout = 0.5
# #
# # epochs = 5
# # batch_sizes = 64
#
#
# ##===================LSTM model for Text Classification=====================================================
# Sg_lstm_model = Sequential()
# Sg_lstm_model.add(Embedding(vocab_size, 100, weights=[sg_embedding_matrix1], input_length=max_length, trainable=True))
# Sg_lstm_model.add(LSTM(100, return_sequences=False))
# Sg_lstm_model.add(Dropout(0.5))
# Sg_lstm_model.add(Dense(1, activation='sigmoid'))
# Sg_lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# print(Sg_lstm_model.summary())
# file_path = "Sg_lstm.h5py"
# checkpoint = ModelCheckpoint(filepath=file_path, monitor='acc', verbose=1, save_best_only=True, mode='max', save_weights_only=True)
# callbacks_list = [checkpoint]
# Sg_LSTM = Sg_lstm_model.fit(Sg_X_train, Sg_y_train, validation_data=(Sg_X_valid, Sg_y_valid), epochs=epochs,
#                             batch_size=batch_sizes,
#                         verbose=1,
#                callbacks=callbacks_list)
# Sg_LSTM_Train = Sg_lstm_model.evaluate(Sg_X_train, Sg_y_train, verbose=0)
# print(('Sg_LSTM_Train_Score: %f' % (Sg_LSTM_Train[1] * 100)))
# Sg_LSTM_FGFS_Test = Sg_lstm_model.evaluate(x_test, y_test, verbose=0)
# print(('Sg_LSTM_FGFS_Test Accuracy: %f' % (Sg_LSTM_FGFS_Test[1]*100)))
# Sg_LSTM_AWR_Test = Sg_lstm_model.evaluate(x_test_1, y_test_1, verbose=0)
# print(('Sg_LSTM_AWR_Test Accuracy: %f' % (Sg_LSTM_AWR_Test[1]*100)))
#
#
# ##====================F_score Sg_lstm==========================================
# # predict probabilities for test set
# Sg_lstm_yhat_probs = Sg_lstm_model.predict(x_test, verbose=0)
# # reduce to 1d array
# Sg_lstm_yhat_probs = Sg_lstm_yhat_probs[:, 0]
#
#
# # accuracy: (tp + tn) / (p + n)
# Sg_lstm_accuracy = accuracy_score(y_test, np.round(abs(Sg_lstm_yhat_probs)))
# print('Sg_lstm_accuracy: %f' % Sg_lstm_accuracy)
# # precision tp / (tp + fp)
# Sg_lstm_precision = precision_score(y_test, np.round(abs(Sg_lstm_yhat_probs)))
# print('Sg_lstm_precision: %f' % Sg_lstm_precision)
# # recall: tp / (tp + fn)
# Sg_lstm_recall = recall_score(y_test, np.round(abs(Sg_lstm_yhat_probs)))
# print('Sg_lstm_recall: %f' % Sg_lstm_recall)
# # f1: 2 tp / (2 tp + fp + fn)
# Sg_lstm_f1 = f1_score(y_test, np.round(abs(Sg_lstm_yhat_probs)))
# print('Sg_lstm_f1: %f' % Sg_lstm_f1)
#
#
#
# ##===============================================Sg GRU model===================================================
# Sg_gru_model = Sequential()
# Sg_gru_model.add(Embedding(vocab_size, 100, weights=[sg_embedding_matrix1], input_length=max_length, trainable=True))
# Sg_gru_model.add(GRU(100, return_sequences=False))
# Sg_gru_model.add(Dropout(0.5))
# Sg_gru_model.add(Dense(1, activation='sigmoid'))
# Sg_gru_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# print(Sg_gru_model.summary())
# file_path = "Sg_GRU.h5py"
# checkpoint = ModelCheckpoint(filepath=file_path, monitor='acc', verbose=1, save_best_only=True, mode='max',
#                              save_weights_only=True)
# callbacks_list = [checkpoint]
# Sg_GRU = Sg_gru_model.fit(Sg_X_train, Sg_y_train, validation_data=(Sg_X_valid, Sg_y_valid), epochs=epochs, batch_size=batch_sizes,
#                          verbose=1,
#                        callbacks=callbacks_list)
# Sg_GRU_Train = Sg_gru_model.evaluate(Sg_X_train, Sg_y_train, verbose=0)
# print(('Sg_GRU_Train_Accuracy: %f' % (Sg_GRU_Train[1] * 100)))
# Sg_GRU_FGFS_Test = Sg_gru_model.evaluate(x_test, y_test, verbose=0)
# print(('Sg_GRU_FGFS_Test Test Accuracy: %f' % (Sg_GRU_FGFS_Test[1]*100)))
# Sg_GRU_AWR_Test = Sg_gru_model.evaluate(x_test_1, y_test_1, verbose=0)
# print(('Sg_GRU_AWR_Test Accuracy: %f' % (Sg_GRU_AWR_Test[1]*100)))
#
#
# ##====================F_score Sg_gru==========================================
# # predict probabilities for test set
# Sg_gru_probs = Sg_gru_model.predict(x_test, verbose=0)
# # reduce to 1d array
# Sg_gru_probs = Sg_gru_probs[:, 0]
#
#
# # accuracy: (tp + tn) / (p + n)
# Sg_gru_accuracy = accuracy_score(y_test, np.round(abs(Sg_gru_probs)))
# print('Sg_gru_accuracy: %f' % Sg_gru_accuracy)
# # precision tp / (tp + fp)
# Sg_gru_precision = precision_score(y_test, np.round(abs(Sg_gru_probs)))
# print('Sg_gru_precision: %f' % Sg_gru_precision)
# # recall: tp / (tp + fn)
# Sg_gru_recall = recall_score(y_test, np.round(abs(Sg_gru_probs)))
# print('Sg_gru_recall: %f' % Sg_gru_recall)
# # f1: 2 tp / (2 tp + fp + fn)
# Sg_gru_f1 = f1_score(y_test, np.round(abs(Sg_gru_probs)))
# print('Sg_gru_f1: %f' % Sg_gru_f1)
#
#
# Sg_CNN_LSTM_model = Sequential()
# Sg_CNN_LSTM_model.add(Embedding(vocab_size, 100, weights=[sg_embedding_matrix1], input_length=max_length, trainable=True))
# Sg_CNN_LSTM_model.add(Conv1D(filters=filters, kernel_size=2, padding='same', activation='relu'))
# Sg_CNN_LSTM_model.add(MaxPooling1D(pool_size=pooling))
# Sg_CNN_LSTM_model.add(LSTM(100, dropout=dropout, return_sequences=True, recurrent_dropout=dropout))
# Sg_CNN_LSTM_model.add(Conv1D(filters=filters, kernel_size=3, padding='same', activation='relu'))
# Sg_CNN_LSTM_model.add(MaxPooling1D(pool_size=pooling))
# Sg_CNN_LSTM_model.add(LSTM(100, dropout=dropout, return_sequences=True, recurrent_dropout=dropout))
# Sg_CNN_LSTM_model.add(Conv1D(filters=filters, kernel_size=4, padding='same', activation='relu'))
# Sg_CNN_LSTM_model.add(MaxPooling1D(pool_size=pooling))
# Sg_CNN_LSTM_model.add(LSTM(100, dropout=dropout, return_sequences=True, recurrent_dropout=dropout))
# Sg_CNN_LSTM_model.add(Dense(10, activation='relu'))
# Sg_CNN_LSTM_model.add(Dense(1, activation='sigmoid'))
# Sg_CNN_LSTM_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# print(Sg_CNN_LSTM_model.summary())
#
# file_path = "Sg_CNN_LSTM.h5py"
# checkpoint = ModelCheckpoint(filepath=file_path, monitor='acc', verbose=1, save_best_only=True, mode='max',
#                              save_weights_only=True)
# callbacks_list = [checkpoint]
# Sg_CNN_LSTM = Sg_CNN_LSTM_model.fit(Sg_X_train, Sg_y_train, validation_data=(Sg_X_valid, Sg_y_valid), epochs=epochs, batch_size=batch_sizes, verbose=1,
#                        callbacks=callbacks_list)
# Sg_CNN_LSTM_Train = Sg_CNN_LSTM_model.evaluate(Sg_X_train, Sg_y_train, verbose=0)
# print(('Train_Accuracy: %f' % (Sg_CNN_LSTM_Train[1] * 100)))
# Sg_CNN_LSTM_FGFS_Test = Sg_CNN_LSTM_model.evaluate(x_test, y_test, verbose=0)
# print(('Sg_CNN_LSTM FGFS Test Accuracy: %f' % (Sg_CNN_LSTM_FGFS_Test[1]*100)))
# Sg_CNN_LSTM_AWR_Test = Sg_CNN_LSTM_model.evaluate(x_test_1, y_test_1, verbose=0)
# print(('Sg_CNN_LSTM_AWR Test Accuracy: %f' % (Sg_CNN_LSTM_AWR_Test[1]*100)))
#
#
# ##====================F_score Sg_cnn_lstm==========================================
# # predict probabilities for test set
# Sg_cnn_lstm_probs = Sg_CNN_LSTM_model.predict(x_test, verbose=0)
# # reduce to 1d array
# Sg_cnn_lstm_probs = Sg_cnn_lstm_probs[:, 0]
#
#
# # accuracy: (tp + tn) / (p + n)
# Sg_cnn_lstm_accuracy = accuracy_score(y_test, np.round(abs(Sg_cnn_lstm_probs)))
# print('Sg_cnn_lstm_accuracy: %f' % Sg_cnn_lstm_accuracy)
# # precision tp / (tp + fp)
# Sg_cnn_lstm_precision = precision_score(y_test, np.round(abs(Sg_cnn_lstm_probs)))
# print('Sg_cnn_lstm_precision: %f' % Sg_cnn_lstm_precision)
# # recall: tp / (tp + fn)
# Sg_cnn_lstm_recall = recall_score(y_test, np.round(abs(Sg_cnn_lstm_probs)))
# print('Sg_cnn_lstm_recall: %f' % Sg_cnn_lstm_recall)
# # f1: 2 tp / (2 tp + fp + fn)
# Sg_cnn_lstm_f1 = f1_score(y_test, np.round(abs(Sg_cnn_lstm_probs)))
# print('Sg_cnn_lstm_f1: %f' % Sg_cnn_lstm_f1)
#
#
#
# #=======================Sg_CNNs model using LSTM ====================================================
# Sg_CNN_GRU_model = Sequential()
# Sg_CNN_GRU_model.add(Embedding(vocab_size, 100, weights=[sg_embedding_matrix1], input_length=max_length, trainable=True))
# Sg_CNN_GRU_model.add(Conv1D(filters=filters, kernel_size=2, padding='same', activation='relu'))
# Sg_CNN_GRU_model.add(MaxPooling1D(pool_size=pooling))
# Sg_CNN_GRU_model.add(GRU(100, return_sequences=True, recurrent_dropout=dropout))
# Sg_CNN_GRU_model.add(Conv1D(filters=filters, kernel_size=3, padding='same', activation='relu'))
# Sg_CNN_GRU_model.add(MaxPooling1D(pool_size=pooling))
# Sg_CNN_GRU_model.add(GRU(100, return_sequences=True, recurrent_dropout=dropout))
# Sg_CNN_GRU_model.add(Conv1D(filters=filters, kernel_size=4, padding='same', activation='relu'))
# Sg_CNN_GRU_model.add(MaxPooling1D(pool_size=pooling))
# Sg_CNN_GRU_model.add(GRU(100, return_sequences=True, recurrent_dropout=dropout))
# Sg_CNN_GRU_model.add(Dense(10, activation='relu'))
# Sg_CNN_GRU_model.add(Dense(1, activation='sigmoid'))
# Sg_CNN_GRU_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# print(Sg_CNN_GRU_model.summary())
#
# file_path = "Sg_CNN_GRU.h5py"
# checkpoint = ModelCheckpoint(filepath=file_path, monitor='acc', verbose=1, save_best_only=True, mode='max',
#                              save_weights_only=True)
# callbacks_list = [checkpoint]
# Sg_CNN_GRU = Sg_CNN_GRU_model.fit(Sg_X_train, Sg_y_train, validation_data=(Sg_X_valid, Sg_y_valid), epochs=epochs, batch_size=batch_sizes, verbose=1,
#                        callbacks=callbacks_list)
# Sg_CNN_GRU_Train = Sg_CNN_GRU_model.evaluate(Sg_X_train, Sg_y_train, verbose=0)
# print(('Train_Accuracy: %f' % (Sg_CNN_GRU_Train[1] * 100)))
# Sg_CNN_GRU_FGFS_Test = Sg_CNN_GRU_model.evaluate(x_test, y_test, verbose=0)
# print(('Sg_CNN_GRU FGFS Test Accuracy: %f' % (Sg_CNN_GRU_FGFS_Test[1]*100)))
# Sg_CNN_GRU_AWR_Test = Sg_CNN_GRU_model.evaluate(x_test_1, y_test_1, verbose=0)
# print(('Sg_CNN_GRU_AWR Test Accuracy: %f' % (Sg_CNN_GRU_AWR_Test[1]*100)))
#
#
# ##====================F_score Sg_cnn_gru==========================================
# # predict probabilities for test set
# Sg_cnn_gru_probs = Sg_CNN_GRU_model.predict(x_test, verbose=0)
# # reduce to 1d array
# Sg_cnn_gru_probs = Sg_cnn_gru_probs[:, 0]
#
#
# # accuracy: (tp + tn) / (p + n)
# Sg_cnn_gru_accuracy = accuracy_score(y_test, np.round(abs(Sg_cnn_gru_probs)))
# print('Sg_cnn_gru_accuracy: %f' % Sg_cnn_gru_accuracy)
# # precision tp / (tp + fp)
# Sg_cnn_gru_precision = precision_score(y_test, np.round(abs(Sg_cnn_gru_probs)))
# print('Sg_cnn_gru_precision: %f' % Sg_cnn_gru_precision)
# # recall: tp / (tp + fn)
# Sg_cnn_gru_recall = recall_score(y_test, np.round(abs(Sg_cnn_gru_probs)))
# print('Sg_cnn_gru_recall: %f' % Sg_cnn_gru_recall)
# # f1: 2 tp / (2 tp + fp + fn)
# Sg_cnn_gru_f1 = f1_score(y_test, np.round(abs(Sg_cnn_gru_probs)))
# print('Sg_cnn_gru_f1: %f' % Sg_cnn_gru_f1)
#
#
#
#
# ##=======================CBOW & Sg Data visualization==================================================
# pyplot.plot(Sg_LSTM.history['acc'],  color='b',  linewidth='1.7', label='Sg_LSTM') #linestyle='solid',
# pyplot.plot(Sg_GRU.history['acc'],  color='y',  linewidth='1.7', label='Sg_GRU') #linestyle='dashed',
# pyplot.plot(Sg_CNN_LSTM.history['acc'],  color='g',  linewidth='1.7', label='Sg_CNN_LSTM') #linestyle='dashdot',
# pyplot.plot(Sg_CNN_GRU.history['acc'], color='r',  linewidth='1.7', label='Sg_CNN_GRU') #linestyle='dotted',
# pyplot.title('Sg Word Embedding Models')
# pyplot.xlabel('Epochs')
# #pyplot.ylim(0, 1)
# pyplot.ylabel('Accuracy')
# pyplot.grid(True)
# pyplot.legend(loc='center right') #['Sg_LSTM', 'Sg_GRU', 'Sg_CNN_LSTM', 'Sg_CNN_GRU'],
# pyplot.savefig('Confer_Sg_Embdding.png')
# #pyplot.show()


#====================================F-Scores Result======================================
title = ['LSTM', 'GRU', 'CNN_LSTM', 'CNN_GRU', 'CNN_BiLSTM', 'CNN_BiGRU']
x = np.arange(len(title))
# CBOW = [CBOW_lstm_f1, CBOW_gru_f1, CBOW_cnn_lstm_f1, CBOW_cnn_gru_f1]
# Sg = [Sg_lstm_f1, Sg_gru_f1, Sg_cnn_lstm_f1, Sg_cnn_gru_f1]

CBOW = [0.586868, 0.586868, 0.629371, 0.730318, 0.725945, 0.745945]
Sg = [0.586868, 0.586868, 0.725245, 0.745945, 0.785318, 0.785318]

# 넓이 지정
width = 0.35

# subplots 생성
fig, axes = plt.subplots()

# 넓이 설정
axes.bar(x - width/2, CBOW, width, color='darkgrey', hatch='//', label='CBOW')
axes.bar(x + width/2, Sg, width, color='black', label='Sg')


# ticks & label 설정
plt.xticks(x)
axes.set_xticklabels(title)
plt.xlabel('Models', fontweight='bold')
plt.ylabel('F-Score', fontweight='bold')
plt.legend(loc='upper left')
plt.ylim(0, 1)
#plt.grid()
plt.savefig('Confer_F_score.png')
