# -*- coding:utf-8 -*-
import os

import keras
import pandas as pd
from docutils.parsers.rst.directives.admonitions import Attention
from keras.initializers import glorot_uniform
from tqdm.autonotebook import *
from keras.utils import multi_gpu_model
from sklearn.model_selection import StratifiedKFold
from gensim.models import FastText, Word2Vec
import re
from keras.layers import *
from keras.models import *
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
from keras.callbacks import *

from keras.optimizers import *
from keras.utils import to_categorical
import tensorflow as tf
import random as rn
import gc
import logging
import gensim

print("start")
from keras.engine.topology import Layer


def set_tokenizer(docs, split_char=' ', max_len=100):
    tokenizer = Tokenizer(lower=False, char_level=False, split=split_char,filters=',')
    tokenizer.fit_on_texts(docs)
    X = tokenizer.texts_to_sequences(docs)
    maxlen = max_len
    X = pad_sequences(X, maxlen=maxlen, value=0)
    word_index = tokenizer.word_index
    return X, word_index


def trian_save_word2vec(docs, embed_size=300, save_name='w2v.txt', split_char=' '):
    input_docs = []
    for i in docs:
        input_docs.append(i.split(split_char))
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
    w2v = Word2Vec(input_docs, size=embed_size, sg=1, window=10, seed=1017, workers=16, min_count=1, iter=10)
    w2v.wv.save_word2vec_format(save_name)
    print("w2v model done")
    return w2v


def get_embedding_matrix(word_index, embed_size=300, Emed_path="w2v_300.txt"):
    embeddings_index = gensim.models.KeyedVectors.load_word2vec_format(
        Emed_path, binary=False)
    nb_words = len(word_index) + 1
    embedding_matrix = np.zeros((nb_words, embed_size))
    count = 0
    for word, i in tqdm(word_index.items()):
        if i >= nb_words:
            continue
        try:
            embedding_vector = embeddings_index[word]
        except:
            embedding_vector = np.zeros(embed_size)
            count += 1
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print("null cnt", count)
    return embedding_matrix




if __name__ == '__main__':
    for index in range(1):
        data = pd.read_csv('/mnt/2TB/jane96/pingan/w2v/question.csv')
        print('split word...')
        temp = list(data['0'])
        del data
        print('tokenzier word...')
        x, seq_index = set_tokenizer(temp, split_char=' ', max_len=100)
        pd.DataFrame(x).to_csv('/mnt/2TB/jane96/pingan/w2v/128_10/question.csv')
        print('sequence: ')
        trian_save_word2vec(temp, embed_size=128, save_name='/mnt/2TB/jane96/pingan/w2v/128_10/question.txt',
                            split_char=' ')
        print('embed word...')

        embd = get_embedding_matrix(seq_index, embed_size=128, Emed_path='/mnt/2TB/jane96/pingan/w2v/128_10/question.txt')
        pd.DataFrame(embd).to_pickle('/mnt/2TB/jane96/pingan/w2v/128_10/question.pickle')