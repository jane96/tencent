# -*- coding:utf-8 -*-
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
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
from keras.utils import plot_model
import matplotlib.pyplot as plt
import random as rn
import gc
import logging
import gensim

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("start")
from keras.engine.topology import Layer


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim

class AdamW(Optimizer):
        def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, weight_decay=1e-4,  # decoupled weight decay (1/4)
                     epsilon=1e-8, decay=0., **kwargs):
            super(AdamW, self).__init__(**kwargs)
            with K.name_scope(self.__class__.__name__):
                self.iterations = K.variable(0, dtype='int64', name='iterations')
                self.lr = K.variable(lr, name='lr')
                self.beta_1 = K.variable(beta_1, name='beta_1')
                self.beta_2 = K.variable(beta_2, name='beta_2')
                self.decay = K.variable(decay, name='decay')
                # decoupled weight decay (2/4)
                self.wd = K.variable(weight_decay, name='weight_decay')
            self.epsilon = epsilon
            self.initial_decay = decay

        @interfaces.legacy_get_updates_support
        def get_updates(self, loss, params):
            grads = self.get_gradients(loss, params)
            self.updates = [K.update_add(self.iterations, 1)]
            wd = self.wd  # decoupled weight decay (3/4)

            lr = self.lr
            if self.initial_decay > 0:
                lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

            t = K.cast(self.iterations, K.floatx()) + 1
            lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                         (1. - K.pow(self.beta_1, t)))

            ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
            vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
            self.weights = [self.iterations] + ms + vs

            for p, g, m, v in zip(params, grads, ms, vs):
                m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
                v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
                # decoupled weight decay (4/4)
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon) - lr * wd * p

                self.updates.append(K.update(m, m_t))
                self.updates.append(K.update(v, v_t))
                new_p = p_t

                # Apply constraints.
                if getattr(p, 'constraint', None) is not None:
                    new_p = p.constraint(new_p)

                self.updates.append(K.update(p, new_p))
            return self.updates

        def get_config(self):
            config = {'lr': float(K.get_value(self.lr)),
                      'beta_1': float(K.get_value(self.beta_1)),
                      'beta_2': float(K.get_value(self.beta_2)),
                      'decay': float(K.get_value(self.decay)),
                      'weight_decay': float(K.get_value(self.wd)),
                      'epsilon': self.epsilon}
            base_config = super(AdamW, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))
class RAdam(keras.optimizers.Optimizer):
    """RAdam optimizer.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        weight_decay: float >= 0. Weight decay for each param.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
        total_steps: int >= 0. Total number of training steps. Enable warmup by setting a positive value.
        warmup_proportion: 0 < warmup_proportion < 1. The proportion of increasing steps.
        min_lr: float >= 0. Minimum learning rate after warmup.
    # References
        - [Adam - A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
        - [On The Variance Of The Adaptive Learning Rate And Beyond](https://arxiv.org/pdf/1908.03265v1.pdf)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., weight_decay=0., amsgrad=False,
                 total_steps=0, warmup_proportion=0.001, min_lr=0., **kwargs):
        super(RAdam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            self.weight_decay = K.variable(weight_decay, name='weight_decay')
            self.total_steps = K.variable(total_steps, name='total_steps')
            self.warmup_proportion = K.variable(warmup_proportion, name='warmup_proportion')
            self.min_lr = K.variable(lr, name='min_lr')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.initial_weight_decay = weight_decay
        self.initial_total_steps = total_steps
        self.amsgrad = amsgrad

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr

        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations, K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1

        if self.initial_total_steps > 0:
            warmup_steps = self.total_steps * self.warmup_proportion
            decay_steps = self.total_steps - warmup_steps
            lr = K.switch(
                t <= warmup_steps,
                lr * (t / warmup_steps),
                lr * (1.0 - K.minimum(t, decay_steps) / decay_steps),
            )

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='m_' + str(i)) for (i, p) in enumerate(params)]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='v_' + str(i)) for (i, p) in enumerate(params)]

        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='vhat_' + str(i)) for (i, p) in enumerate(params)]
        else:
            vhats = [K.zeros(1, name='vhat_' + str(i)) for i in range(len(params))]

        self.weights = [self.iterations] + ms + vs + vhats

        beta_1_t = K.pow(self.beta_1, t)
        beta_2_t = K.pow(self.beta_2, t)

        sma_inf = 2.0 / (1.0 - self.beta_2) - 1.0
        sma_t = sma_inf - 2.0 * t * beta_2_t / (1.0 - beta_2_t)

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

            m_corr_t = m_t / (1.0 - beta_1_t)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                v_corr_t = K.sqrt(vhat_t / (1.0 - beta_2_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                v_corr_t = K.sqrt(v_t / (1.0 - beta_2_t) + self.epsilon)

            r_t = K.sqrt((sma_t - 4.0) / (sma_inf - 4.0) *
                         (sma_t - 2.0) / (sma_inf - 2.0) *
                         sma_inf / sma_t)

            p_t = K.switch(sma_t > 5, r_t * m_corr_t / v_corr_t, m_corr_t)

            if self.initial_weight_decay > 0:
                p_t += self.weight_decay * p

            p_t = p - lr * p_t

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'weight_decay': float(K.get_value(self.weight_decay)),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
            'total_steps': float(K.get_value(self.total_steps)),
            'warmup_proportion': float(K.get_value(self.warmup_proportion)),
            'min_lr': float(K.get_value(self.min_lr)),
        }
        base_config = super(RAdam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def set_tokenizer(docs, split_char=' ', max_len=100):
    tokenizer = Tokenizer(lower=False, char_level=False, split=split_char)
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
    w2v = Word2Vec(input_docs, size=embed_size, sg=1, window=8, seed=1017, workers=24, min_count=1, iter=10)
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

class Lookahead(object):
    """Add the [Lookahead Optimizer](https://arxiv.org/abs/1907.08610) functionality for [keras](https://keras.io/).

    """

    def __init__(self, k=5, alpha=0.5):
        self.k = k
        self.alpha = alpha
        self.count = 0

    def inject(self, model):
        """Inject the Lookahead algorithm for the given model.
        The following code is modified from keras's _make_train_function method.
        See: https://github.com/keras-team/keras/blob/master/keras/engine/training.py#L497
        """
        if not hasattr(model, 'train_function'):
            raise RuntimeError('You must compile your model before using it.')

        model._check_trainable_weights_consistency()

        if model.train_function is None:
            inputs = (model._feed_inputs +
                      model._feed_targets +
                      model._feed_sample_weights)
            if model._uses_dynamic_learning_phase():
                inputs += [K.learning_phase()]
            fast_params = model._collected_trainable_weights

            with K.name_scope('training'):
                with K.name_scope(model.optimizer.__class__.__name__):
                    training_updates = model.optimizer.get_updates(
                        params=fast_params,
                        loss=model.total_loss)
                    slow_params = [K.variable(p) for p in fast_params]
                fast_updates = (model.updates +
                                training_updates +
                                model.metrics_updates)

                slow_updates, copy_updates = [], []
                for p, q in zip(fast_params, slow_params):
                    slow_updates.append(K.update(q, q + self.alpha * (p - q)))
                    copy_updates.append(K.update(p, q))

                # Gets loss and metrics. Updates weights at each call.
                fast_train_function = K.function(
                    inputs,
                    [model.total_loss] + model.metrics_tensors,
                    updates=fast_updates,
                    name='fast_train_function',
                    **model._function_kwargs)

                def F(inputs):
                    self.count += 1
                    R = fast_train_function(inputs)
                    if self.count % self.k == 0:
                        K.batch_get_value(slow_updates)
                        K.batch_get_value(copy_updates)
                    return R

                model.train_function = F

gc.collect()
columns_x = ['c_id', 'ad_id','at_id','pr_id']
embed_arr = []
dims = [200,200,200]
"""

"""
for co in columns_x:
    emb = pd.read_pickle('/home/jane96/tencent/200_60/_{}.pickle'.format(co))
    embed_arr.append(emb)
    del emb
    print('read:', co, '')

x_arr = []
def model_conv():
    x_arr = []
    emb_layer = []
    for index in range(len(columns_x)):
        layer = Embedding(
            input_dim=embed_arr[index].shape[0],
            output_dim=embed_arr[index].shape[1],
            weights=[embed_arr[index]],
            input_length=dims[index],
            trainable=False
        )
        emb_layer.append(layer)
        print('emb layer:', index)
    seqs = []
    for index in range(len(columns_x)):
        seq = Input(shape=(dims[index],))
        seqs.append(seq)
        print('seqs :', index)
        del seq

    for index in range(len(columns_x)):
        now = emb_layer[index](seqs[index])
        x_arr.append(now)
        print('seqs emb :', index)
        del now

    sdrop = SpatialDropout1D(rate=0.2)
    lstm_layer = Bidirectional(CuDNNLSTM(180, return_sequences=True,
                                         kernel_initializer=glorot_uniform(seed=123)))
    gru_layer = Bidirectional(CuDNNLSTM(100, return_sequences=True,
                                        kernel_initializer=glorot_uniform(seed=123)))
    cnn1d_layer = keras.layers.Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")

    avg_pool = GlobalAveragePooling1D()
    max_pool = GlobalMaxPooling1D()
    print("start merge...")
    for index in range(len(columns_x)):
        x_arr[index] = sdrop(x_arr[index])
        lstm1 = lstm_layer(x_arr[index])

        gru1 = gru_layer(lstm1)
        att_1 = Attention(dims[index])(lstm1)
        att_3 = Attention(dims[index])(gru1)
        cnn1 = cnn1d_layer(lstm1)

        x_arr[index] = concatenate([att_1, att_3, max_pool(cnn1),  max_pool(gru1)])
        print('model combine: ', index)

    merge = Multiply()([x for x in x_arr])
    merge = Dropout(0.2)(merge)

    # hin = Input(shape=(num_feature_input,))
    # htime = Dense(16,activations = 'relu')(hin)

    x = concatenate([x for x in x_arr] + [merge])

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(x))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    # pred = Dense(10, activation='softmax')(x)
    pred = Dense(output_dim=2, activation='softmax')(x)
    model = Model(inputs=[seq for seq in seqs], outputs=pred)
    #     from keras.utils import multi_gpu_model
    #     model = multi_gpu_model(model, 2)
    model = multi_gpu_model(model, 2)
    model.compile(
        loss="categorical_crossentropy",
        optimizer='adam',
        metrics=['accuracy']
    )
    model.summary()
    return model


gc.collect()
y = (pd.read_pickle('/home/jane96/tencent/y_gender.pickle') - 1).values.tolist()

skf = StratifiedKFold(n_splits=5, random_state=1013, shuffle=True)
sub = np.zeros((100000, 2))
# oof_pred = np.zeros((train_input_5.shape[0], 6))
score = []
count = 0
# if not os.path.exists("model"):
#     os.mkdir("model")
print('read input...')
all_x = []
test_x = []
level = 900000
count = 0
for co in columns_x:
    data = pd.read_csv('/home/jane96/tencent/200_128/aa_{}.csv'.format(co))
    if data.shape[1] > dims[count]:
        data = data.iloc[:, 1:].values.tolist()
    else:
        data = data.values.tolist()
    all_x.append(data[:level])
    test_x.append(data[level:])
    del data
    count += 1
    print(co)


def generate_batch(batch_size, x, y):
    temp_x = []
    temp_y = []
    a = np.random.randint(0, len(x[0]), batch_size)
    for index in len(x):
        temp_x.append(x[index][a])
        temp_y.append(y[index][a])
    yield (temp_x, temp_y)

import pickle
for i, (train_index, test_index) in enumerate(skf.split(all_x[0], y[:level])):
    print("start train....")
    all_y = keras.utils.to_categorical(y, 2)
    print("FOLD | ", count + 1)
    print("###" * 35)
    gc.collect()

    model_age = model_conv()
    print('load struct finish...')
    arr_tr = []
    arr_va = []
    for k in range(len(columns_x)):
        #x1_tr, x1_va = np.array(all_x[k])[train_index], np.array(all_x[k])[test_index]
        x1_tr, x1_va = np.array(all_x[k]), np.array(all_x[k])[test_index]
        arr_tr.append(x1_tr)
        arr_va.append(x1_va)
        del x1_tr, x1_va
    y_tr, y_va = np.array(all_y), np.array(all_y)[test_index]
    print('begining.....')
    filepath = '/home/jane96/tencent/best_model_gender_{}_0.h5'.format(count)

    checkpoint = ModelCheckpoint(
        filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_acc', factor=0.5, patience=3, min_lr=0.0001, verbose=1)
    earlystopping = EarlyStopping(
        monitor='val_acc', min_delta=0.0001, patience=2, verbose=1, mode='max')

    callbacks = [checkpoint,  earlystopping]
    history = model_age.fit(arr_tr, y_tr, batch_size=128, callbacks=callbacks, epochs=5, validation_data=(arr_va, y_va),
                          verbose=1, shuffle=True)

    # pickle.dump(history.history,'1_{}_hist.txt')
    # fig1, ax_acc = plt.subplots()
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.title('Model - Accuracy')
    # plt.legend(['Training', 'Validation'], loc='lower right')
    # plt.show()
    #
    # fig2, ax_loss = plt.subplots()
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Model- Loss')
    # plt.legend(['Training', 'Validation'], loc='upper right')
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.show()



    del arr_tr,arr_va,y_tr,y_va
    model_age.load_weights(filepath)
    print('load model finish....')
    # oof_pred[test_index] = model_age.predict([x1_va, x3_va],batch_size=2048,verbose=1)
    score = model_age.predict(test_x, batch_size=1024, verbose=1)
    pd.DataFrame(score).to_csv('/home/jane96/tencent/sub/3_5{}_gender_data.csv'.format(i),index=False)
    result = pd.DataFrame()
    result['predicted_age'] = score.argmax(1) + 1
    print('save model....')
    result.to_csv('/home/jane96/tencent/store/3_age_1_5_{}.csv'.format(i), index=False)
    # sub += score/skf.n_splits
    # score.append(np.max(hist.history['val_acc']))
    print('save model finish....')
    count += 1
    gc.collect( )
# print('acc:', np.mean(score))
submit = pd.DataFrame()
submit['predicted_age'] = sub.argmax(1) + 1
submit.to_csv('/home/jane96/tencent/store/31_agender_4_final.csv', index=False)





