# -*- coding:utf-8 -*-
import os

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


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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


embed_arr = []
dims = [100,100]

emb = pd.read_pickle('/mnt/2TB/jane96/pingan/w2v/128_10/question.pickle')
embed_arr.append(emb)
emb = pd.read_pickle('/mnt/2TB/jane96/pingan/w2v/128_10/answer.pickle')
embed_arr.append(emb)




def model_conv():
    x_arr = []
    emb_layer = []
    for index in range(2):
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
    for index in range(2):
        seq = Input(shape=(dims[index],))
        seqs.append(seq)
        print('seqs :', index)
        del seq

    for index in range(2):
        now = emb_layer[index](seqs[index])
        x_arr.append(now)
        print('seqs emb :', index)
        del now

    sdrop = SpatialDropout1D(rate=0.2)

    print("start merge...")
    all_layer = []
    all_layer_avg = []
    for index in range(2):
        x = sdrop(x_arr[index])
        x = Dropout(0.2)(Bidirectional(CuDNNLSTM(300, return_sequences=True))(x))

        semantic = TimeDistributed(Dense(200,activation='tanh'))(x)

        merged_1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(200,))(semantic)
        merged_1_avg = Lambda(lambda x: K.mean(x, axis=1), output_shape=(200,))(semantic)

        all_layer.append(merged_1)
        all_layer_avg.append(merged_1_avg)
        print('model combine: ', index)



    # hin = Input(shape=(num_feature_input,))
    # htime = Dense(16,activations = 'relu')(hin)

    x = concatenate([x for x in all_layer]+[y for y in all_layer_avg] )

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(x))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))

    pred = Dense(output_dim=57, activation='softmax')(x)
    model = Model(inputs=[seq for seq in seqs], outputs=pred)

    # model = multi_gpu_model(model, 2)
    model.compile(
        loss="categorical_crossentropy",
        optimizer='adam',
        metrics=["accuracy"]

    )
    model.summary()
    return model


gc.collect()
trainy= pd.read_excel('/mnt/2TB/jane96/pingan/base/train.xlsx')
trainy = trainy[trainy['category'] == 0]
y = (trainy['label']).values.tolist()

skf = StratifiedKFold(n_splits=5, random_state=1017, shuffle=True)
sub = np.zeros((100000, 2))
# oof_pred = np.zeros((train_input_5.shape[0], 6))
score = []
count = 0
# if not os.path.exists("model"):
#     os.mkdir("model")
print('read input...')
all_x = []
test_x = []
level = 50652
data = pd.read_csv('/mnt/2TB/jane96/pingan/w2v/128_10/question.csv').iloc[:,1:]
all_x.append(data[:level])
test_x.append(data[level:])
data = pd.read_csv('/mnt/2TB/jane96/pingan/w2v/128_10/answer.csv').iloc[:,1:]
all_x.append(data[:level])
test_x.append(data[level:])





from matplotlib import pyplot as plt
def save_train_img(history,filePath):


    # summarize history for loss
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(filePath)
def save_history_text(history,filePath):
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']

    f = open(filePath.format(dir),'w')
    np.savetxt(f,train_acc + val_acc)
    f.close()
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0:2]))).round()
        val_targ = self.validation_data[2]
        _val_f1 = f1_score(val_targ, val_predict,average='macro')
        _val_recall = recall_score(val_targ, val_predict,average='macro')
        _val_precision = precision_score(val_targ, val_predict,average='macro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print("val_f1: %f  val_precision: %f  val_recall %f" %(_val_f1, _val_precision, _val_recall))
        return

metrics = Metrics()
from keras.callbacks import TensorBoard
yLable = pd.Series(y).drop_duplicates().values
y_index = dict()
for index in range(len(yLable)):
    y_index[yLable[index]] = index
y_label = [y_index[x] for x in y]
dir =13
count = 0
for i, (train_index, test_index) in enumerate(skf.split(all_x[0], y_label[:level])):
    K.clear_session()
    print("start train....")
    all_y = keras.utils.to_categorical(y_label, 57)
    print("FOLD | ", count + 1)
    print("###" * 35)
    gc.collect()

    model_age = model_conv()
    print('load struct finish...')
    arr_tr = []
    arr_va = []
    for k in range(2):
        x1_tr, x1_va = np.array(all_x[k])[train_index], np.array(all_x[k])[test_index]
        arr_tr.append(x1_tr)
        arr_va.append(x1_va)
        del x1_tr, x1_va
    y_tr, y_va = np.array(all_y)[train_index], np.array(all_y)[test_index]
    print('begining.....')
    filepath = 'data/store/6_20/{}/best_model_gender_{}_0.h5'.format(dir,count)

    checkpoint = ModelCheckpoint(
        filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_acc', factor=0.5, patience=1, min_lr=0.0001, verbose=1)
    earlystopping = EarlyStopping(
        monitor='val_acc', min_delta=0.000025, patience=5, verbose=1, mode='max')

    vision = TensorBoard(log_dir='data/store/6_20/{}/1_{}'.format(dir,i))

    callbacks = [checkpoint,reduce_lr,  earlystopping,vision,metrics]
    history = model_age.fit(arr_tr, y_tr, batch_size=512, callbacks=callbacks, epochs=15, validation_data=(arr_va, y_va),
                          verbose=1, shuffle=True)


    del arr_tr,arr_va,y_tr,y_va
    model_age.load_weights(filepath)
    print('load model finish....')

    score = model_age.predict(test_x, batch_size=1024, verbose=1)
    pd.DataFrame(score).to_csv('data/store/6_20/{}/scoresub_{}.csv'.format(dir,i), index=False)
    result = pd.DataFrame()
    result['label'] = score.argmax(1)
    print('save model....')
    result.to_csv('data/store/6_20/{}/sub_{}.csv'.format(dir,i), index=False)

    save_train_img(history, 'data/store/6_20/{}/track_{}.png'.format(dir,i))
    save_history_text(history,'data/store/6_20/{}/track_{}.txt'.format(dir,i))
    del model_age
    del history

    print('save model finish....')
    count += 1
print('over!')






