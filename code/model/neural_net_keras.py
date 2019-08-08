####################################################################################
###                               neural_net.py                                  ###
####################################################################################
import os
import numpy as np
import pandas as pd
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Concatenate
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.regularizers import l1, l2
from keras.utils import plot_model
import keras.backend as K
import tensorflow as tf


#################################    model    ###################################
### ========================================================== ###
###                            DNN                             ###
### ========================================================== ###
class neural_net(object):
    '''
    Neural network model.
    '''
    def __init__(self, ts_len, input_len, output_len, loss, lr):
        self.ts_len = ts_len
        self.input_len = input_len
        if loss == 'mse':
            self.loss = 'mse'
            self.loss_name = 'mse'
        elif loss == 'weighted_mse':
            self.loss = self.weighted_mse
            self.loss_name = 'weighted_mse'
        else:
            raise ValueError('Unrecognizable loss function.')
        self.output_len = output_len
        self.lr = lr
        self.layer_name = ''

    def build(self, inputs, outputs):
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss=self.loss, optimizer=Adam(lr=self.lr), metrics=[self.r2, self.mae, self.exp2_mae])
        print('Model structure summary:', flush=True)
        print(model.summary())
        self.model = model

    def fit_config(self, model_name, save_dir, batch_size, epoch, tolerance):
        self.batch_size = batch_size
        self.epoch = epoch
        self.tol = tolerance
        self.model_name = f'{model_name}@ts_{self.ts_len}|in_{self.input_len}|out_{self.output_len}|{self.layer_name}|loss_{self.loss_name}|lr_{self.lr}|batch_{self.batch_size}'.replace(' ', '')
        self.model_path = os.path.join(save_dir, '{}.h5'.format(self.model_name))
        # self.model_fig = os.path.join(save_dir, '{}.png'.format(self.model_name))
        # plot_model(self.model, self.model_fig)
        self.log_dir = os.path.join(save_dir, '{}.log'.format(self.model_name))
        if not os.path.isdir(self.log_dir): os.mkdir(self.log_dir)

    def fit(self, X_train, y_train, X_val, y_val):
        early_stopper = EarlyStopping(patience=self.tol, verbose=1)
        check_point = ModelCheckpoint(self.model_path, verbose=1, save_best_only=True)
        reduce_learner = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=int(self.tol*0.25), min_lr=1e-4, verbose=1)
        train_log = TensorBoard(log_dir=self.log_dir)
        # fit
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[early_stopper, check_point, train_log, reduce_learner],
                       batch_size=self.batch_size, epochs=self.epoch, verbose=1, shuffle=True)

    def load(self):
        print('Loading neural network model ... ', end='', flush=True)
        self.model = load_model(self.model_path, custom_objects={'weighted_mse': self.weighted_mse, 'r2': self.r2, 'mae': self.mae, 'exp2_mae': self.exp2_mae})
        print('Done')

    def predict(self, X, verbose=1, index=0):
        y = self.model.predict(X, verbose=verbose)
        if index == -1:
            return y
        elif index < y.shape[1]:
            return y[:, index]
        else:
            raise ValueError('Index out of range.')

    def evaluate(self, dgr, name, mode='full'):
        print('Calculating metrices of {} model ...'.format(name))
        res = dict()
        res[name + '_train'] = self.cal_metrics(y_true=dgr.resp_train, y_pred=dgr.pred_train, weight=dgr.weight_train, mode=mode)
        res[name+'_val'] = self.cal_metrics(y_true=dgr.resp_val, y_pred=dgr.pred_val, weight=dgr.weight_val, mode=mode)
        res[name+'_test'] = self.cal_metrics(y_true=dgr.resp_test, y_pred=dgr.pred_test, weight=dgr.weight_test, mode=mode)
        return pd.DataFrame(res).T

    def evaluate_mean_guess(self, dgr, mode='full'):
        print('Calculating metrices of mean guess model ...')
        res = dict()
        res['REF_MEAN_train'] = self.cal_metrics(y_true=dgr.resp_train, y_pred=np.ones(shape=dgr.resp_train.shape), weight=dgr.weight_train, mode=mode)
        res['REF_MEAN_val'] = self.cal_metrics(y_true=dgr.resp_val, y_pred=np.ones(shape=dgr.resp_val.shape), weight=dgr.weight_val, mode=mode)
        res['REF_MEAN_test'] = self.cal_metrics(y_true=dgr.resp_test, y_pred=np.ones(shape=dgr.resp_test.shape), weight=dgr.weight_test, mode=mode)
        return pd.DataFrame(res).T

    def evaluate_gaussian_sample(self, dgr, mode='full', epoch=10):
        print('Calculating metrices of gaussian sampling model ', end='', flush=True)
        pred_train = np.array([np.random.normal(loc=miu, scale=np.sqrt(sigma2), size=epoch) for (miu, sigma2) in zip(dgr.resp_train, dgr.var_train)]).T
        pred_val = np.array([np.random.normal(loc=miu, scale=np.sqrt(sigma2), size=epoch) for (miu, sigma2) in zip(dgr.resp_val, dgr.var_val)]).T
        pred_test = np.array([np.random.normal(loc=miu, scale=np.sqrt(sigma2), size=epoch) for (miu, sigma2) in zip(dgr.resp_test, dgr.var_test)]).T
        res = []
        for i in range(epoch):
            print('.', end='', flush=True)
            tmp = dict()
            tmp['REF_GS_train'] = self.cal_metrics(y_true=dgr.resp_train, y_pred=pred_train[i,:], weight=dgr.weight_train, mode=mode)
            tmp['REF_GS_val'] = self.cal_metrics(y_true=dgr.resp_val, y_pred=pred_val[i,:], weight=dgr.weight_val, mode=mode)
            tmp['REF_GS_test'] = self.cal_metrics(y_true=dgr.resp_test, y_pred=pred_test[i,:], weight=dgr.weight_test, mode=mode)
            res.append(pd.DataFrame(tmp).T)
        res = pd.concat(res)
        print(' ')
        return res.groupby(by=res.index).mean()

    def cal_metrics(self, y_true, y_pred, weight, mode):
        ptg_true = np.exp2(y_true) - 1
        ptg_pred = np.exp2(y_pred) - 1
        mae = np.mean(np.abs(ptg_true - ptg_pred) * weight)
        mape = np.mean(np.abs(ptg_true - ptg_pred))
        male = np.mean(np.log2(np.abs(ptg_true - ptg_pred) * weight + 1))
        maple = np.mean(np.abs(y_true - y_pred))
        pci = self.concordance_index(ptg_true, ptg_pred, mode)
        top10pi = self.top_inclusion(ptg_true, ptg_pred, 10)
        top10mae = self.top_mae(ptg_true * weight, ptg_pred * weight, weight, 10)
        top10mape = self.top_mae(ptg_true, ptg_pred, weight, 10)
        return pd.Series([mae, male, mape, maple, pci, top10pi, top10mae, top10mape],
                         index=['MAE', 'MALE', 'MAPE', 'MAPLE', 'PCI', 'PI10', 'MAE10', 'MAPE10'])

    # --------------  utility function  ---------------- #
    @staticmethod
    def concordance_index(y_true, y_pred, mode):
        n = y_true.shape[0]
        if mode == 'full' or n <= 9999:
            seq = np.array(y_pred)[np.argsort(y_true)]
            mat = seq.reshape(1, -1) - seq.reshape(-1, 1)
            score = mat[np.triu_indices(mat.shape[0], 1)]
            return (np.sum(score > 0) + 0.5 * np.sum(score == 0)) / len(score)
        elif mode == 'sample' and n > 9999:
            ci = []
            for _ in range(int(n / 9999)*2+1):
                idx = np.random.choice(n, 9999, replace=False)
                sample_true, sample_pred = y_true[idx], y_pred[idx]
                seq = np.array(sample_pred)[np.argsort(sample_true)]
                mat = seq.reshape(1, -1) - seq.reshape(-1, 1)
                score = mat[np.triu_indices(mat.shape[0], 1)]
                ci.append((np.sum(score > 0) + 0.5 * np.sum(score == 0)) / len(score))
            return np.mean(ci)
        else:
            raise ValueError('Mode should be either sample or full.')

    @staticmethod
    def top_inclusion(y_true, y_pred, percentile):
        perc_true = y_true > np.percentile(y_true, 100 - percentile)
        perc_pred = y_pred > np.percentile(y_pred, 100 - percentile)
        return np.sum(perc_pred & perc_true) / np.sum(perc_true)

    @staticmethod
    def top_mae(y_true, y_pred, base, percentile):
        cut = np.percentile(base, 100 - percentile)
        truth = y_true[base > cut]
        pred = y_pred[base > cut]
        return np.mean(np.abs(truth - pred))


    # ----------------   tf function   -------------- #
    @staticmethod
    def r2(y_true_w, y_pred):
        y_true = tf.slice(y_true_w, [0, 0], [-1, 1])
        ss_res = K.sum(K.square(y_true - y_pred))
        ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return 1 - ss_res / (ss_tot + K.epsilon())

    @staticmethod
    def weighted_mse(y_true_w, y_pred):
        y_true = tf.slice(y_true_w, [0, 0], [-1, 1])
        w = tf.slice(y_true_w, [0, 1], [-1, 1])
        return K.mean(K.square(y_true - y_pred) * w)

    @staticmethod
    def mae(y_true_w, y_pred):
        y_true = tf.slice(y_true_w, [0, 0], [-1, 1])
        return K.mean(K.abs(y_true - y_pred))

    @staticmethod
    def exp2_mae(y_true_w, y_pred):
        base = tf.constant(2, dtype='float32')
        y_true = tf.slice(y_true_w, [0, 0], [-1, 1])
        ratio_true = tf.pow(base, tf.slice(y_true, [0, 0], [-1, 1]))
        ratio_pred = tf.pow(base, tf.slice(y_pred, [0, 0], [-1, 1]))
        return K.mean(K.abs(ratio_true - ratio_pred))

### ========================================================== ###
###                            CNN                             ###
### ========================================================== ###
class cnn(neural_net):
    '''
    Convolutional neural network.
    '''
    def __init__(self, ts_len, input_len, output_len, conv_pool_layer, fc_layer, loss, lr):
        '''
        conv_pool_layer:
            list of (filters, kernel_size, pool_size)
        fc_layer:
            list of (units, dropout, regularize)
        '''
        super().__init__(ts_len=ts_len, input_len=input_len, output_len=output_len, loss=loss, lr=lr)
        self.conv_pool_layer = conv_pool_layer
        self.fc_layer = fc_layer
        name_conv = '.'.join(['conv.' + '.'.join(map(str, l)) for l in self.conv_pool_layer])
        name_fc = '.'.join(['fc.' + '.'.join(map(str, l)) for l in self.fc_layer])
        self.layer_name = name_conv + '|' + name_fc
        self._init_model()

    def _init_model(self):
        '''
        Initialize convolutional neural network model
        '''
        print('Initializing cnn model ...', flush=True)
        # time series features
        ts_inputs, ts_layer = [], []
        for i, tl in enumerate(self.ts_len, 1):
            ts_in = Input(shape=(tl, 1), name='ts_input' + str(i))
            ts_inputs.append(ts_in)
            # convolution layer
            ts = ts_in
            for filters, kernel_size, pool_size in self.conv_pool_layer:
                ts = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu', strides=1)(ts)
                if isinstance(pool_size, int):
                    if pool_size > 0:
                        ts = MaxPooling1D(pool_size=pool_size)(ts)
                    elif pool_size == -1:
                        ts = GlobalMaxPooling1D()(ts)
                elif pool_size is not None:
                    raise ValueError('Unrecognizable pool_size: {}'.format(pool_size))
            ts = ts if len(K.int_shape(ts)) == 2 else Flatten()(ts)
            ts_layer.append(ts)
        # org feature
        org_input = Input(shape=(self.input_len,), name='org_input')
        org_layer = Dense(units=128, activation='relu')(org_input)
        org_layer = Dense(units=64, activation='relu')(org_layer)
        # fully connected layer
        fc_layer = Concatenate()(ts_layer + [org_layer])
        for units, dropout, regularize in self.fc_layer:
            if regularize is not None:
                fc_layer = Dense(units=units, activation='relu', kernel_regularizer=l2(regularize), activity_regularizer=l1(regularize))(fc_layer)
            else:
                fc_layer = Dense(units=units, activation='relu')(fc_layer)
            if dropout is not None:
                fc_layer = Dropout(dropout)(fc_layer)
        output = Dense(self.output_len, activation='linear')(fc_layer)
        # model
        super().build(inputs=ts_inputs+[org_input], outputs=output)

### ========================================================== ###
###                           LSTM                             ###
### ========================================================== ###
class lstm(neural_net):
    '''
    Long short term memory model
    '''
    def __init__(self, ts_len, input_len, output_len, lstm_layer, fc_layer, loss, lr):
        '''
        lstm_layer:
            list of (units, dropout, recurrent_dropout)
        fc_layer:
            list of (units, dropout, regularize)
        '''
        super().__init__(ts_len=ts_len, input_len=input_len, output_len=output_len, loss=loss, lr=lr)
        self.lstm_layer = lstm_layer
        self.fc_layer = fc_layer
        name_lstm = '.'.join(['lstm.' + '.'.join(map(str, l)) for l in self.lstm_layer])
        name_fc = '.'.join(['fc.' + '.'.join(map(str, l)) for l in self.fc_layer])
        self.layer_name = name_lstm + '|' + name_fc
        self._init_model()

    def _init_model(self):
        '''
        Initialize LSTM model
        '''
        print('Initializing lstm model ...', flush=True)
        # time series features
        ts_inputs, ts_layer = [], []
        for i, tl in enumerate(self.ts_len, 1):
            ts_in = Input(shape=(tl, 1), name='ts_input' + str(i))
            ts_inputs.append(ts_in)
            # lstm layer
            ts = ts_in
            for i, (units, drop, rec_drop) in enumerate(self.lstm_layer):
                if i < len(self.lstm_layer) - 1:
                    ts = LSTM(units=units, return_sequences=True, dropout=drop, recurrent_dropout=rec_drop)(ts)
                else:
                    ts = LSTM(units=units, return_sequences=False, dropout=drop, recurrent_dropout=rec_drop)(ts)
            ts_layer.append(ts)
        # org feature
        org_input = Input(shape=(self.input_len,), name='org_input')
        org_layer = Dense(units=128, activation='relu')(org_input)
        org_layer = Dense(units=64, activation='relu')(org_layer)
        # fully connected layer
        fc_layer = Concatenate()(ts_layer + [org_layer])
        for units, dropout, regularize in self.fc_layer:
            if regularize is not None:
                fc_layer = Dense(units=units, activation='relu', kernel_regularizer=l2(regularize), activity_regularizer=l1(regularize))(fc_layer)
            else:
                fc_layer = Dense(units=units, activation='relu')(fc_layer)
            if dropout is not None:
                fc_layer = Dropout(dropout)(fc_layer)
        output = Dense(self.output_len, activation='linear')(fc_layer)
        # model
        super().build(inputs=ts_inputs+[org_input], outputs=output)


### ========================================================== ###
###                           FNN                              ###
### ========================================================== ###
class fnn(neural_net):
    '''
    Feedforward neural network
    '''
    def __init__(self, ts_len, input_len, output_len, fc_layer_ts, fc_layer, loss, lr):
        '''
        fc_layer_ts:
            list of (units, dropout, regularize)
        fc_layer:
            list of (units, dropout, regularize)
        '''
        super().__init__(ts_len=ts_len, input_len=input_len, output_len=output_len, loss=loss, lr=lr)
        self.fc_layer_ts = fc_layer_ts
        self.fc_layer = fc_layer
        name_ts = '.'.join(['fc.' + '.'.join(map(str, l)) for l in self.fc_layer_ts])
        name_fc = '.'.join(['fc.' + '.'.join(map(str, l)) for l in self.fc_layer])
        self.layer_name = name_ts + '|' + name_fc
        self._init_model()

    def _init_model(self):
        '''
        Initialize feedforward neural network
        '''
        print('Initializing feedforward model ...', flush=True)
        # time series features
        ts_inputs, ts_layer = [], []
        for i, tl in enumerate(self.ts_len, 1):
            ts_in = Input(shape=(tl, 1), name='ts_input' + str(i))
            ts_inputs.append(ts_in)
            ts = ts_in if len(K.int_shape(ts_in)) == 2 else Flatten()(ts_in)
            for units, dropout, regularize in self.fc_layer_ts:
                if regularize is not None:
                    ts = Dense(units=units, activation='relu', kernel_regularizer=l2(regularize), activity_regularizer=l1(regularize))(ts)
                else:
                    ts = Dense(units=units, activation='relu')(ts)
                if dropout is not None:
                    ts = Dropout(dropout)(ts)
            ts_layer.append(ts)
        # org feature
        org_input = Input(shape=(self.input_len,), name='org_input')
        org_layer = Dense(units=128, activation='relu')(org_input)
        org_layer = Dense(units=64, activation='relu')(org_layer)
        # fully connected layer
        fc_layer = Concatenate()(ts_layer + [org_layer])
        for units, dropout, regularize in self.fc_layer:
            if regularize is not None:
                fc_layer = Dense(units=units, activation='relu', kernel_regularizer=l2(regularize), activity_regularizer=l1(regularize))(fc_layer)
            else:
                fc_layer = Dense(units=units, activation='relu')(fc_layer)
            if dropout is not None:
                fc_layer = Dropout(dropout)(fc_layer)
        output = Dense(self.output_len, activation='linear')(fc_layer)
        # model
        super().build(inputs=ts_inputs+[org_input], outputs=output)

