####################################################################################
###                               neural_net.py                                  ###
####################################################################################
import os
import numpy as np
import pandas as pd
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, ReLU, multiply, add
from keras.layers import BatchNormalization, Flatten, Concatenate, Dropout
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.regularizers import l1, l2
from keras.utils import plot_model
import keras.backend as K
import tensorflow as tf

#################################    function    ###################################
def DenseNorm(units, dropout, name=None):
    model = Sequential()
    model.add(Dense(units))
    model.add(BatchNormalization())
    model.add(ReLU())
    if dropout > 0:
        model.add(Dropout(dropout))
    if name is not None:
        model.name = name
    else:
        model.name = model.name.replace('sequential', 'dense_norm')
    return model

def unit_decay(units, i):
    return int(2 * units / (i + 1))


#################################    model    ###################################
### ========================================================== ###
###                            DNN                             ###
### ========================================================== ###
class neural_net(object):
    '''
    Neural network model.
    '''
    def __init__(self, ts_len, t_len, geo_len, org_len, output_len, loss, lr):
        self.ts_len = ts_len
        self.t_len = t_len
        self.geo_len = geo_len
        self.org_len = org_len
        self.input_len = f'{self.ts_len}.{self.t_len}.{self.geo_len}.{self.org_len}'
        self.output_len = output_len
        assert loss in ['mse', 'weighted_mse'], 'Unrecognizable loss function.'
        if loss == 'mse':
            self.loss = 'mse'
            self.loss_name = 'mse'
        elif loss == 'weighted_mse':
            self.loss = self.weighted_mse
            self.loss_name = 'weighted_mse'
        self.lr = lr
        self.conv_pool_layer = []
        self.lstm_layer = []
        self.fc_layer_ts = []
        self.fc_layer = []
        self.layer_name = ''

    def _build(self, struc):
        '''
        Initialize neural network model
        '''
        # time series features
        ts_inputs, ts_layer = [], []
        if struc == 'cnn':
            print('Initializing cnn model ...', flush=True)
            for i, tl in enumerate(self.ts_len, 1):
                ts_in = Input(shape=(tl, 1), name='ts_input' + str(i))
                ts_inputs.append(ts_in)
                ts = ts_in
                for filters, kernel_size, pool_size in self.conv_pool_layer:
                    ts = Conv1D(filters=unit_decay(filters, i), kernel_size=kernel_size, padding='same', activation='relu', strides=1)(ts)
                    if pool_size is not None:
                        assert isinstance(pool_size, int), 'Unrecognizable pool_size: {}'.format(pool_size)
                        if pool_size > 0:
                            ts = MaxPooling1D(pool_size=pool_size)(ts)
                        elif pool_size == -1:
                            ts = GlobalMaxPooling1D()(ts)
                ts = ts if len(K.int_shape(ts)) == 2 else Flatten()(ts)
                ts_layer.append(ts)
        elif struc == 'lstm':
            print('Initializing lstm model ...', flush=True)
            for i, tl in enumerate(self.ts_len, 1):
                ts_in = Input(shape=(tl, 1), name='ts_input' + str(i))
                ts_inputs.append(ts_in)
                ts = ts_in
                for j, (units, drop, rec_drop) in enumerate(self.lstm_layer):
                    if j < len(self.lstm_layer) - 1:
                        ts = LSTM(units=unit_decay(units, i), return_sequences=True, dropout=drop, recurrent_dropout=rec_drop)(ts)
                    else:
                        ts = LSTM(units=unit_decay(units, i), return_sequences=False, dropout=drop, recurrent_dropout=rec_drop)(ts)
                ts_layer.append(ts)
        elif struc == 'cnn_lstm':
            print('Initializing cnn model ...', flush=True)
            for i, tl in enumerate(self.ts_len, 1):
                ts_in = Input(shape=(tl, 1), name='ts_input' + str(i))
                ts_inputs.append(ts_in)
                ts = ts_in
                for filters, kernel_size, pool_size in self.conv_pool_layer:
                    ts = Conv1D(filters=unit_decay(filters, i), kernel_size=kernel_size, padding='same', activation='relu', strides=1)(ts)
                    if pool_size is not None:
                        assert isinstance(pool_size, int) and pool_size > 0, 'Unrecognizable pool_size: {}'.format(pool_size)
                        ts = MaxPooling1D(pool_size=pool_size, padding='same')(ts)
                for j, (units, drop, rec_drop) in enumerate(self.lstm_layer):
                    if j < len(self.lstm_layer) - 1:
                        ts = LSTM(units=unit_decay(units, i), return_sequences=True, dropout=drop, recurrent_dropout=rec_drop)(ts)
                    else:
                        ts = LSTM(units=unit_decay(units, i), return_sequences=False, dropout=drop, recurrent_dropout=rec_drop)(ts)
                ts_layer.append(ts)
        elif struc == 'fnn':
            print('Initializing feedforward model ...', flush=True)
            for i, tl in enumerate(self.ts_len, 1):
                ts_in = Input(shape=(tl, 1), name='ts_input' + str(i))
                ts_inputs.append(ts_in)
                ts = ts_in if len(K.int_shape(ts_in)) == 2 else Flatten()(ts_in)
                for units, dropout in self.fc_layer_ts:
                    ts = DenseNorm(unit_decay(units, i), dropout)(ts)
                ts_layer.append(ts)
        ts_layer = Concatenate()(ts_layer)
        # time modulator features
        t_input = Input(shape=(self.t_len,), name='t_input')
        time = Dense(16, activation='relu')(t_input)
        time = Dense(64, activation='relu')(time)
        ts_shape = K.int_shape(ts_layer)[1]
        gamma = Dense(ts_shape, activation='sigmoid', name='gamma')(time)
        beta = Dense(ts_shape, activation='tanh', name='beta')(time)
        ts_layer = add([multiply([ts_layer, gamma]), beta], name='ts_feature')
        # geographical features
        geo_input = Input(shape=(self.geo_len,), name='geo_input')
        geo_layer = DenseNorm(128, 0.2)(geo_input)
        geo_layer = DenseNorm(64, 0, name='geo_feature')(geo_layer)
        # org feature
        org_input = Input(shape=(self.org_len,), name='org_input')
        org_layer = DenseNorm(32, 0.2)(org_input)
        org_layer = DenseNorm(16, 0, name='org_feature')(org_layer)
        # fully connected layer
        fc_layer = Concatenate()([ts_layer, geo_layer, org_layer])
        for units, dropout in self.fc_layer:
            fc_layer = DenseNorm(units, dropout)(fc_layer)
        output = Dense(self.output_len, activation='linear', name='output')(fc_layer)
        # model
        model = Model(inputs=ts_inputs + [t_input, geo_input, org_input], outputs=output)
        model.compile(loss=self.loss, optimizer=Adam(lr=self.lr), metrics=[self.r2, self.exp2_mae])
        print('Model structure summary:', flush=True)
        print(model.summary())
        self.model = model

    def fit_config(self, model_name, save_dir, batch_size, epoch, tolerance):
        self.batch_size = batch_size
        self.epoch = epoch
        self.tol = tolerance
        self.model_name = f'{model_name}@in.{self.input_len}-out.{self.output_len}-{self.layer_name}-loss.{self.loss_name}-lr.{self.lr}-batch.{self.batch_size}'.replace(' ', '')
        self.model_path = os.path.join(save_dir, '{}.h5'.format(self.model_name))
        self.model_fig = os.path.join(save_dir, '{}.png'.format(self.model_name))
        plot_model(self.model, self.model_fig, show_shapes=True)
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
        self.model = load_model(self.model_path, custom_objects={'weighted_mse': self.weighted_mse, 'r2': self.r2, 'exp2_mae': self.exp2_mae})
        print('Done')

    def predict(self, X, verbose=1, index=0):
        y = self.model.predict(X, verbose=verbose)
        if index == -1:
            return y
        elif index < y.shape[1]:
            return y[:, index]
        else:
            raise ValueError('Index out of range.')

    def get_intermediate_layer(self, layer_names, X, verbose=1):
        intermediate_layer = Model(inputs=self.model.input, outputs=[self.model.get_layer(name).output for name in layer_names])
        return intermediate_layer.predict(X, verbose=verbose)

    def evaluate(self, dgr, name, mode='full', ci=False):
        print('Calculating metrices of {} model ...'.format(name))
        res = dict()
        res[name+'_train'] = self.cal_metrics(y_true=dgr.resp_train, y_pred=dgr.pred_train, weight=dgr.weight_train, mode=mode, ci=ci)
        res[name+'_val'] = self.cal_metrics(y_true=dgr.resp_val, y_pred=dgr.pred_val, weight=dgr.weight_val, mode=mode, ci=ci)
        res[name+'_test'] = self.cal_metrics(y_true=dgr.resp_test, y_pred=dgr.pred_test, weight=dgr.weight_test, mode=mode, ci=ci)
        return pd.DataFrame(res).T

    def evaluate_mean_guess(self, dgr, mode='full', ci=False):
        print('Calculating metrices of mean guess model ...')
        res = dict()
        res['REF_MEAN_train'] = self.cal_metrics(y_true=dgr.resp_train, y_pred=np.ones(shape=dgr.resp_train.shape), weight=dgr.weight_train, mode=mode, ci=ci)
        res['REF_MEAN_val'] = self.cal_metrics(y_true=dgr.resp_val, y_pred=np.ones(shape=dgr.resp_val.shape), weight=dgr.weight_val, mode=mode, ci=ci)
        res['REF_MEAN_test'] = self.cal_metrics(y_true=dgr.resp_test, y_pred=np.ones(shape=dgr.resp_test.shape), weight=dgr.weight_test, mode=mode, ci=ci)
        return pd.DataFrame(res).T

    def evaluate_gaussian_sample(self, dgr, mode='full', epoch=10, ci=False):
        print('Calculating metrices of gaussian sampling model ', end='', flush=True)
        pred_train = np.array([np.random.normal(loc=miu, scale=np.sqrt(sigma2), size=epoch) for (miu, sigma2) in zip(dgr.resp_train, dgr.var_train)]).T
        pred_val = np.array([np.random.normal(loc=miu, scale=np.sqrt(sigma2), size=epoch) for (miu, sigma2) in zip(dgr.resp_val, dgr.var_val)]).T
        pred_test = np.array([np.random.normal(loc=miu, scale=np.sqrt(sigma2), size=epoch) for (miu, sigma2) in zip(dgr.resp_test, dgr.var_test)]).T
        res = []
        for i in range(epoch):
            print('.', end='', flush=True)
            tmp = dict()
            tmp['REF_GS_train'] = self.cal_metrics(y_true=dgr.resp_train, y_pred=pred_train[i,:], weight=dgr.weight_train, mode=mode, ci=ci)
            tmp['REF_GS_val'] = self.cal_metrics(y_true=dgr.resp_val, y_pred=pred_val[i,:], weight=dgr.weight_val, mode=mode, ci=ci)
            tmp['REF_GS_test'] = self.cal_metrics(y_true=dgr.resp_test, y_pred=pred_test[i,:], weight=dgr.weight_test, mode=mode, ci=ci)
            res.append(pd.DataFrame(tmp).T)
        res = pd.concat(res)
        print(' ')
        return res.groupby(by=res.index).mean()

    def cal_metrics(self, y_true, y_pred, weight, mode, ci):
        ptg_true = np.exp2(y_true) - 1
        ptg_pred = np.exp2(y_pred) - 1
        mae = np.mean(np.abs(ptg_true - ptg_pred) * weight)
        mape = np.mean(np.abs(ptg_true - ptg_pred))
        maple = np.mean(np.abs(y_true - y_pred))
        top10pi = self.top_inclusion(ptg_true, ptg_pred, 10)
        mape250 = self.top_mae(ptg_true, ptg_pred, weight, n=250)
        mape1000 = self.top_mae(ptg_true, ptg_pred, weight, n=1000)
        pci = self.concordance_index(ptg_true, ptg_pred, mode) if ci else None
        return pd.Series([mae, mape, maple, pci, top10pi, mape250, mape1000], index=['MAE', 'MAPE', 'MAPLE', 'PCI', 'PI10', 'MAPE250', 'MAPE1000'])


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
    def top_mae(y_true, y_pred, base, percentile=None, n=None):
        assert (percentile is None) != (n is None), 'set only either percentile or n.'
        if percentile is not None:
            cut = np.percentile(base, 100 - percentile)
        if n is not None:
            cut = np.sort(base)[-n]
        truth = y_true[base >= cut]
        pred = y_pred[base >= cut]
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
    def __init__(self, ts_len, t_len, geo_len, org_len, output_len, conv_pool_layer, fc_layer, loss, lr):
        '''
        conv_pool_layer:
            list of (filters, kernel_size, pool_size)
        fc_layer:
            list of (units, dropout, regularize)
        '''
        super().__init__(ts_len=ts_len, t_len=t_len, geo_len=geo_len, org_len=org_len, output_len=output_len, loss=loss, lr=lr)
        self.conv_pool_layer = conv_pool_layer
        self.fc_layer = fc_layer
        name_conv = '.'.join(['conv.' + '.'.join(map(str, l)) for l in self.conv_pool_layer])
        name_fc = '.'.join(['fc.' + '.'.join(map(str, l)) for l in self.fc_layer])
        self.layer_name = name_conv + '-' + name_fc
        super()._build(struc='cnn')

### ========================================================== ###
###                           LSTM                             ###
### ========================================================== ###
class lstm(neural_net):
    '''
    Long short term memory model
    '''
    def __init__(self, ts_len, t_len, geo_len, org_len, output_len, lstm_layer, fc_layer, loss, lr):
        '''
        lstm_layer:
            list of (units, dropout, recurrent_dropout)
        fc_layer:
            list of (units, dropout, regularize)
        '''
        super().__init__(ts_len=ts_len, t_len=t_len, geo_len=geo_len, org_len=org_len, output_len=output_len, loss=loss, lr=lr)
        self.lstm_layer = lstm_layer
        self.fc_layer = fc_layer
        name_lstm = '.'.join(['lstm.' + '.'.join(map(str, l)) for l in self.lstm_layer])
        name_fc = '.'.join(['fc.' + '.'.join(map(str, l)) for l in self.fc_layer])
        self.layer_name = name_lstm + '-' + name_fc
        super()._build(struc='lstm')

### ========================================================== ###
###                            CNN                             ###
### ========================================================== ###
class cnn_lstm(neural_net):
    '''
    Convolutional neural network.
    '''
    def __init__(self, ts_len, t_len, geo_len, org_len, output_len, conv_pool_layer, lstm_layer, fc_layer, loss, lr):
        '''
        conv_pool_layer:
            list of (filters, kernel_size, pool_size)
        fc_layer:
            list of (units, dropout, regularize)
        '''
        super().__init__(ts_len=ts_len, t_len=t_len, geo_len=geo_len, org_len=org_len, output_len=output_len, loss=loss, lr=lr)
        self.conv_pool_layer = conv_pool_layer
        self.lstm_layer = lstm_layer
        self.fc_layer = fc_layer
        name_conv = '.'.join(['conv.' + '.'.join(map(str, l)) for l in self.conv_pool_layer])
        name_lstm = '.'.join(['lstm.' + '.'.join(map(str, l)) for l in self.lstm_layer])
        name_fc = '.'.join(['fc.' + '.'.join(map(str, l)) for l in self.fc_layer])
        self.layer_name = name_conv + '-' + name_lstm + '-' + name_fc
        super()._build(struc = 'cnn_lstm')

### ========================================================== ###
###                           FNN                              ###
### ========================================================== ###
class fnn(neural_net):
    '''
    Feedforward neural network
    '''
    def __init__(self, ts_len, t_len, geo_len, org_len, output_len, fc_layer_ts, fc_layer, loss, lr):
        '''
        fc_layer_ts:
            list of (units, dropout, regularize)
        fc_layer:
            list of (units, dropout, regularize)
        '''
        super().__init__(ts_len=ts_len, t_len=t_len, geo_len=geo_len, org_len=org_len, output_len=output_len, loss=loss, lr=lr)
        self.fc_layer_ts = fc_layer_ts
        self.fc_layer = fc_layer
        name_ts = '.'.join(['fc.' + '.'.join(map(str, l)) for l in self.fc_layer_ts])
        name_fc = '.'.join(['fc.' + '.'.join(map(str, l)) for l in self.fc_layer])
        self.layer_name = name_ts + '-' + name_fc
        super()._build(struc='fnn')
