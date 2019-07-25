#############################################################################
###                               cnn.py                                  ###
#############################################################################
import os
import numpy as np
import pandas as pd
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.regularizers import l1, l2
import keras.backend as K
import tensorflow as tf


#################################    model    ###################################
### ========================================================== ###
###                            DNN                             ###
### ========================================================== ###
class nn(object):
    '''
    Neural network model.
    '''
    def __init__(self, input_len, output_len, loss, lr):
        self.input_len = input_len
        if loss == 'mse':
            self.loss = 'mse'
        elif loss == 'weighted_mse':
            self.loss = self.weighted_mse
        else:
            raise ValueError('Unrecognizable loss function.')
        self.output_len = output_len
        self.lr = lr
        self.layer_name = ''

    def build(self, inputs, outputs):
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss=self.loss, optimizer=Adam(lr=self.lr), metrics=[self.r2, self.mape])
        print('Model structure summary:', flush=True)
        print(model.summary())
        return model

    def fit_config(self, model_name, save_dir, batch_size, epoch, tolerance):
        self.batch_size = batch_size
        self.epoch = epoch
        self.tol = tolerance
        self.model_name = f'{model_name}@in_{self.input_len}|out_{self.output_len}|{self.layer_name}|loss_{self.loss}|lr_{self.lr}|batch_{self.batch_size}'
        self.model_path = os.path.join(save_dir, '{}.h5'.format(self.model_name))
        self.log_dir = os.path.join(save_dir, '{}.log'.format(self.model_name))
        if not os.path.isdir(self.log_dir): os.mkdir(self.log_dir)

    def fit(self, X_train, y_train, X_test, y_test):
        early_stopper = EarlyStopping(patience=self.tol, verbose=1)
        check_point = ModelCheckpoint(self.model_path, verbose=1, save_best_only=True)
        reduce_learner = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=int(self.tol*0.7), cooldown=int(self.tol*0.2), min_lr=1e-4, verbose=1)
        train_log = TensorBoard(log_dir=self.log_dir)
        # fit
        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[early_stopper, check_point, train_log, reduce_learner],
                       batch_size=self.batch_size, epochs=self.epoch, verbose=1, shuffle=True)

    def load(self):
        print('Loading neural network model ... ', end='', flush=True)
        self.model = load_model(self.model_path, custom_objects={'r2': self.r2, 'mape': self.mape})
        print('Done')

    def predict(self, X, verbose=1):
        return self.model.predict(X, verbose=verbose)

    def evaluate_model(self, y_train, y_test, pred_train, pred_test, weight_train, weight_test):
        model_train = self.evaluate(y_true=y_train, y_pred=pred_train, weight=weight_train)
        model_test = self.evaluate(y_true=y_test, y_pred=pred_test, weight=weight_test)
        ref_train = self.evaluate(y_true=y_train, y_pred=np.ones(shape=pred_train.shape), weight=weight_train)
        ref_test = self.evaluate(y_true=y_test, y_pred=np.ones(shape=pred_test.shape), weight=weight_test)
        return pd.DataFrame({'model_train': model_train, 'model_test': model_test, 'ref_train': ref_train, 'ref_test': ref_test}).T

    def evaluate(self, y_true, y_pred, weight):
        ptg_true = np.exp2(y_true) - 2
        ptg_pred = np.exp2(y_pred) - 2
        mae = self.mean_absolute_error(ptg_true, ptg_pred, weight)
        mape = self.mean_absolute_percentage_error(ptg_true, ptg_pred)
        mse = self.mean_square_error(ptg_true, ptg_pred, weight)
        mspe = self.mean_square_percentage_error(ptg_true, ptg_pred)
        ci = self.concordance_index(ptg_true, ptg_pred, weight)
        pci = self.percentage_concordance_index(ptg_true, ptg_pred)
        return pd.Series([mae, mape, mse, mspe, ci, pci], index=['mae', 'mape', 'mse', 'mspe', 'ci', 'pci'])

    # --------------  utility function  ---------------- #
    def mean_absolute_error(self, y_true, y_pred, base):
        return np.round(np.mean(np.abs(y_true - y_pred) * base), 4)

    def mean_absolute_percentage_error(self, y_true, y_pred):
        return np.round(np.mean(np.abs(y_true - y_pred)), 4)

    def mean_square_error(self, y_true, y_pred, base):
        return np.round(np.mean(np.square(y_true - y_pred) * base), 4)

    def mean_square_percentage_error(self, y_true, y_pred):
        return np.round(np.mean(np.square(y_true - y_pred)), 4)

    def concordance_index(self, y_true, y_pred, base):
        return self.percentage_concordance_index(y_true * base, y_pred * base)

    def percentage_concordance_index(self, y_true, y_pred):
        seq = np.array(y_pred)[np.argsort(y_true)]
        mat = seq.reshape(1, -1) - seq.reshape(-1, 1)
        score = mat[np.triu_indices(mat.shape[0], 1)]
        ci = (np.sum(score > 0) + 0.5 * np.sum(score == 0)) / len(score)
        print(ci)
        return np.round(ci, 4)

    @staticmethod
    def r2(truth, pred):
        y_true = tf.slice(truth, [0, 0], [-1, 1])
        y_pred = tf.slice(pred, [0, 0], [-1, 1])
        ss_res = K.sum(K.square(y_true - y_pred))
        ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return 1 - ss_res / (ss_tot + K.epsilon())

    @staticmethod
    def weighted_mse(truth, pred):
        y_true = tf.slice(truth, [0, 0], [-1, 1])
        y_pred = tf.slice(pred, [0, 0], [-1, 1])
        w = tf.slice(truth, [0, 1], [-1, 1])
        return K.mean(K.square(y_true - y_pred) * w, axis=-1)

    @staticmethod
    def mape(truth, pred):
        base = tf.constant(2, dtype='float32')
        y_true = tf.math.pow(base, tf.slice(truth, [0, 0], [-1, 1]))
        y_pred = tf.math.pow(base, tf.slice(pred, [0, 0], [-1, 1]))
        return K.mean(K.abs(y_true - y_pred))

### ========================================================== ###
###                            CNN                             ###
### ========================================================== ###
class cnn(nn):
    '''
    Convolutional neural network.
    '''
    def __init__(self, input_len, output_len, conv_pool_layer, fc_layer, loss, lr):
        '''
        conv_pool_layer:
            tuple of (filters, kernel_size, padding, pool_size)
        fc_layer:
            tuple of (dropout, units)
        '''
        super().__init__(input_len=input_len, output_len=output_len, loss=loss, lr=lr)
        self.conv_pool_layer = conv_pool_layer
        self.fc_layer = fc_layer
        name_conv = '.'.join(['conv.' + '.'.join(map(str, l)) for l in self.conv_pool_layer])
        name_fc = '.'.join(['fc.' + '.'.join(map(str, l)) for l in self.fc_layer])
        self.layer_name = name_conv + '|' + name_fc
        self.model = self._init_model()

    def _init_model(self):
        '''
        Initialize convolutional neural network model
        '''
        print('Initializing cnn model ...', flush=True)
        inputs = Input(shape=(self.input_len, 1), name='ts_input')
        # convolution layer
        ts_layer = inputs
        for filters, kernel_size, padding, pool_size in self.conv_pool_layer:
            ts_layer = Conv1D(filters=filters, kernel_size=kernel_size, padding=padding, activation='relu', strides=1)(ts_layer)
            if isinstance(pool_size, int):
                if pool_size > 0:
                    ts_layer = MaxPooling1D(pool_size=pool_size)(ts_layer)
                elif pool_size == -1:
                    ts_layer = GlobalMaxPooling1D()(ts_layer)
            elif pool_size is not None:
                raise ValueError('Unrecognizable pool_size: {}'.format(pool_size))
        # fully connected layer
        fc_layer = ts_layer if len(K.int_shape(ts_layer)) == 2 else Flatten()(ts_layer)
        for units, dropout, regularize in self.fc_layer:
            if regularize is not None:
                fc_layer = Dense(units=units, activation='relu', kernel_regularizer=l2(regularize), activity_regularizer=l1(regularize))(fc_layer)
            else:
                fc_layer = Dense(units=units, activation='relu')(fc_layer)
            if dropout is not None:
                fc_layer = Dropout(dropout)(fc_layer)
        output = Dense(self.output_len, activation='linear')(fc_layer)
        # model
        return super().build(inputs=inputs, outputs=output)

### ========================================================== ###
###                           LSTM                             ###
### ========================================================== ###
class lstm(nn):
    '''
    Long short term memory model
    '''
    def __init__(self, input_len, output_len, lstm_layer, fc_layer, loss, lr):
        super().__init__(input_len=input_len, output_len=output_len, loss=loss, lr=lr)
        self.lstm_layer = lstm_layer
        self.fc_layer = fc_layer
        self.layer_name = 'lstm.' + '.'.join(map(str, lstm_layer))
        self.model = self._init_model()

    def _init_model(self):
        '''
        Initialize LSTM model
        '''
        print('Initializing lstm model ...', flush=True)
        # lstm layer
        inputs = Input(shape=(self.input_len, 1), name='ts_input')
        ts_layer = inputs
        for i, (units, drop, rec_drop) in enumerate(self.lstm_layer):
            if i < len(self.lstm_layer) - 1:
                ts_layer = LSTM(units=units, return_sequences=True, dropout=drop, recurrent_dropout=rec_drop)(ts_layer)
            else:
                ts_layer = LSTM(units=units, return_sequences=False, dropout=drop, recurrent_dropout=rec_drop)(ts_layer)
        # fully connected layer
        fc_layer = ts_layer
        for units, dropout, regularize in self.fc_layer:
            if regularize is not None:
                fc_layer = Dense(units=units, activation='relu', kernel_regularizer=l2(regularize), activity_regularizer=l1(regularize))(fc_layer)
            else:
                fc_layer = Dense(units=units, activation='relu')(fc_layer)
            if dropout is not None:
                fc_layer = Dropout(dropout)(fc_layer)
        output = Dense(self.output_len, activation='linear')(fc_layer)
        # model
        return super().build(inputs=inputs, outputs=output)



