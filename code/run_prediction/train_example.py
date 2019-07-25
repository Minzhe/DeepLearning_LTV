proj_dir = '/Users/minzhe.zhang/Documents/DL-LTV'
import os
import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.join(proj_dir, 'code'))
from model.neural_net import cnn_model, lstm_model
import matplotlib.pyplot as plt

################################     function     ################################
def generate_train_test(window_size, normalize):
    if normalize:
        X, y, base = [], [], []
        for i in range(len(data) - window_size):
            s = data[i:i + window_size]
            b = np.mean(s)
            X.append(s / b)
            y.append(data[i + window_size] / b)
            base.append(b)
        X, y, base = np.array(X), np.array(y), np.array(base)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        X_train, X_test = X[:int(X.shape[0] * 0.75),:,:], X[int(X.shape[0] * 0.75):,:,:]
        y_train, y_test = y[:int(y.shape[0] * 0.75)], y[int(y.shape[0] * 0.75):]
        base_train, base_test = base[:int(y.shape[0] * 0.75)], base[int(y.shape[0] * 0.75):]
        truth = pd.Series(y, index=data.index[window_size:]) * base
        return X_train, X_test, y_train, y_test, base_train, base_test, truth
    else:
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        X_train, X_test = X[:int(X.shape[0] * 0.75),:,:], X[int(X.shape[0] * 0.75):,:,:]
        y_train, y_test = y[:int(y.shape[0] * 0.75)], y[int(y.shape[0] * 0.75):]
        truth = pd.Series(y, index=data.index[window_size:])
        return X_train, X_test, y_train, y_test, truth


#################################      main     #################################
data_path = os.path.join(proj_dir, 'data/airline-passengers.csv')
data = pd.read_csv(data_path, index_col=0, squeeze=True)

X_train, X_test, y_train, y_test, base_train, base_test, truth = generate_train_test(window_size=12, normalize=True)
save_dir = os.path.join(proj_dir, 'code/model/archive')

nn_framework = 'lstm'
if nn_framework == 'cnn':
    ### ----------  cnn model  ---------- ###
    model_name = 'example_cnn'
    conv_pool_layer = [(64, 3, 'same', 'relu', None),
                       (128, 3, 'same', 'relu', 2),]
    fc_layer = [(128, 'relu', None, None),
                (64, 'relu', None, None)]
    model = cnn_model(input_len=X_train.shape[1], conv_pool_layer=conv_pool_layer, fc_layer=fc_layer, loss='mse', lr=3e-4)

elif nn_framework == 'lstm':
    ## ---------  lstm model  ---------- ###
    model_name='example_lstm'
    lstm_layer=[(128, 0.1, 0), (128, 0.1, 0),]
    fc_layer = [(128, 'relu', 0.1, None),]
    model = lstm_model(input_len=X_train.shape[1], lstm_layer=lstm_layer, fc_layer=fc_layer, loss='mse', lr=1e-3)

model.fit_config(model_name=model_name, save_dir=save_dir, batch_size=32, epoch=300, tolerance=30)
model.fit(X_train, y_train, X_test, y_test)
model.load()

pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

pred_train = pd.Series(pred_train * base_train, index=truth.index[:len(pred_train)])
pred_test = pd.Series(pred_test * base_test, index=truth.index[-len(pred_test):])

fig_path = os.path.join(proj_dir, 'result/img/example.png')
plt.plot(truth, label='Truth')
plt.plot(pred_train, label='Train')
plt.plot(pred_test, label='Test')
plt.legend(loc='upper left')
plt.xticks('')
plt.savefig(fig_path, transparent=True)

