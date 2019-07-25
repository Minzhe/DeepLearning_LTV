proj_dir = '/Users/minzhe.zhang/Documents/DL-LTV'
import os
import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.join(proj_dir, 'code'))
from model.neural_net import cnn, lstm
from model.dataGenerator import dataGenerator
from sklearn.model_selection import train_test_split
from utility import plotting as plot

#################################      main     #################################
data_path = os.path.join(proj_dir, 'data/processed/weekly_gb_wb52_wf_52_mingb100_sampleperorg10.csv')
data = pd.read_csv(data_path)
dgr = dataGenerator(gb_data=data)
X_train, X_test, y_train, y_test, weight_train, weight_test = dgr.split_train_test(test_size=0.25, random_state=1234, weight='base')
save_dir = os.path.join(proj_dir, 'code/model/archive')

### ----------------   model  ------------------ ###
nn_framework = 'cnn'
if nn_framework == 'cnn':
    model_name = 'simple_cnn'
    conv_pool_layer = [(64, 3, 'same', None),
                       (128, 3, 'same', 3),
                       (256, 3, 'same', 3),]
    fc_layer = [(512, 0.2, None),
                (256, 0.1, None),]
    model = cnn(input_len=X_train.shape[1], output_len=2, conv_pool_layer=conv_pool_layer, fc_layer=fc_layer, loss='mse', lr=1e-3)
elif nn_framework == 'lstm':
    model_name = 'simple_lstm'
    lstm_layer = [(128, 0.1, 0), (128, 0.1, 0),]
    fc_layer = [(128, 0.1, None),]
    model = lstm(input_len=X_train.shape[1], output_len=2, lstm_layer=lstm_layer, fc_layer=fc_layer, loss='mse', lr=1e-3)
model.fit_config(model_name=model_name, save_dir=save_dir, batch_size=128, epoch=5, tolerance=25)
# model.fit(X_train, y_train, X_test, y_test)
model.load()

### -------------------  evaluate  -------------------- ###
pred_train = model.predict(X_train)[:,0]
pred_test = model.predict(X_test)[:,0]
res = model.evaluate_model(y_train, y_test[:,0], pred_train, pred_test, weight_train=weight_train, weight_test=weight_test)
res_path = os.path.join(proj_dir, 'result/metrics/{}.csv'.format(model.model_name))
res.to_csv(res_path)
# plot
fig_path = os.path.join(proj_dir, 'result/img/{}.png'.format(model.model_name))
plot.plot_corr_train_val(y_train[:,0], y_test[:,0], pred_train, pred_test, fig_path)

