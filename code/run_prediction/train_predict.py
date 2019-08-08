proj_dir = '/Users/minzhe.zhang/Documents/DeepLearning_LTV'
import os
import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.join(proj_dir, 'code'))
from model.neural_net_keras import cnn, lstm, fnn
from model.dataGenerator import dataGenerator
from utility import plotting as plot

#################################      function     #################################
def generate_data(gb_path, trip_path, rider_path, org_path, out_path, samples, random_state):
    gb_data = pd.read_csv(gb_path)
    trip_data = pd.read_csv(trip_path)
    rider_data = pd.read_csv(rider_path)
    org_data = pd.read_csv(org_path)
    dgr = dataGenerator(gb_data=gb_data, trip_data=trip_data, rider_data=rider_data, org_data=org_data, samples=samples, random_state=random_state)
    dgr.split_train_val_test(validation_size=0.2, test_step=8, random_state=random_state, weight='log_base')
    dgr.save(out_path)

def model_const(dgr, frame):
    ts_len = tuple(x.shape[1] for x in dgr.X_train[:-1])
    in_len = dgr.X_train[-1].shape[1]
    if frame == 'cnn':
        conv_pool_layer = [(128, 3, None), (128, 3, 3), (256, 3, -1),]
        fc_layer = [(512, 0.2, None), (256, 0.1, None), (256, 0.1, None)]
        model = cnn(ts_len=ts_len, input_len=in_len, output_len=1, conv_pool_layer=conv_pool_layer, fc_layer=fc_layer, loss='weighted_mse', lr=1e-3)
    elif frame == 'lstm':
        lstm_layer = [(128, 0.1, 0), (128, 0.1, 0),]
        fc_layer = [(512, 0.1, None), (256, 0.1, None), (256, 0.1, None)]
        model = lstm(ts_len=ts_len, input_len=in_len, output_len=1, lstm_layer=lstm_layer, fc_layer=fc_layer, loss='weighted_mse', lr=1e-3)
    elif frame == 'fnn':
        fc_layer_ts = [(128, None, None), (256, None, None), (256, 0.1, None),]
        fc_layer = [(512, 0.1, None), (256, 0.1, None), (256, 0.1, None)]
        model = fnn(ts_len=ts_len, input_len=in_len, output_len=1, fc_layer_ts=fc_layer_ts, fc_layer=fc_layer, loss='weighted_mse', lr=1e-3)
    else:
        raise ValueError('Unrecognized model framework.')
    return model

#################################      main     #################################
gb_path = os.path.join(proj_dir, 'data/processed/weekly_loggb_wb4_wf_52_mh56_ms5_spo-1.csv')
trip_path = os.path.join(proj_dir, 'data/processed/weekly_logtrip_wb4_wf_52_mh56_ms5_spo-1.csv')
rider_path = os.path.join(proj_dir, 'data/processed/weekly_logrider_wb4_wf_52_mh56_ms5_spo-1.csv')
org_path = os.path.join(proj_dir, 'data/processed/org_data.csv')
out_path = os.path.join(proj_dir, 'data/processed/weekly_gb_wb4_wf_52_mh56_ms5_sample-1.pkl')
# generate_data(gb_path, trip_path, rider_path, org_path, out_path, samples=-1, random_state=1234)
dgr = dataGenerator.load(out_path)
# print(dgr.X_train[0].shape, dgr.X_val[0].shape, dgr.X_test[0].shape)
# exit()

### ----------------   model  ------------------ ###
save_dir = os.path.join(proj_dir, 'code/model/archive')
model_name = 'lstm'
model = model_const(dgr, model_name)
model.fit_config(model_name=model_name+'.4_52', save_dir=save_dir, batch_size=128, epoch=250, tolerance=40)
model.fit(dgr.X_train, dgr.y_train, dgr.X_val, dgr.y_val)
model.load()
exit()

### -------------------  evaluate  -------------------- ###
# dgr.pred_train = model.predict(dgr.X_train)
# dgr.pred_val = model.predict(dgr.X_val)
# dgr.pred_test = model.predict(dgr.X_test)
# mtc_model = model.evaluate(dgr, name=model_name.upper(), mode='sample')
# mtc_model = round(mtc_model, 4)
# res_path = os.path.join(proj_dir, 'result/metrics/{}.csv'.format(model.model_name))
# mtc_model.to_csv(res_path)

mtc_mean = model.evaluate_mean_guess(dgr, mode='sample')
mtc_gs = model.evaluate_gaussian_sample(dgr, mode='sample')
mtc_ref = round(pd.concat([mtc_mean, mtc_gs]), 4)
ref_path = os.path.join(proj_dir, 'result/metrics/reference_3ts.csv')
mtc_ref.to_csv(ref_path)

# plot
# fig_path = os.path.join(proj_dir, 'result/img/{}.png'.format(model.model_name))
# plot.plot_corr_train_val(dgr.y_train[:,0], dgr.y_val[:,0], dgr.y_test[:,0], pred_train, pred_val, pred_test, fig_path, lims=(0,3))

