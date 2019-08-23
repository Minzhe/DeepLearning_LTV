proj_dir = '/Users/minzhe.zhang/Documents/DeepLearning_LTV'
import os
import numpy as np
import pandas as pd
import datetime as dt
import sys
sys.path.append(os.path.join(proj_dir, 'code'))
from model.neural_net_keras import cnn, lstm, fnn, cnn_lstm
from model.dataGenerator import dataGenerator

#################################      function     #################################
def model_const(dgr, frame):
    in_len = tuple(x.shape[1] for x in dgr.train['X'])
    if frame == 'cnn':
        conv_pool_layer = [(64, 3, None), (128, 3, 3), (256, 3, -1),]
        fc_layer = [(512, 0.3), (256, 0.1), (256, 0.1)]
        model = cnn(ts_len=in_len[:3], t_len=in_len[3], geo_len=in_len[4], org_len=in_len[5], output_len=1,
                    conv_pool_layer=conv_pool_layer, fc_layer=fc_layer, loss='wmse', lr=1e-3)
    elif frame == 'cnn_lstm':
        conv_pool_layer = [(48, 4, None), (96, 4, 4), (172, 3, 3),]
        lstm_layer = [(128, 0, 0),]
        fc_layer = [(320, 0.2), (256, 0.1)]
        model = cnn_lstm(ts_len=in_len[:3], t_len=in_len[3], geo_len=in_len[4], org_len=in_len[5], output_len=1,
                         conv_pool_layer=conv_pool_layer, lstm_layer=lstm_layer, fc_layer=fc_layer, loss='wmse', lr=1e-3)
    elif frame == 'lstm':
        # lstm_layer = [(128, 0, 0),]
        lstm_layer = [(64, 0, 0), (128, 0.1, 0),]
        fc_layer = [(384, 0.3), (256, 0.2), (128, 0.1)]
        model = lstm(ts_len=in_len[:3], t_len=in_len[3], geo_len=in_len[4], org_len=in_len[5], output_len=1,
                     lstm_layer=lstm_layer, fc_layer=fc_layer, loss='wmse', lr=1e-3)
    elif frame == 'fnn':
        fc_layer_ts = [(128, 0), (256, 0), (256, 0.1),]
        fc_layer = [(512, 0.2), (256, 0.1), (256, 0.1)]
        model = fnn(ts_len=in_len[:3], t_len=in_len[3], geo_len=in_len[4], org_len=in_len[5], output_len=1,
                    fc_layer_ts=fc_layer_ts, fc_layer=fc_layer, loss='wmse', lr=1e-3)
    else:
        raise ValueError('Unrecognized model framework.')
    return model

#################################      main     #################################
sample = -1
train_path = os.path.join(proj_dir, 'data/processed/weekly_ts_train_wb52_wf52_mh52_mgb20.csv')
test_path = os.path.join(proj_dir, 'data/processed/weekly_ts_test_wb52_wf52_mh2_mgb0.csv')
org_path = os.path.join(proj_dir, 'data/processed/org_data.csv')
out_path = os.path.join(proj_dir, 'data/processed/weekly_ts_52weeks_sample{}.pkl'.format(sample))
# dgr = dataGenerator(train_path, test_path, org_path, val_split=0.2, sample=sample, random_state=1234)
# dgr.save(out_path)
dgr = dataGenerator.load(out_path)
# exit()

### ----------------   model  ------------------ ###
save_dir = os.path.join(proj_dir, 'code/model/archive')
model_name = 'cnn_lstm'
model = model_const(dgr, model_name)
model.fit_config(model_name=model_name + '.52_52', save_dir=save_dir, batch_size=128, epoch=250, tolerance=40)
model.fit(dgr.train['X'], dgr.train['y'], dgr.val['X'], dgr.val['y'], keep_record=True)
model.load_weight()
exit()

### ----------------   make prediction   ------------------ ###
# pred_path = os.path.join(proj_dir, 'result/prediction/cnn_lstm_52weeks_2018-04-01.csv')
# print('predicting {}'.format(np.unique(dgr.test['week_starting'])))
# dgr.test['pred'] = model.predict(dgr.test['X'])
# res = pd.DataFrame({'org_uuid': dgr.test['org_uuid'], 'gb_last': dgr.test['base'] * 52, 'gb_true': dgr.test['y'][:, 0], 'gb_pred': dgr.test['pred']})
# res['gb_true'] = (np.exp2(res['gb_true']) - 1) * res['gb_last']
# res['gb_pred'] = (np.exp2(res['gb_pred']) - 1) * res['gb_last']
# res = res.loc[res['gb_true'] > 0,:]
# res['mape'] = (res['gb_true'] - res['gb_pred']) / res['gb_true']
# res.to_csv(pred_path, index=None)
# exit()

### -------------------  evaluate  -------------------- ###
# dgr.train['pred'] = model.predict(dgr.train['X'])
# dgr.val['pred'] = model.predict(dgr.val['X'])
# dgr.test['pred'] = model.predict(dgr.test['X'])
# mtc_model = model.evaluate(dgr, name=model_name.upper(), mode='sample')
# mtc_model = round(mtc_model, 4)
# res_path = os.path.join(proj_dir, 'result/metrics/{}.csv'.format(model.model_name))
# mtc_model.to_csv(res_path)

# mtc_mean = model.evaluate_mean_guess(dgr, mode='sample')
# mtc_gs = model.evaluate_gaussian_sample(dgr, mode='sample')
# mtc_ref = round(pd.concat([mtc_mean, mtc_gs]), 4)
# ref_path = os.path.join(proj_dir, 'result/metrics/ref.csv')
# mtc_ref.to_csv(ref_path)
# exit()

### -------------------  get intermediate layer  -------------------- ###
# fetr = model.get_intermediate_layer(layer_names=['ts_feature', 'geo_feature', 'org_feature'], X=dgr.X_val)
# fetr['week_starting'] = dgr.week_starting[dgr.val_idx]
# fetr['org_uuid'] = dgr.org_uuid[dgr.val_idx]
# fetr['resp'] = dgr.resp_val
# fetr['pred'] = model.predict(dgr.X_val)
# fetr['error'] = np.exp2(fetr['pred']) - np.exp2(fetr['resp'])
# res_path = os.path.join(proj_dir, 'result/metrics/feature_map.{}.52_52.val.csv'.format(model_name))
# fetr.to_csv(res_path, index=None)


# plot
# fig_path = os.path.join(proj_dir, 'result/img/{}.png'.format(model.model_name))
# plot.plot_corr_train_val(dgr.y_train[:,0], dgr.y_val[:,0], dgr.y_test[:,0], pred_train, pred_val, pred_test, fig_path, lims=(0,3))

