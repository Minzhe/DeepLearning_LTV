proj_dir = '/Users/minzhe.zhang/Documents/DeepLearning_LTV'
import os
import numpy as np
import pandas as pd

yuan_path = os.path.join(proj_dir, 'result/prediction/1yGB_2018-04-01.csv')
nn_path = os.path.join(proj_dir, 'result/prediction/cnn_lstm_52weeks_2018-04-01.csv')

yuan = pd.read_csv(yuan_path)
nn = pd.read_csv(nn_path)
yuan['mape'] = np.abs(yuan['gb_true'] - yuan['gb_pred']) / yuan['gb_true']
nn['mape'] = np.abs(nn['gb_true'] - nn['gb_pred']) / nn['gb_true']
res = nn[['org_uuid', 'gb_true', 'mape']].merge(yuan[['org_uuid', 'mape']], how='inner', on=['org_uuid'], suffixes=['_nn', '_yuan'])
res = res[['org_uuid', 'gb_true', 'mape_nn', 'mape_yuan']].sort_values(['gb_true'], ascending=False).head(200)
print(res.mean())
