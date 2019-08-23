proj_dir = '/Users/minzhe.zhang/Documents/DeepLearning_LTV'
import os
import sys
sys.path.append(os.path.join(proj_dir, 'code'))
import numpy as np
import pandas as pd
import functools
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import utility.plotting as pf
from model.dataGenerator import dataGenerator

#########################       function      #########################
def build_lda(feature_path, model_path, feature):
    data = pd.read_csv(feature_path)
    data.loc[data['resp'] > 2, 'resp'] = 2
    data['error'] = data['error'].apply(lambda x: -1 if x < -1 else x if x < 1 else 1)
    data = data.sample(frac=1)
    X = data.loc[:, data.columns.to_series().str.contains(feature)]
    # lda model
    lda = LinearDiscriminantAnalysis(n_components=2)
    y = data['resp'].apply(lambda x: 0 if x < 0.8 else 1 if x < 1.2 else 2)
    lda.fit(X, y)
    coord = lda.transform(X)
    with open(model_path, 'wb') as f:
        pkl.dump({'lda': lda, 'X': X, 'coord': coord, 'info': data[['week_starting', 'org_uuid', 'resp', 'pred', 'error']]}, file=f)

def build_lda_gb(gb_path, model_path):
    data = pd.read_csv(gb_path)
    data.loc[data['loggb'] > 2, 'loggb'] = 2
    data = data.sample(frac=1)
    X = data.loc[:, data.columns.to_series().str.contains('t_')]
    # lda model
    lda = LinearDiscriminantAnalysis(n_components=2)
    y = data['loggb'].apply(lambda x: 0 if x < 0.8 else 1 if x < 1.2 else 2)
    lda.fit(X, y)
    coord = lda.transform(X)
    with open(model_path, 'wb') as f:
        pkl.dump({'lda': lda, 'X': X, 'coord': coord, 'info': data[['week_starting', 'org_uuid', 'loggb', 'loggb_var']]}, file=f)

def plot_lda_component(model_path, fig_path):
    with open(model_path, 'rb') as f:
        model = pkl.load(f)
    coord = model['coord']
    cmap = plt.cm.get_cmap('RdYlGn')
    pf.plot_scatter(coord[:, 0], coord[:, 1], s=8, c=model['info']['resp'], cmap=cmap, fig_path=fig_path)

def get_lda_subregion(model_path, region, radius, out_path, feature, gb_path=None, geo_path=None):
    print('Reading orginal data ...')
    with open(model_path, 'rb') as f:
        model = pkl.load(f)
    coord = model['coord']
    info = model['info']
    idx = [(abs(coord[:, i] - region[i]) < radius) for i in range(len(region))]
    idx = functools.reduce(lambda x, y: (x & y), idx)
    coord = pd.DataFrame(coord[idx,:], columns=['PC'+str(i) for i in range(len(region))])
    df = pd.concat([info.loc[idx,:].reset_index(drop=True), coord], axis=1)
    # original ts
    if feature == 'ts':
        gb_ts = pd.read_csv(gb_path)
        df = df.merge(gb_ts, how='inner', on=['org_uuid', 'week_starting'])
    if feature == 'geo':
        geo_org = pd.read_csv(geo_path)
        geo_org.columns = [col.replace('uuid', 'org_uuid') for col in geo_org.columns]
        df = df.merge(geo_org, how='inner', on=['org_uuid'])
    df.to_csv(out_path, index=None)

def plot_ts_4(up_path, flat_path, down_path, mix_path, fig_path):
    fig, ax = plt.subplots(4, 1, figsize=(16, 12))
    col = ['#2ca02c', '#ff7f0e', '#d62728', '#580194']
    for i, path in enumerate([up_path, flat_path, down_path, flat_path]):
        data = pd.read_csv(path)
        data = data.loc[:, [col for col in data.columns if 't_' in col]].sample(10)
        data.columns = list(range(data.shape[1]))
        ax[i].plot(data.T, c=col[i])
        ax[i].axhline(y=1, ls='--', c='grey')
    fig.savefig(fig_path, transparent=True)

def plot_ts_2(up_path, down_path, fig_path):
    fig, ax = plt.subplots(2, 1, figsize=(16, 6))
    col = ['#2ca02c', '#d62728']
    for i, path in enumerate([up_path, down_path]):
        data = pd.read_csv(path)
        n = min(10, data.shape[0])
        data = data.loc[:, [col for col in data.columns if 't_' in col]].sample(n)
        data.columns = list(range(data.shape[1]))
        ax[i].plot(data.T, c=col[i])
        ax[i].axhline(y=1, ls='--', c='grey')
    fig.savefig(fig_path, transparent=True)

#########################      lda model      ###########################
feature = 'ts'
feature_path = os.path.join(proj_dir, 'result/data/feature_map.cnn_lstm.52_52.val.csv')
lda_path = os.path.join(proj_dir, 'code/model/archive/lda.{}.val.cnn_lstm.52_52.pkl'.format(feature))
lda_gb_path = os.path.join(proj_dir, 'code/model/archive/lda.{}.original_gb.val.cnn_lstm.52_52.pkl'.format(feature))
gb_path = os.path.join(proj_dir, 'data/processed/weekly_loggb_trip_rider_wb52_wf52_mh104_ms10_spo-1.val.csv')
geo_path = os.path.join(proj_dir, 'data/processed/org_data.csv')

# build_lda(feature_path, lda_path, feature)
build_lda_gb(gb_path, lda_gb_path)

### plot componenet
fig_path = os.path.join(proj_dir, 'result/img/lda.ts.val.cnn_lstm.52_52.png')
# plot_lda_component(lda_path, fig_path)
fig_path = os.path.join(proj_dir, 'result/img/lda.ts.gb.val.cnn_lstm.52_52.png')
plot_lda_component(lda_gb_path, fig_path)
exit()
### get lda subregion
# ----- ts ----- #
down_path = os.path.join(proj_dir, 'result/data/lda_comp.ts.down.cnn_lstm.52_52.val.csv')
# get_lda_subregion(lda_path, region=(-3, 3), radius=0., out_path=down_path, feature='ts', gb_path=gb_path)
up_path = os.path.join(proj_dir, 'result/data/lda_comp.ts.up.cnn_lstm.52_52.val.csv')
# get_lda_subregion(lda_path, region=(3, 4), radius=0.5, out_path=up_path, feature='ts', gb_path=gb_path)
mix_path = os.path.join(proj_dir, 'result/data/lda_comp.ts.mix.cnn_lstm.52_52.val.csv')
# get_lda_subregion(lda_path, region=(0, 2), radius=0.5, out_path=mix_path, feature='ts', gb_path=gb_path)
flat_path = os.path.join(proj_dir, 'result/data/lda_comp.ts.flat.cnn_lstm.52_52.val.csv')
# get_lda_subregion(lda_path, region=(0.5, -2.5), radius=0.5, out_path=flat_path, feature='ts, gb_path=gb_path)
mix_up_path = os.path.join(proj_dir, 'result/data/lda_comp.ts.mix_up.cnn_lstm.52_52.val.csv')
# get_lda_subregion(lda_path, region=(0, 2, 2.8), radius=0.7, out_path=mix_up_path, feature='ts', gb_path=gb_path)
mix_down_path = os.path.join(proj_dir, 'result/data/lda_comp.ts.mix_down.cnn_lstm.52_52.val.csv')
# get_lda_subregion(lda_path, region=(0, 2, -2.5), radius=0.7, out_path=mix_down_path, feature='ts', gb_path=gb_path)
# ----- geo ----- #
geo_up_path = os.path.join(proj_dir, 'result/data/lda_comp.geo.up.cnn_lstm.52_52.val.csv')
# get_lda_subregion(lda_path, region=(2, 2), radius=1, out_path=geo_up_path, feature='geo', geo_path=geo_path)
geo_down_path = os.path.join(proj_dir, 'result/data/lda_comp.geo.down.cnn_lstm.52_52.val.csv')
# get_lda_subregion(lda_path, region=(-4, 2), radius=1, out_path=geo_down_path, feature='geo', geo_path=geo_path)
# geo features
# up = pd.read_csv(geo_up_path).iloc[:,7:-14]
# down = pd.read_csv(geo_down_path).iloc[:, 7:-14]
# df = pd.concat([up.mean(), down.mean()], axis=1)
# df['diff'] = df.iloc[:,0] - df.iloc[:,1]
# print(df.sort_values(by=['diff']))

### plot lda subregion time series
fig_path = os.path.join(proj_dir, 'result/img/ts_gb.lda_ts.val.cnn_lstm.52_52.png')
# plot_ts_4(up_path, flat_path, down_path, mix_path, fig_path=fig_path)
fig_path = os.path.join(proj_dir, 'result/img/ts_gb_mix.lda_ts.val.cnn_lstm.52_52.png')
plot_ts_2(mix_up_path, mix_down_path, fig_path=fig_path)