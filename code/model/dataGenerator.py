###############################################################################
###                            dataGenerator.py                             ###
###############################################################################
proj_dir = '/Users/minzhe.zhang/Documents/DeepLearning_LTV'
import os
import sys
sys.path.append(os.path.join(proj_dir, 'code'))
import numpy as np
import pandas as pd
import pickle as pkl
import utility.utility as util

class dataGenerator(object):
    def __init__(self, train_path, test_path, org_path, val_split, sample=-1, random_state=1234):
        # time series data
        train_data = pd.read_csv(train_path, parse_dates=[2])
        test_data = pd.read_csv(test_path, parse_dates=[2])
        org_data = pd.read_csv(org_path)
        print('Reading training data ...')
        self.train = self.generate_X_y(train_data, org_data)
        print('Reading testing data ...')
        self.test = self.generate_X_y(test_data, org_data)
        self.train, self.val = self.split_train_val(self.train, val_split, sample, random_state)

    def split_train_val(self, data, val_split, sample, random_state):
        print('Spliting training, validation dataset ...')
        # sample data
        if sample > 0 and sample < data['y'].shape[0]:
            print('Sampling {} data ...'.format(sample))
            np.random.seed(random_state)
            idx = np.random.choice(data['y'].shape[0], size=sample, replace=False)
            data = self.subset_data(data, idx)
        # split data based on org_uuid
        uuid = np.unique(data['org_uuid'])
        val_id = np.random.choice(uuid, size=int(len(uuid) * val_split), replace=False)
        val_idx = np.isin(data['org_uuid'], val_id)
        val_data = self.subset_data(data, val_idx)
        train_data = self.subset_data(data, ~val_idx)
        assert len(set(train_data['org_uuid']) & set(val_data['org_uuid'])) == 0, 'duplicated id in train and val set.'
        print('Train data: {} orgs, {} instance.'.format(len(np.unique(train_data['org_uuid'])), train_data['y'].shape[0]))
        print('Train data: {} orgs, {} instance.'.format(len(np.unique(val_data['org_uuid'])), val_data['y'].shape[0]))
        return train_data, val_data

    def save(self, path):
        print('Writing out data ...')
        with open(path, 'wb') as f:
            pkl.dump({'dgr': self}, file=f)

    @staticmethod
    def load(path):
        print('Loading data ...')
        with open(path, 'rb') as f:
            return pkl.load(f)['dgr']

    @staticmethod
    def generate_X_y(ts_data, org_data):
        print('Parsing ts_data', end='', flush=True)
        ts_data = ts_data.loc[ts_data['org_uuid'].isin(org_data['uuid'].tolist()),:]
        gb_data = ts_data.loc[ts_data['ts'] == 'gb',:].drop(['ts'], axis=1)
        trip_data = ts_data.loc[ts_data['ts'] == 'trip',:].drop(['ts', 'n_confirm', 'n_invite'], axis=1)
        rider_data = ts_data.loc[ts_data['ts'] == 'rider',:].drop(['ts', 'n_confirm', 'n_invite'], axis=1)
        gb_data, trip_data, rider_data = util.reorder_dataframe(gb_data, trip_data, rider_data, on=['org_uuid', 'week_starting'], sort_key=1)
        gb_data['n_confirm'] = util.truncate((gb_data['n_confirm'] / gb_data['n_invite']).fillna(0).replace(np.inf, 0), upper_bound=2)
        gb_data['n_invite'] = np.log2(gb_data['n_invite'] + 1)
        # date information
        print(', date_info', end='', flush=True)
        week_of_year = gb_data['week_starting'].apply(lambda x: x.isocalendar()[1] / 52).values.reshape(-1,1)
        month = gb_data['week_starting'].apply(lambda x: x.month / 12).values.reshape(-1,1)
        time_data = np.concatenate((week_of_year, month), axis=1)
        # geographical features
        print(', geo_data', end='', flush=True)
        org_info = org_data.set_index('uuid')
        geo_info, org_info = util.slice_columns(org_info, ['region', 'country', 'city'], return_left=True)
        geo_info = np.array(geo_info.loc[gb_data['org_uuid'],:])
        # add time independent feature to org
        print(', org_data', flush=True)
        org_info = np.array(org_info.loc[gb_data['org_uuid'],:])
        base = np.array(gb_data['base'])
        weight = util.truncate(np.array(np.log2(base - min(base) + 2)), upper_bound=15)
        org_info = np.concatenate((org_info,
                                   gb_data[['n_invite', 'n_confirm']],
                                   np.log2(base.reshape(-1, 1) + 1),
                                   np.log2(trip_data['base'].values.reshape(-1, 1) + 1),
                                   np.log2(rider_data['base'].values.reshape(-1, 1) + 1)), axis=1)
        gb_data = gb_data.drop(['n_invite', 'n_confirm'], axis=1)
        # reshape time series
        print('Reshaping time series')
        ts_gb = np.array(gb_data.iloc[:, 5:])
        ts_gb = np.reshape(ts_gb, (ts_gb.shape[0], ts_gb.shape[1], 1))
        ts_trip = np.array(trip_data.iloc[:, 5:])
        ts_trip = np.reshape(ts_trip, (ts_trip.shape[0], ts_trip.shape[1], 1))
        ts_rider = np.array(rider_data.iloc[:, 5:])
        ts_rider = np.reshape(ts_rider, (ts_rider.shape[0], ts_rider.shape[1], 1))
        X = [ts_gb, ts_trip, ts_rider, time_data, geo_info, org_info]
        y = np.array([gb_data['logratio'], weight]).T
        y_sigma = np.array(gb_data['sigma'])
        week_starting = np.array(gb_data['week_starting'])
        org_uuid = np.array(gb_data['org_uuid'])
        return {'X': X, 'y': y, 'y_sigma': y_sigma, 'base': base, 'week_starting': week_starting, 'org_uuid': org_uuid}

    @staticmethod
    def subset_data(data, index):
        res = dict()
        res['X'] = [x[index,:] for x in data['X']]
        res['y'] = data['y'][index,:]
        res['y_sigma'] = data['y_sigma'][index]
        res['base'] = data['base'][index]
        res['week_starting'] = data['week_starting'][index]
        res['org_uuid'] = data['org_uuid'][index]
        return res


