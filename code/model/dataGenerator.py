###############################################################################
###                            dataGenerator.py                             ###
###############################################################################
proj_dir = '/Users/minzhe.zhang/Documents/DeepLearning_LTV'
import os
import sys
sys.path.append(os.path.join(proj_dir, 'code'))
import numpy as np
import pickle as pkl
import utility.utility as util

class dataGenerator(object):
    def __init__(self, gb_data, trip_data, rider_data, org_data, random_state, samples=-1):
        # time series data
        print('Parsing gb_data', end='', flush=True)
        self.gb_data = gb_data.loc[gb_data['org_uuid'].isin(org_data['uuid'].tolist()),:]
        if samples > 0:
            if self.gb_data.shape[0] < samples:
                raise ValueError('Sample number exceeds limit.')
            np.random.seed(random_state)
            self.gb_data = self.gb_data.iloc[np.random.choice(self.gb_data.shape[0], size=samples, replace=False),:]
        self.gb_data, self.trip_data, self.rider_data = util.reorder_dataframe(self.gb_data, trip_data, rider_data, on=['org_uuid', 'week_starting'], sort_key=1)
        self.gb_data['n_confirm'] = util.truncate((self.gb_data['n_confirm'] / self.gb_data['n_invite']).fillna(0).replace(np.inf, 0), upper_bound=2)
        self.gb_data['n_invite'] = np.log2(self.gb_data['n_invite'] + 1)
        # date information
        print(', date_info', end='', flush=True)
        week_of_year = self.gb_data['week_starting'].apply(lambda x: x.isocalendar()[1] / 52).values.reshape(-1,1)
        month = self.gb_data['week_starting'].apply(lambda x: x.month / 12).values.reshape(-1,1)
        self.time_data = np.concatenate((week_of_year, month), axis=1)
        # geographical features
        print(', geo_data', end='', flush=True)
        self.org_data = org_data.set_index('uuid')
        self.geo_data, self.org_data = util.slice_columns(self.org_data, ['region', 'country', 'city'], return_left=True)
        self.geo_data = np.array(self.geo_data.loc[self.gb_data['org_uuid'],:])
        # add time independent feature to org
        print(', org_data', flush=True)
        self.org = np.array(self.org_data.loc[self.gb_data['org_uuid'],:])
        self.weight = util.truncate(np.array(self.gb_data['base']), upper_bound=3000)     # truncate base to maximum 3000
        self.log_weight = util.truncate(np.array(np.log2(self.weight - min(self.weight) + 2)), upper_bound=15)
        self.org = np.concatenate((self.org,
                                   self.gb_data[['n_invite', 'n_confirm']],
                                   self.log_weight.reshape(-1, 1),
                                   self.time_data), axis=1)
        self.gb_data = self.gb_data.drop(['n_invite', 'n_confirm'], axis=1)
        # reshape time series
        print('Reshaping time series')
        ts_gb = np.array(self.gb_data.iloc[:, 5:])
        ts_gb = np.reshape(ts_gb, (ts_gb.shape[0], ts_gb.shape[1], 1))
        ts_trip = np.array(self.trip_data.iloc[:, 5:])
        ts_trip = np.reshape(ts_trip, (ts_trip.shape[0], ts_trip.shape[1], 1))
        ts_rider = np.array(self.rider_data.iloc[:, 5:])
        ts_rider = np.reshape(ts_rider, (ts_rider.shape[0], ts_rider.shape[1], 1))
        self.X = [ts_gb, ts_trip, ts_rider, self.time_data, self.geo_data, self.org]
        self.y = self.gb_data['loggb']
        self.y_var = self.gb_data['loggb_var']
        self.week_starting = self.gb_data['week_starting']
        del self.gb_data, self.trip_data, self.rider_data, self.geo_data, self.org_data, self.org

    def split_train_val_test(self, validation_size, test_step, random_state=1234, weight='base'):
        print('Spliting training, validation and test dataset ...')
        # test index
        test_time = self.week_starting.unique()[-test_step:]
        test_idx = np.array(self.week_starting.isin(test_time))
        # validation index
        np.random.seed(random_state)
        val_idx = np.random.choice(self.y[~test_idx].shape[0], int(self.y[~test_idx].shape[0] * validation_size), replace=False)
        val_idx = np.array([i in val_idx for i in range(self.y.shape[0])])
        train_idx = (~test_idx) & (~val_idx)
        # combine response and weight
        if weight == 'base':
            resp = np.array([self.y, self.weight / min(self.weight)]).T
        elif weight == 'log_base':
            resp = np.array([self.y, self.log_weight]).T
        else:
            raise ValueError('Unrecognizable weight: {}'.format(weight))
        # combine ts and org features, response
        self.X_train, self.y_train, self.resp_train = util.stack_slice(self.X, train_idx), resp[train_idx], resp[train_idx][:,0]
        self.X_val, self.y_val, self.resp_val = util.stack_slice(self.X, val_idx), resp[val_idx], resp[val_idx][:,0]
        self.X_test, self.y_test, self.resp_test = util.stack_slice(self.X, test_idx), resp[test_idx], resp[test_idx][:,0]
        # weight and variance
        self.weight_train, self.var_train = self.weight[train_idx], self.y_var[train_idx]
        self.weight_val, self.var_val = self.weight[val_idx], self.y_var[val_idx]
        self.weight_test, self.var_test = self.weight[test_idx], self.y_var[test_idx]

    def subset_X_train(self, index):
        return [self.X_train[0][:, -index:], self.X_train[1]]

    def subset_X_val(self, index):
        return [self.X_val[0][:, -index:], self.X_val[1]]

    def subset_X_test(self, index):
        return [self.X_test[0][:, -index:], self.X_test[1]]

    def save(self, path):
        print('Writing out data ...')
        del self.X, self.y, self.weight, self.log_weight, self.y_var, self.week_starting
        with open(path, 'wb') as f:
            pkl.dump({'dgr': self}, file=f)

    @staticmethod
    def load(path):
        print('Loading data ...')
        with open(path, 'rb') as f:
            return pkl.load(f)['dgr']


