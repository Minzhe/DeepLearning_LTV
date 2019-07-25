####################################################################
###                    process_gb_data.py                        ###
####################################################################
proj_dir = '/Users/minzhe.zhang/Documents/DL-LTV'
import numpy as np
import pandas as pd
import datetime as dt
import os
import sys
sys.path.append(os.path.join(proj_dir, 'code'))
import utility.utility as util

############################    function   ##############################
class gbDataGenerator(object):
    def __init__(self, df):
        self.raw_data = df

    def create_time_series(self, min_history, t0):
        self.t0 = t0
        self.min_history = min_history
        print('Making time series table ...')
        data = self.raw_data.loc[self.raw_data['week_starting'] >= t0,:]
        org_counts = data['org_uuid'].value_counts()
        orgs = org_counts[org_counts > min_history].index.tolist()
        data = data.loc[data['org_uuid'].isin(orgs),:]
        data['week_starting'] = data['week_starting'].apply(lambda t: (t - t0).days)
        self.gb_ts = data.pivot(index='org_uuid', columns='week_starting', values='g_bookings').fillna(0)
        self.check_day_consecutive(self.gb_ts.columns.tolist())

    def generate_training_instance(self, w_back, w_forecast, min_weekly_gb):
        self.w_back = w_back
        self.w_forecast = w_forecast
        self.min_weekly_gb = min_weekly_gb
        print('Generating training instance ...')
        inst = []
        for org_uuid, row in self.gb_ts.iterrows():
            ts = self.drop_na_and_first_week(np.array(row))
            inst.append(self.generate_instance_from_ts(ts, w_back, w_forecast, min_weekly_gb, org_uuid))
        self.instance_pool = pd.concat(inst)

    def sample_train_test_data(self, sample_per_org):
        self.sample_per_org = sample_per_org
        print('Sampling data for training and testing ...')
        org_counts = self.instance_pool['org_uuid'].value_counts()
        orgs = org_counts[org_counts > sample_per_org].index.tolist()
        train_data = self.instance_pool.loc[self.instance_pool['org_uuid'].isin(orgs),:]
        train_data = train_data.groupby(by=['org_uuid'], as_index=False).apply(lambda df: df.loc[np.random.choice(df.index.tolist(), sample_per_org, replace=False),:])
        return train_data.reset_index().drop(['level_0', 'level_1'], axis=1)

    # *****************   help function   *************** #
    @staticmethod
    def generate_instance_from_ts(arr, w_b, w_f, min_base, org_uuid):
        inst = []
        for s in range(0, len(arr) - w_b - w_f + 1):
            time_window = arr[s:s + w_b]
            base = np.mean(time_window)
            gb = np.sum(arr[s + w_b:s + w_b + w_f])
            if base >= min_base:
                tmp = np.concatenate(([base], [gb / base / w_f], time_window / base))
                inst.append([org_uuid] + list(np.round(tmp, 4)))
        return pd.DataFrame(inst, columns=['org_uuid', 'base', 'gb'] + ['t_' + str(i) for i in range(w_b)])

    @staticmethod
    def drop_na_and_first_week(arr):
        flag = 0
        for i in range(len(arr)):
            if arr[i] != 0:
                if flag == 0:
                    flag = 1
                else:
                    return arr[i:]
        return []

    @staticmethod
    def check_day_consecutive(arr):
        if not all(pd.Series(arr).diff().dropna() == 7):
            raise ValueError('The time series table does not have consecutive weeks.')


###########################    main   ###########################
data_path = os.path.join(proj_dir, 'data/processed/weekly_gb.csv')
data = pd.read_csv(data_path, parse_dates=[0])
dgr = gbDataGenerator(data)
dgr.create_time_series(min_history=104, t0=dt.datetime(2016, 1, 3))
dgr.generate_training_instance(w_back=52, w_forecast=52, min_weekly_gb=100)
train_data = dgr.sample_train_test_data(sample_per_org=10)
out_path = os.path.join(proj_dir, 'data/processed/weekly_gb_wb{}_wf_{}_mingb{}_sampleperorg{}.csv'.format(52, 52, 100, 10))
train_data.to_csv(out_path, index=None)


