####################################################################
###                    process_gb_data.py                        ###
####################################################################
proj_dir = '/Users/minzhe.zhang/Documents/DeepLearning_LTV'
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

    def create_time_series(self, min_history: int, t0):
        self.t0 = t0
        self.min_history = min_history
        print('Making time series table ...')
        data = self.raw_data.loc[self.raw_data['week_starting'] >= t0,:]
        org_counts = data['org_uuid'].value_counts()
        orgs = org_counts[org_counts > min_history].index.tolist()
        data = data.loc[data['org_uuid'].isin(orgs),:]
        data['week_starting'] = data['week_starting'].apply(lambda t: (t - t0).days)
        self.gb_ts = data.pivot(index='org_uuid', columns='week_starting', values='g_bookings').fillna(0)
        self.trip_ts = data.pivot(index='org_uuid', columns='week_starting', values='n_trips').fillna(0)
        self.rider_ts = data.pivot(index='org_uuid', columns='week_starting', values='n_active_riders').fillna(0)
        self.invite_ts = data.pivot(index='org_uuid', columns='week_starting', values='n_invited').fillna(0).cumsum(axis=1).reset_index()
        self.invite_ts = self.invite_ts.melt(id_vars=['org_uuid'], var_name='week_starting', value_name='n_invite')
        self.confirm_ts = data.pivot(index='org_uuid', columns='week_starting', values='n_confirmed').fillna(0).cumsum(axis=1).reset_index()
        self.confirm_ts = self.confirm_ts.melt(id_vars=['org_uuid'], var_name='week_starting', value_name='n_confirm')
        self.invite_confirm = self.invite_ts.merge(self.confirm_ts, how='outer', on=['org_uuid', 'week_starting'])
        self.check_day_consecutive(self.gb_ts.columns.tolist())

    def generate_training_instance(self, w_back: int, w_forecast: int, min_weekly_gb: int):
        self.w_back = w_back
        self.w_forecast = w_forecast
        self.min_weekly_gb = min_weekly_gb
        print('Generating gb_ts training instance ...')
        inst = []
        for org_uuid, row in self.gb_ts.iterrows():
            ts, week_starting = self.drop_na_and_first_week(row)
            inst.append(self.slice_ts_window(ts, self.w_back, self.w_forecast, self.min_weekly_gb, org_uuid, week_starting, 'gb'))
        self.gb_instance = pd.concat(inst)
        self.gb_instance = self.gb_instance.merge(self.invite_confirm, how='left', on=['org_uuid', 'week_starting'])
        print('Generating trip_ts training instance ...')
        inst = []
        for org_uuid, row in self.trip_ts.iterrows():
            ts, week_starting = self.drop_na_and_first_week(row)
            inst.append(self.slice_ts_window(ts, self.w_back, self.w_forecast, 0, org_uuid, week_starting, 'trip'))
        self.trip_instance = pd.concat(inst)
        print('Generating rider_ts training instance ...')
        inst = []
        for org_uuid, row in self.rider_ts.iterrows():
            ts, week_starting = self.drop_na_and_first_week(row)
            inst.append(self.slice_ts_window(ts, self.w_back, self.w_forecast, 0, org_uuid, week_starting, 'rider'))
        self.rider_instance = pd.concat(inst)

    def sample_train_test_data(self, org_min_sample: int, sample_per_org: int):
        self.org_min_sample = org_min_sample
        self.sample_per_org = sample_per_org
        print('Sampling data for training and testing ...')
        # subset organization count
        if org_min_sample > 1:
            org_counts = self.gb_instance['org_uuid'].value_counts()
            orgs = org_counts[org_counts > org_min_sample].index.tolist()
            gb_data = self.gb_instance.loc[self.gb_instance['org_uuid'].isin(orgs),:]
        else:
            gb_data = self.gb_instance.copy()
        # sample instance
        if sample_per_org > 1:
            gb_data = gb_data.groupby(by=['org_uuid'], as_index=False).apply(lambda df: df.loc[np.random.choice(df.index.tolist(), sample_per_org, replace=False),:])
            gb_data = gb_data.reset_index().drop(['level_0', 'level_1'], axis=1)
        self.gb_data, self.trip_data, self.rider_data = self.get_common_org_week(gb_data, self.trip_instance, self.rider_instance, self.t0)

    def write_out(self, dir_path):
        print('Writing output ...')
        gb_path = os.path.join(dir_path, f'weekly_loggb_wb{self.w_back}_wf_{self.w_forecast}_mh{self.min_history}_ms{self.org_min_sample}_spo{self.sample_per_org}.csv')
        trip_path = os.path.join(dir_path, f'weekly_logtrip_wb{self.w_back}_wf_{self.w_forecast}_mh{self.min_history}_ms{self.org_min_sample}_spo{self.sample_per_org}.csv')
        rider_path = os.path.join(dir_path, f'weekly_logrider_wb{self.w_back}_wf_{self.w_forecast}_mh{self.min_history}_ms{self.org_min_sample}_spo{self.sample_per_org}.csv')
        self.gb_data.to_csv(gb_path, index=None)
        self.trip_data.to_csv(trip_path, index=None)
        self.rider_data.to_csv(rider_path, index=None)

    # *****************   help function   *************** #
    @staticmethod
    def get_common_org_week(gb, trip, rider, t0):
        gb_data, trip_data, rider_data = gb.copy(), trip.copy(), rider.copy()
        gb_data.index = list(zip(gb_data['org_uuid'], gb_data['week_starting']))
        trip_data.index = list(zip(trip_data['org_uuid'], trip_data['week_starting']))
        rider_data.index = list(zip(rider_data['org_uuid'], rider_data['week_starting']))
        index = list(set(gb_data.index) & set(trip_data.index) & set(rider_data.index))
        gb_data = gb_data.loc[index,:].reset_index(drop=True)
        trip_data = trip_data.loc[index,:].reset_index(drop=True)
        rider_data = rider_data.loc[index,:].reset_index(drop=True)
        gb_data.loc[:,'week_starting'] = gb_data['week_starting'].apply(lambda x: (dt.timedelta(x) + t0).strftime('%Y-%m-%d'))
        trip_data.loc[:,'week_starting'] = trip_data['week_starting'].apply(lambda x: (dt.timedelta(x) + t0).strftime('%Y-%m-%d'))
        rider_data.loc[:,'week_starting'] = rider_data['week_starting'].apply(lambda x: (dt.timedelta(x) + t0).strftime('%Y-%m-%d'))
        return gb_data, trip_data, rider_data

    @staticmethod
    def slice_ts_window(arr, w_b, w_f, min_base, org_uuid, week_starting, name):
        inst = []
        for s in range(0, len(arr) - w_b - w_f + 1):
            week = week_starting + s * 7
            time_window = arr[s:s + w_b]
            forecast_window = arr[s + w_b:s + w_b + w_f]
            base = np.mean(time_window)
            if base <= min_base: continue
            # normalize by the base value
            time_window = np.log2(time_window / base + 1)
            gb = np.mean(np.log2(forecast_window / base + 1))                   # sample mean: miu
            gb_var = np.var(np.log2(forecast_window / base + 1), ddof=1) / w_f    # variance of sample mean: miu = sigma2 / n
            tmp = np.concatenate(([week, base, gb, gb_var], time_window))
            inst.append([org_uuid] + list(np.round(tmp, 6)))
        return pd.DataFrame(inst, columns=['org_uuid', 'week_starting', 'base', 'log{}'.format(name), 'log{}_var'.format(name)] + ['t_' + str(i) for i in range(w_b)])

    @staticmethod
    def drop_na_and_first_week(arr):
        flag = 0
        for i in range(len(arr)):
            if arr.iloc[i] != 0:
                if flag == 0:
                    flag = 1
                else:
                    ts = arr.iloc[i:]
                    return np.array(ts), ts.index[0]
        return []

    @staticmethod
    def check_day_consecutive(arr):
        if not all(pd.Series(arr).diff().dropna() == 7):
            raise ValueError('The time series table does not have consecutive weeks.')


###########################    main   ###########################
data_path = os.path.join(proj_dir, 'data/processed/weekly_gb.csv')
data = pd.read_csv(data_path, parse_dates=[0])
dgr = gbDataGenerator(data)
dgr.create_time_series(min_history=56, t0=dt.datetime(2016, 1, 3))
dgr.generate_training_instance(w_back=4, w_forecast=52, min_weekly_gb=50)
dgr.sample_train_test_data(org_min_sample=5, sample_per_org=-1)
out_dir = os.path.join(proj_dir, 'data/processed/')
dgr.write_out(out_dir)


