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
import multiprocessing as mp
import utility.utility as util

############################    function   ##############################
class gbDataGenerator(object):
    def __init__(self, df):
        self.raw_data = df

    def create_time_series(self, min_history, t0, t1):
        print('Making time series table ...')
        self.t0 = t0
        self.t1 = t1
        self.min_history = min_history
        data = self.raw_data.loc[(self.raw_data['week_starting'] >= t0) & (self.raw_data['week_starting'] <= t1),:]
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

    def generate_instance(self, w_back: int, w_forecast: int, min_weekly_gb: int):
        self.w_back = w_back
        self.w_forecast = w_forecast
        self.min_weekly_gb = min_weekly_gb
        manager = mp.Manager()
        return_dict = manager.dict()
        # generating function
        def inst_generator(df, name, min_val, return_dict):
            print('Generating {} training instance ...'.format(name))
            print('Num of orgs: {}, length of weeks: {}'.format(*df.shape))
            inst = []
            for org_uuid, row in df.iterrows():
                ts, week_starting = self.drop_na_and_first_week(row)
                if len(ts) < (w_back + w_forecast): continue
                inst.append(self.slice_ts_window(ts, self.w_back, self.w_forecast, min_val, org_uuid, week_starting))
            return_dict[name] = pd.concat(inst)
        # multiprocessing
        jobs = []
        p = mp.Process(target=inst_generator, args=(self.gb_ts, 'gb', self.min_weekly_gb, return_dict)); jobs.append(p); p.start()
        p = mp.Process(target=inst_generator, args=(self.trip_ts, 'trip', 0, return_dict)); jobs.append(p); p.start()
        p = mp.Process(target=inst_generator, args=(self.rider_ts, 'rider', 0, return_dict)); jobs.append(p); p.start()
        for proc in jobs:
            proc.join()
        # store instance
        self.gb_instance = return_dict['gb'].merge(self.invite_confirm, how='left', on=['org_uuid', 'week_starting'])
        self.trip_instance = return_dict['trip']
        self.rider_instance = return_dict['rider']
        del return_dict

    def sample_instance(self, org_min_sample: int, sample_per_org: int):
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
        self.ts_data = self.get_common_org_week(gb_data, self.trip_instance, self.rider_instance, self.t0)

    def write_out(self, dir_path, name):
        print('Writing output ...')
        ts_path = os.path.join(dir_path, f'weekly_ts_{name}_wb{self.w_back}_wf{self.w_forecast}_mh{self.min_history}_mgb{self.min_weekly_gb}.csv')
        self.ts_data.to_csv(ts_path, index=None)

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
        gb_data.insert(0, 'ts', 'gb'); trip_data.insert(0, 'ts', 'trip'); rider_data.insert(0, 'ts', 'rider')
        print('Merging gb, trip and rider data ...')
        ts_data = gb_data.merge(trip_data, how='outer').merge(rider_data, how='outer')
        ts_data = ts_data.sort_values(by=['week_starting', 'org_uuid', 'ts'])
        ts_data.loc[:, 'week_starting'] = ts_data['week_starting'].apply(lambda x: (dt.timedelta(x) + t0).strftime('%Y-%m-%d'))
        return ts_data

    @staticmethod
    def slice_ts_window(arr, w_b, w_f, min_base, org_uuid, week_starting):
        inst = []
        for s in range(0, len(arr) - w_b - w_f + 1):
            week = week_starting + (s + w_b) * 7
            time_window = arr[s:s + w_b]
            forecast_window = arr[s + w_b:s + w_b + w_f]
            base = np.mean(time_window)
            if base <= min_base: continue
            # normalize by the base value
            time_window = np.log2(time_window / base + 1)
            gb = np.log2(np.mean(forecast_window / base) + 1)                         # !!! take mean first, then log transform
            gb_sigma = np.sqrt(np.var(forecast_window / base, ddof=1) / w_f)          # variance of sample mean: miu = sigma2 / n
            tmp = np.concatenate(([week, base, gb, gb_sigma], time_window))
            inst.append([org_uuid] + list(np.round(tmp, 4)))
        return pd.DataFrame(inst, columns=['org_uuid', 'week_starting', 'base', 'logratio', 'sigma'] + ['t_' + str(i) for i in range(w_b)])

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
        return [], None

    @staticmethod
    def check_day_consecutive(arr):
        if not all(pd.Series(arr).diff().dropna() == 7):
            raise ValueError('The time series table does not have consecutive weeks.')


###########################    main   ###########################
data_path = os.path.join(proj_dir, 'data/processed/weekly_gb.csv')
data = pd.read_csv(data_path, parse_dates=[0])
# -------------------- 52 weeks data -------------------- #
# make training and validation
dgr = gbDataGenerator(data)
dgr.create_time_series(min_history=52, t0=dt.datetime(2014, 4, 27), t1=dt.datetime(2018, 9, 30))
dgr.generate_instance(w_back=52, w_forecast=52, min_weekly_gb=20)
dgr.sample_instance(org_min_sample=-1, sample_per_org=-1)
out_dir = os.path.join(proj_dir, 'data/processed/')
dgr.write_out(out_dir, 'train')
# make test
# dgr = gbDataGenerator(data)
# test_time = dt.datetime(2018, 4, 1)
# dgr.create_time_series(min_history=2, t0=test_time-dt.timedelta(7*53), t1=test_time+dt.timedelta(7*51))
# dgr.generate_instance(w_back=52, w_forecast=52, min_weekly_gb=0)
# dgr.sample_instance(org_min_sample=-1, sample_per_org=-1)
# out_dir = os.path.join(proj_dir, 'data/processed/')
# dgr.write_out(out_dir, 'test')



