####################################################################
###                   process_org_data.py                        ###
####################################################################
proj_dir = '/Users/minzhe.zhang/Documents/DeepLearning_LTV'
import numpy as np
import pandas as pd
import datetime as dt
import os
import sys
sys.path.append(os.path.join(proj_dir, 'code'))
import utility.utility as util

#########################     main    ############################
data_path = os.path.join(proj_dir, 'data/raw/eligible_organization.csv')
data = pd.read_csv(data_path, index_col=0)

region = util.create_dummy(data['mega_region'], prefix='region')
country = util.create_dummy(data['country_name'], prefix='country', min_counts=1000)
city = util.create_dummy(data['city_name'], prefix='city', min_counts=500)
acquire = util.create_dummy(data['acquisition_type'], prefix='acquire')
bill = util.create_dummy(data['billing_mode'], prefix='bill')
collection = util.create_dummy(data['collection_type'], prefix='collection')
pay = util.create_dummy(data['payment_type'], prefix='pay')

close_date = data['close_date'].apply(util.parse_time)
upgrade_date = data['upgraded_at'].apply(util.parse_time)
first_invite_date = data['first_invite_at'].apply(util.parse_time)
first_trip_date = data['first_travel_trip_at'].apply(util.parse_time)

upgrade_date = (upgrade_date - close_date).apply(lambda x: x.days) / 365
if_upgrade = upgrade_date.isnull().astype(int)
upgrade_date = upgrade_date.fillna(0)

first_invite_date = (first_invite_date - close_date).apply(lambda x: x.days) / 365
if_invite = first_invite_date.isnull().astype(int)
first_invite_date = first_invite_date.fillna(0)

first_trip_date = (first_trip_date - close_date).apply(lambda x: x.days) / 365
if_first_trip = first_trip_date.isnull().astype(int)
first_trip_date = first_trip_date.fillna(0)

date_df = pd.DataFrame({'upgrade_date': upgrade_date, 'upgrade': if_upgrade,
                        'first_invite_date': first_invite_date, 'invite': if_invite,
                        'first_trip_date': first_trip_date, 'first_trip': if_first_trip})

org_data = pd.concat([region, country, city, acquire, bill, collection, pay, date_df], axis=1)
if np.sum(np.isnan(org_data.values)) != 0:
    raise ValueError('DataFrame contains missing value.')
out_path = os.path.join(proj_dir, 'data/processed/org_data.csv')
org_data.to_csv(out_path)