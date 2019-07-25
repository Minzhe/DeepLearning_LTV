####################################################################
###                   process_org_data.py                        ###
####################################################################
proj_dir = '/Users/minzhe.zhang/Documents/DL-LTV'
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
upgrade_time = data['upgraded_at'].apply(util.parse_time)
first_invite_date = data['first_invite_at'].apply(util.parse_time)
first_trip_date = data['first_travel_trip_at'].apply(util.parse_time)
upgrade = (upgrade_time - close_date).apply(lambda x: x.days) / 365
if_upgrade = upgrade.isnull().astype(int)
upgrade = upgrade.fillna(0)
first_invite = (first_invite_date - close_date).apply(lambda x: x.days) / 365
if_invite = first_invite.isnull().astype(int)
first_invite = first_invite.fillna(0)
first_trip = (first_trip_date - close_date).apply(lambda x: x.days) / 365

org_data = pd.concat([region, country, city, acquire, bill, collection, pay,
                      pd.DataFrame({'upgrade_day': upgrade, 'upgrade': if_upgrade, 'first_invite': first_invite, 'invite': if_invite, 'first_trip': first_trip})], axis=1)

out_path = os.path.join(proj_dir, 'data/processed/org_data.csv')
org_data.to_csv(out_path)