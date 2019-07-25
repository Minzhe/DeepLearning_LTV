###############################################################
###                      utility.py                         ###
###############################################################
import os
import pandas as pd
import datetime as dt

def create_dummy(x, prefix, min_counts=50):
    '''
    Create dummy variables for categorical features.
    '''
    s = pd.Series(x.copy())
    label_counts = s.value_counts()
    # minor label to others
    minor_labels = list((label_counts[label_counts < min_counts]).index)
    s[s.isin(minor_labels)] = 'others'
    dummy_var = pd.get_dummies(s, prefix=prefix)
    dummy_var.columns = dummy_var.columns.to_series().apply(lambda x: x.replace(' ', ''))
    return dummy_var

def parse_time(x):
    try:
        return dt.datetime.strptime(x, '%m/%d/%y').date()
    except ValueError:
        return None

def parse_int(x):
    try:
        return int(x)
    except ValueError:
        return None

def parse_float(x, keep=2):
    try:
        return round(float(x), 2)
    except ValueError:
        return None

def clean_weekly_gb_data(path):
    data = pd.read_csv(path)
    data['week_starting'] = data['week_starting'].apply(lambda x: parse_time(x))
    data = data.sort_values(by=['week_starting', 'org_uuid'])
    data['g_bookings'] = data['g_bookings'].apply(parse_float)
    data['n_invited'] = data['n_invited'].apply(parse_int)
    data['n_confirmed'] = data['n_confirmed'].apply(parse_int)
    data['n_trips'] = data['n_trips'].apply(parse_int)
    data['n_active_riders'] = data['n_active_riders'].apply(parse_int)
    out_path = os.path.join(os.path.dirname(path), 'weekly_gb.csv')
    data.to_csv(out_path, index=None)