import pandas as pd
import numpy as np
from datetime import date, datetime


def get_train_test_data():
    try:
        db = pd.read_csv('Data/train.csv', parse_dates=["timestamp"]) 
    except:
        db = None
    train = db.query('building_id == 107')
    train = train.dropna()
   # del train['building_id']
    train = train.drop('building_id', axis = 1)
    test = train.query('anomaly == 1')
    train = train[train['anomaly'] == 0]
    train = train.drop('anomaly', axis = 1)
    test = test.drop('anomaly', axis = 1)
    #train.to_csv('C:/Users/ecbey/Downloads/train_building_107.csv')  
    # sec_origin_row = train.iloc[1:1]
    # sec_origin_ser = sec_origin_row['timestamp']
    #print(train.columns.tolist())
    train_sec_origin = train.iloc[0]['timestamp'].timestamp()
    test_sec_origin = test.iloc[0]['timestamp'].timestamp()
    train['seconds'] = train['timestamp'].map(lambda t: t.timestamp() - train_sec_origin )
    test['seconds'] = test['timestamp'].map(lambda t: t.timestamp() - test_sec_origin )
    hour_sec = 60 * 60
    day_sec = 24 * hour_sec
    month_sec = 30.4167 * day_sec
    

############ THIS IS THE ORIGINAL APPROACH ###########################
    train['month_sin'] = np.sin(train['seconds']) * (2 * np.pi / month_sec)
    train['month_cos'] = np.cos(train['seconds']) * (2 * np.pi / month_sec)
    train['day_sin'] = np.sin(train['seconds']) * (2 * np.pi / day_sec)
    train['day_cos'] = np.cos(train['seconds']) * (2 * np.pi / day_sec)
    train['hour_sin'] = np.sin(train['seconds']) * (2 * np.pi / hour_sec)
    train['hour_cos'] = np.cos(train['seconds']) * (2 * np.pi / hour_sec)

    test['month_sin'] = np.sin(test['seconds']) * (2 * np.pi / month_sec)
    test['month_cos'] = np.cos(test['seconds']) * (2 * np.pi / month_sec)
    test['day_sin'] = np.sin(test['seconds']) * (2 * np.pi / day_sec)
    test['day_cos'] = np.cos(test['seconds']) * (2 * np.pi / day_sec)
    test['hour_sin'] = np.sin(test['seconds']) * (2 * np.pi / hour_sec)
    test['hour_cos'] = np.cos(test['seconds']) * (2 * np.pi / hour_sec)
###########################################################################
    # test['month_sin'] = test['seconds'] / month_sec
    # test['month_cos'] = test['seconds'] / month_sec
    # test['day_sin'] = test['seconds'] / day_sec
    # test['day_cos'] = test['seconds'] / day_sec
    # test['hour_sin'] = test['seconds'] / hour_sec
    # test['hour_cos'] = test['seconds'] / hour_sec

    k = 0
    for index, row in train.iterrows(): 
        row.month_sin = row.month_sin + np.sin((float(k) / (10000.0 ** (2*0/6))))
        row.month_cos = row.month_cos + np.cos((float(k) / (10000.0 ** (2*1/6))))

        row.day_sin = row.day_sin + np.sin((float(k) / (10000.0 ** (2*2/6))))
        row.day_cos = row.day_cos + np.cos((float(k) / (10000.0 ** (2*3/6))))

        row.hour_sin = row.hour_sin + np.sin((float(k) / (10000.0 ** (2*4/6))))
        row.hour_cos = row.hour_cos + np.cos((float(k) / (10000.0 ** (2*5/6))))

    # for i in range(len(test)):
    #     a = test[i:'month_sin']

    train = train.drop('seconds', axis = 1)
    train = train.drop('timestamp', axis = 1)
    test = test.drop('seconds', axis = 1)
    test = test.drop('timestamp', axis = 1)
    return train, test