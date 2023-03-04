import pandas as pd
import numpy as np
from datetime import date, datetime
from sklearn.model_selection import train_test_split


def get_train_test_data():  
    try:
        db = pd.read_csv('Data/train.csv', parse_dates=["timestamp"]) 
    except:
        db = None
    train = db.query('building_id == 107')
    train = train.dropna()
   # del train['building_id']
    train = train.drop('building_id', axis = 1)
    anomalies = train.query('anomaly == 1')
    train = train[train['anomaly'] == 0]
    train = train.drop('anomaly', axis = 1)
    anomalies = anomalies.drop('anomaly', axis = 1)
    #train.to_csv('C:/Users/ecbey/Downloads/train_building_107.csv')  
    # sec_origin_row = train.iloc[1:1]
    # sec_origin_ser = sec_origin_row['timestamp']
    #print(train.columns.tolist())
    train_sec_origin = train.iloc[0]['timestamp'].timestamp()
    anomalies_sec_origin = anomalies.iloc[0]['timestamp'].timestamp()
    train['seconds'] = train['timestamp'].map(lambda t: t.timestamp() - train_sec_origin )
    anomalies['seconds'] = anomalies['timestamp'].map(lambda t: t.timestamp() - anomalies_sec_origin )
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

    anomalies['month_sin'] = np.sin(anomalies['seconds']) * (2 * np.pi / month_sec)
    anomalies['month_cos'] = np.cos(anomalies['seconds']) * (2 * np.pi / month_sec)
    anomalies['day_sin'] = np.sin(anomalies['seconds']) * (2 * np.pi / day_sec)
    anomalies['day_cos'] = np.cos(anomalies['seconds']) * (2 * np.pi / day_sec)
    anomalies['hour_sin'] = np.sin(anomalies['seconds']) * (2 * np.pi / hour_sec)
    anomalies['hour_cos'] = np.cos(anomalies['seconds']) * (2 * np.pi / hour_sec)
###########################################################################
    # test['month_sin'] = test['seconds'] / month_sec
    # test['month_cos'] = test['seconds'] / month_sec
    # test['day_sin'] = test['seconds'] / day_sec
    # test['day_cos'] = test['seconds'] / day_sec
    # test['hour_sin'] = test['seconds'] / hour_sec
    # test['hour_cos'] = test['seconds'] / hour_sec

    k = 0
    #CONSTANT = 10000.0
    #CONSTANT = float(len(train. index)) there are missing places in the series
    CONSTANT = 5500.00
    for index, row in train.iterrows(): 
        row.month_sin = row.month_sin + np.sin((float(k) / (CONSTANT ** (2*0/6))))
        row.month_cos = row.month_cos + np.cos((float(k) / (CONSTANT ** (2*1/6))))

        row.day_sin = row.day_sin + np.sin((float(k) / (CONSTANT ** (2*2/6))))
        row.day_cos = row.day_cos + np.cos((float(k) / (CONSTANT ** (2*3/6))))

        row.hour_sin = row.hour_sin + np.sin((float(k) / (CONSTANT ** (2*4/6))))
        row.hour_cos = row.hour_cos + np.cos((float(k) / (CONSTANT ** (2*5/6))))

    # for i in range(len(test)):
    #     a = test[i:'month_sin']

    train = train.drop('seconds', axis = 1)
    train = train.drop('timestamp', axis = 1)
    anomalies = anomalies.drop('seconds', axis = 1)
    anomalies = anomalies.drop('timestamp', axis = 1)
    return train, anomalies

def generate_train_test_anomaly_data():  
    try:
        db = pd.read_csv('Data/train.csv', parse_dates=["timestamp"]) 
    except:
        db = None
    train = db.query('building_id == 107')
    train = train.dropna()
    #train.to_csv('C:/Users/ecbey/Downloads/data_building_107.csv')  
    train = train.loc[:895988]
    #train = train.loc[986188:]
    sample_sec_origin = train.iloc[0]['timestamp'].timestamp()
    
    train['time_sec'] = train['timestamp'].map(lambda t: (t.timestamp() - sample_sec_origin) / 3600 )
    sample_time_sec_end = train.iloc[-1]['time_sec']
    train = train.drop('building_id', axis = 1)
    anomalies = train.query('anomaly == 1')
    train = train[train['anomaly'] == 0]
    train = train.drop('anomaly', axis = 1)
    anomalies = anomalies.drop('anomaly', axis = 1)
    #train.to_csv('C:/Users/ecbey/Downloads/train_building_107.csv')  
    hour_sec = 60 * 60
    day_sec = 24 * hour_sec
    month_sec = 30.4167 * day_sec
    

############ GENERATING DERIVED DATE FIELDS ###########################
#MONTH
    train['year_quarter_1'] = train['timestamp'].map(lambda t: 1 if (datetime.fromtimestamp(t.timestamp())).month >= 1 and (datetime.fromtimestamp(t.timestamp())).month < 4 else 0.0001 )
    train['year_quarter_2'] = train['timestamp'].map(lambda t: 1 if (datetime.fromtimestamp(t.timestamp())).month > 3 and (datetime.fromtimestamp(t.timestamp())).month < 7 else 0.0001 )
    train['year_quarter_3'] = train['timestamp'].map(lambda t: 1 if (datetime.fromtimestamp(t.timestamp())).month > 6 and (datetime.fromtimestamp(t.timestamp())).month < 10 else 0.0001 )
    train['year_quarter_4'] = train['timestamp'].map(lambda t: 1 if (datetime.fromtimestamp(t.timestamp())).month > 9 and (datetime.fromtimestamp(t.timestamp())).month <= 12 else 0.0001 )

#DAY
    train['day_holliday'] = train['timestamp'].map(lambda t: 1 if (datetime.fromtimestamp(t.timestamp())).isoweekday() == 6 or (datetime.fromtimestamp(t.timestamp())).isoweekday() == 6 else 0.0001 )

#HOUR
    train['day_midnight'] = train['timestamp'].map(lambda t: 1 if (datetime.fromtimestamp(t.timestamp())).hour >= 0 and (datetime.fromtimestamp(t.timestamp())).hour < 7 else 0.0001 )
    train['day_morning'] = train['timestamp'].map(lambda t: 1 if (datetime.fromtimestamp(t.timestamp())).hour > 6 and (datetime.fromtimestamp(t.timestamp())).hour <= 12 else 0.0001 )
    train['day_afternoon'] = train['timestamp'].map(lambda t: 1 if (datetime.fromtimestamp(t.timestamp())).hour > 12 and (datetime.fromtimestamp(t.timestamp())).hour <= 18 else 0.0001 )
    train['day_night'] = train['timestamp'].map(lambda t: 1 if (datetime.fromtimestamp(t.timestamp())).hour > 18 and (datetime.fromtimestamp(t.timestamp())).hour <= 23 else 0.0001 )

#  output = output_start + ((output_end - output_start) / (input_end - input_start)) * (input - input_start)
#  output_end = 1.3 output_start = 0.2   1.1
#  output = 0.2 + (1.1 / (input_end - input_start)) * (input - input_start)
#  TRANSF:  e^x^3  ==> e^(output^3)
    
    output_end = 1.3
    #output_end = 1.5
    output_start = 0.2 
    tranf = 0.0
    for index, row in train.iterrows():     
        output = output_start + ((output_end -output_start ) / (sample_time_sec_end - 0.0001)) * (float(row['time_sec']) - 0.0001)
        out_inv = output_end - (output -output_start)  
        tranf = (np.e ** (float(-out_inv)**3))/5 #5 is to diminish the importance of this wheight
        train.loc[index,'year_quarter_1'] = row.year_quarter_1 * tranf
        train.loc[index,'year_quarter_2'] = row.year_quarter_2 * tranf
        train.loc[index,'year_quarter_3'] = row.year_quarter_3 * tranf
        train.loc[index,'year_quarter_4'] = row.year_quarter_4 * tranf
        train.loc[index,'day_holliday'] = row.day_holliday  * tranf
        train.loc[index,'day_midnight'] = row.day_midnight  * tranf
        train.loc[index,'day_morning'] = row.day_morning  * tranf
        train.loc[index,'day_afternoon'] = row.day_afternoon  * tranf
        train.loc[index,'day_night'] = row.day_night  * tranf

 
    #train.to_csv('C:/Users/ecbey/Downloads/train_building_107.csv')  

    
    train = train.drop('timestamp', axis = 1)
    #train.to_csv('C:/Users/ecbey/Downloads/train_building_107.csv') 
   
    anomalies = anomalies.drop('timestamp', axis = 1)

    train_set_size = int(len(train) * 0.8)
    test_set_size = len(train) - train_set_size
    train_set, test_set = train_test_split(train, test_size=200, random_state=4)
    indrow = train_set.iloc[[0]]
    ind = indrow.index[0]
    i_row = 0
    train = train.sort_values(by=['time_sec'], ascending=True)
    test_set = test_set.sort_values(by=['time_sec'], ascending=True)
    train_set = train_set.sort_values(by=['time_sec'], ascending=True)
    test_with_ave =  test_set.copy()
    for index, row in test_set.iterrows(): 
        ind1 = -1
        ind2 = -1
        
        while i_row  < train.shape[0]:
            ind = train.iloc[[i_row]].index[0]

            if index <= ind:
                ind1 = train.iloc[[i_row-1]].index[0] if i_row - 1 >= 0 else train.iloc[[0]].index[0]
                ind2 = train.iloc[[i_row + 1]].index[0] if i_row + 1 < train.shape[0] else ind1
                val1 = train.loc[ind1, 'meter_reading']
                val2 = train.loc[ind2, 'meter_reading']
                #i_row = i_row +1
                val_ave = (val1 + val2) / 2
                test_with_ave.loc[index, 'meter_reading'] = val_ave
                i_row = i_row +1
                break
            else:
                i_row = i_row +1
        

    train_with_ave =  train_set.copy()
    i_row = 0
    for index, row in train_set.iterrows(): 
        ind1 = -1
        ind2 = -1
        
        while i_row  < train.shape[0]:
            ind = train.iloc[[i_row]].index[0]

            if index <= ind:
                ind1 = train.iloc[[i_row-1]].index[0] if i_row - 1 >= 0 else train.iloc[[0]].index[0]
                ind2 = train.iloc[[i_row + 1]].index[0] if i_row + 1 < train.shape[0] else ind1
                val1 = train.loc[ind1, 'meter_reading']
                val2 = train.loc[ind2, 'meter_reading']
                #i_row = i_row +1
                val_ave = (val1 + val2) / 2
                train_with_ave.loc[index, 'meter_reading_ave'] = val_ave
                i_row = i_row +1
                break
            else:
                i_row = i_row +1
        
    return train, anomalies, test_with_ave,train_with_ave, train_set, test_set
    