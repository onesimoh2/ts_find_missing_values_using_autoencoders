
import pandas as pd
from datetime import date, datetime
from prepare_data import get_train_test_data, generate_train_test_anomaly_data
from train_execute import train_test
from sklearn.model_selection import train_test_split
import numpy  as np




#train, test = get_train_test_data()
train, anomalies = generate_train_test_anomaly_data()
 # Random split
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



err_ave = pd.concat([test_set["meter_reading"], test_with_ave["meter_reading"]], axis=1)
nparr = np.array(err_ave)
error = np.subtract(nparr[:, 0], nparr[:, 1])
mean_squared_error = np.mean(np.square(error))   

train_test(train_set, test_set, test_with_ave, train_with_ave)

# try:
#     db = pd.read_csv('Data/train.csv') 
# except:
#     i = 1
# db_last = db.iloc[-1:]
i  = 1
#catfich_1986_2001(False)


