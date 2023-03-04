
import pandas as pd
from datetime import date, datetime
from prepare_data import get_train_test_data, generate_train_test_anomaly_data
from train_execute import train_test
from sklearn.model_selection import train_test_split
import numpy  as np
import matplotlib.pyplot as plt




#train, test = get_train_test_data()
train, anomalies, test_with_ave,train_with_ave, train_set, test_set = generate_train_test_anomaly_data()

fig1 = plt.figure()
ax1 = plt.axes()

ax1.plot(train.iloc[:, 1], train.iloc[:, 0])
plt.show()


 # Random split
# train_set_size = int(len(train) * 0.8)
# test_set_size = len(train) - train_set_size
# train_set, test_set = train_test_split(train, test_size=200, random_state=4)
# indrow = train_set.iloc[[0]]
# ind = indrow.index[0]
# i_row = 0
# train = train.sort_values(by=['time_sec'], ascending=True)
# test_set = test_set.sort_values(by=['time_sec'], ascending=True)
# train_set = train_set.sort_values(by=['time_sec'], ascending=True)
# test_with_ave =  test_set.copy()
# for index, row in test_set.iterrows(): 
#     ind1 = -1
#     ind2 = -1
    
#     while i_row  < train.shape[0]:
#         ind = train.iloc[[i_row]].index[0]

#         if index <= ind:
#             ind1 = train.iloc[[i_row-1]].index[0] if i_row - 1 >= 0 else train.iloc[[0]].index[0]
#             ind2 = train.iloc[[i_row + 1]].index[0] if i_row + 1 < train.shape[0] else ind1
#             val1 = train.loc[ind1, 'meter_reading']
#             val2 = train.loc[ind2, 'meter_reading']
#             #i_row = i_row +1
#             val_ave = (val1 + val2) / 2
#             test_with_ave.loc[index, 'meter_reading'] = val_ave
#             i_row = i_row +1
#             break
#         else:
#             i_row = i_row +1
    

# train_with_ave =  train_set.copy()
# i_row = 0
# for index, row in train_set.iterrows(): 
#     ind1 = -1
#     ind2 = -1
    
#     while i_row  < train.shape[0]:
#         ind = train.iloc[[i_row]].index[0]

#         if index <= ind:
#             ind1 = train.iloc[[i_row-1]].index[0] if i_row - 1 >= 0 else train.iloc[[0]].index[0]
#             ind2 = train.iloc[[i_row + 1]].index[0] if i_row + 1 < train.shape[0] else ind1
#             val1 = train.loc[ind1, 'meter_reading']
#             val2 = train.loc[ind2, 'meter_reading']
#             #i_row = i_row +1
#             val_ave = (val1 + val2) / 2
#             train_with_ave.loc[index, 'meter_reading_ave'] = val_ave
#             i_row = i_row +1
#             break
#         else:
#             i_row = i_row +1



err_ave = pd.concat([test_set["meter_reading"], test_with_ave["meter_reading"]], axis=1)
nparr = np.array(err_ave)
error = np.subtract(nparr[:, 0], nparr[:, 1])
mean_squared_error = np.mean(np.square(error))   

fig1 = plt.figure()
plt.plot(test_set["time_sec"], test_set["meter_reading"], label='Original')
plt.plot(test_set["time_sec"], test_with_ave["meter_reading"], label='Generated')
#ax1.plot(test_set["time_sec"], test_set["meter_reading"], test_with_ave["meter_reading"])
plt.legend()
plt.show()
train_test(train_set, test_set, test_with_ave, train_with_ave)


i  = 1



