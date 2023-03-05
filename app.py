
import pandas as pd
from datetime import date, datetime
from prepare_data import  generate_train_test_anomaly_data
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



