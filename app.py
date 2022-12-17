
import pandas as pd
from datetime import date, datetime
from prepare_data import get_train_test_data, generate_train_test_data
from train_execute import train_test

#train, test = get_train_test_data()
train, test = generate_train_test_data()
train_test(train, test)

# try:
#     db = pd.read_csv('Data/train.csv') 
# except:
#     i = 1
# db_last = db.iloc[-1:]
i  = 1
#catfich_1986_2001(False)


