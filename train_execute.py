from utilities import get_csv_from_blob, DateUtils
import pandas as pd
import numpy  as np
from datetime import date, datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statistics import mean, median
from sklearn.preprocessing import MinMaxScaler
from fft_functions import fourier_extrapolation
from autoencoder_module import FeatureDatasetFromDf, autoencoder
from torch.utils.data import DataLoader, Subset



def train_test(train, test, test_with_ave, train_with_ave) :
    MAX_TRAINING_LOSS_VAR =  3.0 #number of sigmas from the mean to consider a value is an anomaly
    LAYER_REDUCTION_FACTOR = 1.5 #reduce half of the input variables in the latent space 
    BATCH_SIZE = int(len(train)/100) 
    #COLUMN_NAMES = ['meter_reading', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos']
    COLUMN_NAMES = ['meter_reading','year_quarter_1','year_quarter_2','year_quarter_3','year_quarter_4','day_holliday',	'day_midnight',	'day_morning','day_afternoon', 'day_night']
    PREDICTED_COLUMNS = ['meter_reading']
    DATE_COLUMN_NAME = ''
        # defining the random seed
    seed = int(median(list(train['meter_reading'])))      
    np.random.seed(seed)
    scaler = MinMaxScaler()
    n_df = len(train)
    train_split = FeatureDatasetFromDf(train, scaler, 'true', COLUMN_NAMES, DATE_COLUMN_NAME, PREDICTED_COLUMNS, 1, n_df)
    
    columns_with_ave = COLUMN_NAMES.copy()
    columns_with_ave.append("meter_reading_ave")
    train_with_ave_tensor = FeatureDatasetFromDf(train_with_ave, scaler, 'true', columns_with_ave, DATE_COLUMN_NAME, PREDICTED_COLUMNS, 1, n_df)
    
    test_with_ave_tensor = FeatureDatasetFromDf(test_with_ave, scaler, 'true', COLUMN_NAMES, DATE_COLUMN_NAME, PREDICTED_COLUMNS, 1, n_df)
    
    # create the data loader for testing
    data_loader = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True)
    data_loader_ave = DataLoader(train_with_ave_tensor, batch_size=BATCH_SIZE, shuffle=True)
    number_of_features = int(train_split.X_train.size(dim=1))
    # create the model for the autoencoder

    model = autoencoder(epochs = 70, batchSize = BATCH_SIZE, number_of_features = number_of_features, layer_reduction_factor = LAYER_REDUCTION_FACTOR,  seed = seed)
    max_training_loss, train_ave = model.train_only(data_loader, MAX_TRAINING_LOSS_VAR)
    compare_vals = model.execute_evaluate(test_with_ave_tensor, test, max_training_loss, test.index, scaler)
    error = calc_error (compare_vals)
    model.optimizer.param_groups[0]['lr'] = 1e-3
    max_training_loss, train_ave = model.train_only_with_denoising_prediction(data_loader_ave, MAX_TRAINING_LOSS_VAR,BATCH_SIZE)
    compare_vals = model.execute_evaluate(test_with_ave_tensor, test, max_training_loss, test.index, scaler)
    error = calc_error (compare_vals)
    model.optimizer.param_groups[0]['lr'] = 1e-3
    max_training_loss, train_ave = model.train_only(data_loader, MAX_TRAINING_LOSS_VAR)
    compare_vals = model.execute_evaluate(test_with_ave_tensor, test, max_training_loss, test.index, scaler)
    error = calc_error (compare_vals)
    model.optimizer.param_groups[0]['lr'] = 1e-3
    max_training_loss, train_ave = model.train_only_with_denoising_prediction(data_loader, MAX_TRAINING_LOSS_VAR,BATCH_SIZE)
    compare_vals = model.execute_evaluate(test_with_ave_tensor, test, max_training_loss, test.index, scaler)
    error = calc_error (compare_vals)

######################## FIND MISSING POINTS ###################
    
    for index, row in train.iterrows(): 
        row_next = train.iloc[index+1,:]
        date1 = row['timestamp']
        date2 = row_next['timestamp']
######################## END MISSING POINTS ####################  
# ####################FIND ERROR MEAN ###########################
def calc_error (test_pred):
    nparr = np.array(test_pred)
    error = np.subtract(nparr[:, 0], nparr[:, 1])
    mean_squared_error = np.mean(np.square(error))
    return mean_squared_error
# ############################################################# 
