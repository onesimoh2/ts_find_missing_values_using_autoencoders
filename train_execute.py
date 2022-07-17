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



def train_test(train, test) :
    MAX_TRAINING_LOSS_VAR =  3.0 #number of sigmas from the mean to consider a value is an anomaly
    LAYER_REDUCTION_FACTOR = 1.6 #how much each layer of the autoencoder decreases 
    #LAYER_REDUCTION_FACTOR = 1.2 #how much each layer of the autoencoder decreases 
    BATCH_SIZE = int(len(train)/100) 
    COLUMN_NAMES = ['meter_reading', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos']
    DATE_COLUMN_NAME = ''
        # defining the random seed
    seed = int(median(list(train['meter_reading'])))      
    np.random.seed(seed)
    scaler = MinMaxScaler()
    n_df = len(train)
    train_split = FeatureDatasetFromDf(train, scaler, 'true', COLUMN_NAMES, DATE_COLUMN_NAME, 1, n_df)
   

    
    # create the data loader for testing
    data_loader = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=False)

    number_of_features = int(train_split.X_train.size(dim=1))
    # create the model for the autoencoder

    model = autoencoder(epochs = 500, batchSize = BATCH_SIZE, number_of_features = number_of_features, layer_reduction_factor = LAYER_REDUCTION_FACTOR,  seed = seed)
    max_training_loss, train_ave = model.train_only(data_loader, MAX_TRAINING_LOSS_VAR)
    # fig1 = plt.figure()
    # ax1 = plt.axes()
    # epoch1 = []
    # i = 1
    # for item in train_ave:
    #     epoch1.append(i)
    #     i += 1
    # ax1.plot(epoch1, train_ave)
    # plt.show()
################### EXECUTE ####################################

    index_df = pd.DataFrame()
    n_df = len(test)
    anomaly_file_ds = FeatureDatasetFromDf(test, scaler, 'false', COLUMN_NAMES, DATE_COLUMN_NAME, 1, n_df)
    index_df.insert(0,'ID123',test.index)  


    # n_df = len(train)
    # #anomaly_file_ds = FeatureDatasetFromDf(train, scaler, 'true', COLUMN_NAMES, DATE_COLUMN_NAME, 1, n_df)
    # anomaly_file_ds = train_split 
    # index_df.insert(0,'ID123',train.index)  
    


    model.eval()
    detected_anomalies1, pcent_anomalies_detected1, test_loss1 = model.execute_evaluate(anomaly_file_ds, max_training_loss, index_df, scaler)
