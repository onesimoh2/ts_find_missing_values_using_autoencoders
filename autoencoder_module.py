import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
from torch.nn import functional as F
from utilities import DateUtils, variance
from fft_functions import fourier_extrapolation, fourierPrediction
import matplotlib.pyplot as plt

class FeatureDatasetFromDf(Dataset):


    def auto_inc(self, ipos):
        self.iPos += 1
        return ipos

    def __init__(self, *args):
        if len(args) == 7:
            df, scaler, fit_dat, columns_names, dateName, pred_columns, ser_pos, n_df = args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]
        else:
            df, scaler, fit_dat, columns_names, dateName, pred_columns, ser_pos, n_df = args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]

        x = df 
        x = x.reset_index()
        
        self.iPos = ser_pos
        self.x_train = np.array(x[columns_names].values)
        
        # axFFT = plt.axes()
        # tt = np.arange(0, len(trend))
        # axFFT.plot(tt, np.abs(trend))
        # axFFT.plot(tt, np.abs(restored_sig))
        # plt.show()  
              
        #all_columns_list = list(x.columns.values)
        #column_numbers = [all_columns_list.index(element) for element in columns_names if element in columns_names]
        column_numbers = list(range(len(pred_columns)))
        num_of_columns = len(pred_columns)

        if(fit_dat == 'true'):            
            self.x_train[:,column_numbers] = scaler.fit_transform(self.x_train[:,column_numbers].reshape(-1, num_of_columns)).reshape(-1, num_of_columns)            
        else:
            self.x_train[:,column_numbers] = scaler.transform(self.x_train[:,column_numbers].reshape(-1, num_of_columns)).reshape(-1, num_of_columns)            



        #trainn.to_csv('C:/Users/ecbey/Downloads/x_train.csv')  
        self.X_train = torch.tensor(self.x_train, dtype=torch.float32)

    def __len__(self):
        return len(self.x_train)
    
    def __getitem__(self, idx):
        return self.X_train[idx]


#definition of all the features for creating a model, training, testing and execution. 
class autoencoder(nn.Module):

    #iitialize weights
    def initialize_weights(self):
        for n in self.modules():
            if isinstance(n, nn.Linear):
                nn.init.kaiming_uniform_(n.weight)
                nn.init.constant_(n.bias, 0)

    def halving_numbers(self,m, n):
        numbers = []
        while m > n:
            numbers.append(m)
            m = int(m - (m - n) / 2)
        numbers.append(m)
        return numbers

    def __init__(self, epochs=15, batchSize=10, learningRate=1e-3, weight_decay=1e-5, layer_reduction_factor = 2, number_of_features = 29, seed=15000):
        super(autoencoder, self).__init__()

        bottleneck_node_number = int(number_of_features / layer_reduction_factor)
        numbers_in_layers = self.halving_numbers(number_of_features, bottleneck_node_number) 
        numbers_in_layers = [10,10,9,9,8,8,7,7,6,6] #0.0219
        
        result_pairs = list(zip(numbers_in_layers, numbers_in_layers[1:]))
        reverse_numbers_in_layers = list(reversed(numbers_in_layers))
        reversed_pairs = list(zip(reverse_numbers_in_layers, reverse_numbers_in_layers[1:]))

        #seed = 15000
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.epochs = epochs
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.weight_decay = weight_decay
        self.number_0f_features = number_of_features

        #defining the structure of the autoencoder, this is a general method that should fit different structure depending on the number of input nodes 
        self.ecoder_layers = nn.ModuleList()
        from_nodes = 0
        to_nodes = 0
        #for from_nodes, to_nodes in result_pairs for variational autoencoders:
        # for i, pairs in enumerate(result_pairs): 
        #     from_nodes = pairs[0]
        #     to_nodes = pairs[1]       
        #     if i == len(result_pairs) - 1: 
        #         break
        #     linear_layer = nn.Linear(from_nodes, to_nodes)
        #     self.ecoder_layers.append(linear_layer)
        # self.get_mean = nn.Linear(from_nodes, to_nodes)
        # self.get_std = nn.Linear(from_nodes, to_nodes)

        #for regular autoencoders
        for i, pairs in enumerate(result_pairs): 
            from_nodes = pairs[0]
            to_nodes = pairs[1]       
            linear_layer = nn.Linear(from_nodes, to_nodes)
            self.ecoder_layers.append(linear_layer)
        
        self.decoder_layers = nn.ModuleList()
        for from_nodes, to_nodes in reversed_pairs:            
            linear_layer = nn.Linear(from_nodes, to_nodes)
            self.decoder_layers.append(linear_layer)

            

        # self.first_encode_layer = nn.Linear(7, 6)
        # self.second_encode_layer = nn.Linear(6, 6)
        # self.get_mean = nn.Linear(6, 5)
        # self.get_std = nn.Linear(6, 5)
        # self.first_decode_layer = nn.Linear(5, 6)
        # self.second_decode_layer = nn.Linear(6, 6)
        # self.third_decode_layer = nn.Linear(6, 7)
        self.initialize_weights()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate, weight_decay=self.weight_decay)     
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.05, mode='min', verbose=True)

    #variational autoencoder encoder
    # def encoder(self, z):   
    #     for i, layer in enumerate(self.ecoder_layers):
    #         #if i == len(self.ecoder_layers) - 1:
    #         #    return self.get_mean(z), self.get_std(z)
    #         x = F.relu(layer(z))
    #         z = x
    #     return self.get_mean(z), self.get_std(z)

    def encoder(self, z):   
        for i, layer in enumerate(self.ecoder_layers):
            x = F.relu(layer(z))
            z = x
        return z

            
    def decoder(self, z):
        for i, layer in enumerate(self.decoder_layers):
            x = F.relu(layer(z))
            z = x    
        return z       


         

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self,recon_x, x, mu, logvar):
        #BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        BCE = rr = F.mse_loss(recon_x, x)
        # min_val = torch.min(recon_x)
        # max_val = torch.max(recon_x)
        # BCE = F.binary_cross_entropy(recon_x, x, reduction='none') 
        # BCE = BCE.sum(dim=1).mean() 
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD   



    def loss_function_ex(self,recon_x, x, mu, logvar):
        #BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        BCE = rr = F.mse_loss(recon_x, x)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD, BCE   
        
        #self.loss = nn.MSELoss()

    def loss_function_reduction_none(self,recon_x, x, mu, logvar):
        #BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        BCE = rr = F.mse_loss(recon_x, x)
        test_loss_list = rr = F.mse_loss(recon_x, x, reduction='none')
        numCols = test_loss_list.size(dim=1)
        last_epoch_individual_loss = []
        
        sumAll = 0.0 #calculate individual loss
        for xsqTen in test_loss_list:
            sumAll = 0.0
            for xsq in xsqTen:
                sumAll = sumAll + xsq

            indivAve = float(sumAll/numCols)
            last_epoch_individual_loss.append(indivAve)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD, last_epoch_individual_loss   
    
    #variational autoencoder forward
    # def forwardvar(self, x):
    #     mu, logvar = self.encoder(x)
    #     z = self.reparameterize(mu, logvar)
    #     return self.decoder(z), mu, logvar

    #regular autoencoder forward
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


    # def train_only_with_denoising_prediction(self, train_sample, max_training_loss_var_num):                               
    #     train_ave = []
    #     last_epoch_loss = []
    #     last_epoch_individual_loss_all = []
    #     max_training_loss = 0
    #     #criterion_no_reduced = nn.MSELoss(reduction = 'none')

    #     for epoch in range(self.epochs):
    #         self.train() 
    #         train_epc = 0.0
    #         train_num = 0.0
    #         for data in train_sample:
    #             data_without_change =  data.clone()
    #             #predict
    #             #data = data_subset[:10]
    #             #data = np.delete(data, 0, axis=1)
    #             #data = np.insert(data, 0, random_numbers, axis=1)
    #             random_numbers_for_batch = np.random.random(data.size(0))
    #             data[:,0] = torch.from_numpy(random_numbers_for_batch)
    #             recon_batch, mu, logvar =  self.forward(data)
    #             #output = self(data)
    #             # find loss

    #             loss = self.loss_function(recon_batch, data_without_change, mu, logvar)
    #             # if epoch + 1 == self.epochs:
    #             #     loss, last_epoch_individual_loss = self.loss_function_reduction_none(recon_batch, data, mu, logvar)
    #             #     last_epoch_individual_loss_all.extend(last_epoch_individual_loss)
    #             # else:
    #             #     loss = self.loss_function(recon_batch, data, mu, logvar)
                
    #             lossTrainData = loss.data.item() 
    #             train_epc = train_epc + lossTrainData
    #             train_num =  train_num + 1
    #             # perform back propagation
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()

    #         loss_train_data = loss.data.item() 
    #         #average of the losses for a given epoch
    #         epoc_t_ave = train_epc/train_num
    #         #add all losses in an array
    #         train_ave.append(float(epoc_t_ave))
            
    #         print(f'******epoch {epoch + 1}, loss: {train_ave[epoch]:.4f}')
            
    #         #supply the current loss for the scheduler
    #         min_loss_round = round(train_ave[epoch], 4)
    #         self.scheduler.step(min_loss_round)
    #         #accumulae the losses for each element
    #         train_epc = train_epc + loss_train_data
    #         train_num =  train_num + 1
             

    #         #supply the current loss for the scheduler
    #         min_loss_round = round(train_ave[epoch], 4)
    #         self.scheduler.step(min_loss_round)
       

    #     #calculate the theashold to detect anomalies
    #     mean, var, sig = variance(last_epoch_individual_loss_all)
    #     max_training_loss = mean +  (max_training_loss_var_num * sig)
    #     return max_training_loss, train_ave

    def train_only_with_denoising_prediction(self, train_sample, max_training_loss_var_num, batch_size):                               
        train_ave = []
        last_epoch_loss = []
        last_epoch_individual_loss_all = []
        max_training_loss = 0
        
        random_numbers_batchSize = np.random.random(batch_size)
        #criterion_no_reduced = nn.MSELoss(reduction = 'none')
        
        for epoch in range(self.epochs):
            self.train() 
            train_epc = 0.0
            train_num = 0.0
            for data_ave in train_sample:
                data  = data_ave[:, 0: 10]
                data_transf = data.clone()
                data_transf[:,0] = data_ave[:,10] #put the average in th# pred val
                recon_batch =  self.forward(data_transf)

                #new_tensor = torch.cat((data_without_change[:,0], data[:,0], recon_batch[:,0]), dim=0)
                # A = data_without_change[:,0].reshape(data.size(0),1)
                # B = data[:,0].reshape(data.size(0),1)
                # C = recon_batch[:,0].reshape(data.size(0),1)
                # tens = torch.cat((A,B , C), 1)
                loss = F.mse_loss(recon_batch, data)


                #loss = F.mse_loss(C, A)
                # if epoch + 1 == self.epochs:
                #     loss, last_epoch_individual_loss = self.loss_function_reduction_none(recon_batch, data, mu, logvar)
                #     last_epoch_individual_loss_all.extend(last_epoch_individual_loss)
                # else:
                #     loss = self.loss_function(recon_batch, data, mu, logvar)
                
                lossTrainData = loss.data.item() 
                train_epc = train_epc + lossTrainData
                train_num =  train_num + 1

                # perform back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            loss_train_data = loss.data.item() 
            #average of the losses for a given epoch
            epoc_t_ave = train_epc/train_num
            #add all losses in an array
            train_ave.append(float(epoc_t_ave))
            
            print(f'******epoch {epoch + 1}, loss: {train_ave[epoch]:.4f}')
            
            #supply the current loss for the scheduler
            min_loss_round = round(train_ave[epoch], 4)
            self.scheduler.step(min_loss_round)
            #accumulae the losses for each element
            train_epc = train_epc + loss_train_data
            train_num =  train_num + 1
             

            #supply the current loss for the scheduler
            min_loss_round = round(train_ave[epoch], 4)
            self.scheduler.step(min_loss_round)
       

        #calculate the theashold to detect anomalies
        # mean, var, sig = variance(last_epoch_individual_loss_all)
        # max_training_loss = mean +  (max_training_loss_var_num * sig)
        max_training_loss = 0
        train_ave = 0
        return max_training_loss, train_ave


    def train_only(self, train_sample, max_training_loss_var_num):                               
        train_ave = []
        last_epoch_loss = []
        last_epoch_individual_loss_all = []
        max_training_loss = 0
        #criterion_no_reduced = nn.MSELoss(reduction = 'none')

        for epoch in range(self.epochs):
            self.train() 
            train_epc = 0.0
            train_num = 0.0
            for data in train_sample:
                #predict
                #data = data_subset[:10]
                recon_batch =  self.forward(data)
                #output = self(data)
                # find loss
                loss = rr = F.mse_loss(recon_batch, data)
                #loss function for a variational autoencoder
                #loss = self.loss_function(recon_batch, data, mu, logvar)

                # if epoch + 1 == self.epochs:
                #     loss, last_epoch_individual_loss = self.loss_function_reduction_none(recon_batch, data, mu, logvar)
                #     last_epoch_individual_loss_all.extend(last_epoch_individual_loss)
                # else:
                #     loss = self.loss_function(recon_batch, data, mu, logvar)
                
                lossTrainData = loss.data.item() 
                train_epc = train_epc + lossTrainData
                train_num =  train_num + 1
                # perform back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                B = data[:,0].reshape(data.size(0),1)
                C = recon_batch[:,0].reshape(data.size(0),1)
                tens = torch.cat((B , C), 1)

            loss_train_data = loss.data.item() 
            #average of the losses for a given epoch
            epoc_t_ave = train_epc/train_num
            #add all losses in an array
            train_ave.append(float(epoc_t_ave))
            
            print(f'******epoch {epoch + 1}, loss: {train_ave[epoch]:.4f}')
            
            #supply the current loss for the scheduler
            min_loss_round = round(train_ave[epoch], 4)
            self.scheduler.step(min_loss_round)
            #accumulae the losses for each element
            train_epc = train_epc + loss_train_data
            train_num =  train_num + 1
             

            #supply the current loss for the scheduler
            min_loss_round = round(train_ave[epoch], 4)
            self.scheduler.step(min_loss_round)
       

        #calculate the theashold to detect anomalies
        # mean, var, sig = variance(last_epoch_individual_loss_all)
        # max_training_loss = mean +  (max_training_loss_var_num * sig)
        max_training_loss = 0
        train_ave = 0
        return max_training_loss, train_ave


    def execute_evaluate(self, feature_sample, test_sample, max_training_loss, index_df, scaler):
        self.eval()
        indx = 0
        test_epc = 0.0
        test_num = 0.0
        detected_anomalies = []
        compare_vals = []
        i = 0

        with torch.no_grad(): # Run without Autograd
            for original in feature_sample:
                
                recon_batch = self.forward(original)  # model can't use test to learn
                reconstructed_original_inverse = scaler.inverse_transform(recon_batch[0].reshape(1, 1)).reshape(-1)
                original_inverse = scaler.inverse_transform(original[0].reshape(1, 1)).reshape(-1)
                #test_loss, bce = self.loss_function_ex(recon_batch, original, mu, logvar)
                a = np.asscalar(reconstructed_original_inverse)
                #a = np.asscalar(recon_batch[0])
                b = np.asscalar(original_inverse)
                #b = test_sample.iloc[i, 0]
                i = i + 1
                new_cols = [b, a]
                compare_vals.append(new_cols)
                
                
            
        #         test_epc = test_epc + bce
        #         test_num = test_num + 1

        #         indx1 = index_df.iloc[[indx], [0]]
        #         print('test_loss=', bce, ' Indx=', indx)

        #         if bce > (1 * max_training_loss) :                    
        #             item = [bce, int(indx1['ID123'])]
        #             detected_anomalies.append(item)
        #             print('          test_loss=', test_loss, ' Indx=', indx1['ID123'])
        #             #pd.DataFrame(original)
                

        #         indx = indx + 1
    
        # print('max_training_loss=', max_training_loss )
        # test_loss = (test_epc/test_num)
        # pcent_anomalies_detected = (len(detected_anomalies) / len(feature_sample)) * 100
        # #print(f'     Validate_loss: {test_loss:.4f}')
        return compare_vals  #, pcent_anomalies_detected, test_loss

    def get_mean_std_for_2_points(self, pont1, point2):
        mean1, std1 = self.encoder(pont1)
        mean2, std2 = self.encoder(point2)
        return mean1, std1, mean2, std2

    def get_num_of_places_from_one_date_to_other(date1, date2, steap_type):
        if steap_type == 'hours':
            difsec = date2.timestamp() - date1.timestamp()
            return difsec
        else:
            return 0