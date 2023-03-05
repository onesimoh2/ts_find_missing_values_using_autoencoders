# THIS REPO IS UNDER CONSTRUCTION

## ABSTRACT

Time series missing values are a common problem in many fields such as finance, meteorology, and healthcare. Missing values occur when observations are not recorded or lost during data collection, and this can lead to biased or inaccurate analysis results. To overcome this issue, generating missing values is necessary to complete the dataset and preserve the time series properties.

However, generating missing values can be challenging, especially when dealing with time series that exhibit trend and seasonal patterns. Trend is the long-term pattern of a time series, while seasonality is the repeating pattern within a shorter time frame. These properties make it difficult to generate missing values that preserve the underlying patterns of the time series, as the generated values must fit within the overall trend and seasonal fluctuations.

Most common techniques for generating missing values, such as linear interpolation or mean imputation, do not tend to preserve the seasonal fluctuations of a time series. This can lead to biased or inaccurate analysis results, especially in applications where the seasonal component is critical, such as in forecasting or anomaly detection.

To overcome this problem, more advanced techniques such as autoencoders can be used. In this project the this approach is explored. The isea is that by training an autoencoder on an incomplete time series dataset, it can learn to capture the important features of the time series, such as trend and seasonality, and use this knowledge to generate missing values.

## DATA PREPARATION

The data was obtained from a Kaggle contest named Large-scale Energy Anomaly Detection (LEAD) https://www.kaggle.com/competitions/energy-anomaly-detection/overview.

This data is a compilation of each data point of electricity meters from approximately 400 commercial buildings. In this project the univariate time series for building number 107 was used. Following there is a graph of the data.

![image](data/original_serie.png)

The data is compiled hourly and from this the following fields were created:

- time_sec: it contains a  sequential number built from the 'timestamp' field from which several other fields will be computed. Notice that every time that there is a gap in the sequence is because there are missing values.

- year_quarter_1, year_quarter_2, year_quarter_3, year_quarter_4: These field are populated with 1 if the annotation belongs to a specific quarter, otherwise it is assigned the value 0.0001.

- day_midnight, day_morning, day_afternoon, day_night: These fields are populated with 1 if the annotation belongs to a specific category, otherwise is 0.0001.

- day_holiday: Is assigned to 1 if the timestamp of the observation is in what can be considered a holiday, otherwise is 0.0001.

After, a positional encoding factor is multiplied to each of them to provide the autoencoder with the sense of the time order.

By looking at the time series graph we can see an abrupt descend in the values. It looks like there is a part of the seties that doesen't match with the other, when applying the algorithms described here the resolts were not entirely good so the data was shoped and in the example we use just the first part of the time series, this is shown in the following graph.

![image](data/ts_truncated_first_part.png)

Also in this module the anomalies are extracted from the dataset and it is then divided in a training and a testing set. Finally, the test set is estimated by calculating the average of the two adjacent points and this is returned. 

The data preparation is done in the module prepare_data.py.

## GENERATING THE TEST SET BY OBTAINING THE MEAN OF IT ADJACENT POINTS

This is done in order to provide those calculated values to the autoencoder to finally generate the real value and also to calculate how eficient is this method to reconstruct the missing values. The following graphic just show this:

![image](data/test_generated_by_average.png)

The original value is in blue and the generated ones are in orange.

As you can see this is not so bad aproximation and when the squared average of the errors it gives the number 260.09. 







 



## REFERENCES

1- A Guide to RNN: Understanding Recurrent Neural Networks and LSTM Network. Niklas Donges. 2021. https://builtin.com/data-science/recurrent-neural-networks-and-lstm

2- Multivariate Time Series Forecasting Using LSTM, GRU & 1d CNNs. Greg Hogg. 2021. https://www.youtube.com/watch?v=kGdbPnMCdOg  

3- Attention Is All You Need. Ashish Vaswani.2 017. file:///C:/Users/ecbey/OneDrive/NY%202012/Documents/AttentionAllNeed.pdf 

4- Discrete Fourier Transform (numpy.fft) — NumPy v1.22 Manual  

5- Fourier Extrapolation in Python. Artem Tartakynov. 2015. https://gist.github.com/tartakynov/83f3cd8f44208a1856ce

6- Time-Series-Analysis-1/Anomaly Detection. Anh Nguyen https://github.com/anhnguyendepocen/Time-Series-Analysis-1 

7- torch.manual seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision. David Picard. 2021.

8- Assessing a Variational Autoencoder on MNIST using Pytorch. Mauro Camara Escudero, 2020. https://maurocamaraescudero.netlify.app/post/assessing-a-variational-autoencoder-on-mnist-using-pytorch/

9- Durk Kingma. https://github.com/dpkingma/examples/blob/master/vae/main.py

10- Deep Learning, Variational Autoencoder. Ali Ghodsi, 2017. https://www.youtube.com/watch?v=uaaqyVS9-rM





