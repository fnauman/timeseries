# timeseries

Time series modeling using:
 - **Deep Learning** (TensorFlow, Keras): [multivariate_tf2_keras_cnn_rnn.ipynb](multivariate_tf2_keras_cnn_rnn.ipynb) 
 This is the multivariate generalization of the univariate notebook on time seriesfrom the 2nd edition:
[Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow](https://github.com/ageron/handson-ml2/blob/master/15_processing_sequences_using_rnns_and_cnns.ipynb)
 - **Econometrics/Financial** approach (SARIMA - Seasonal Autoregressive Integrated Moving Average): [sarima_candy.ipynb](sarima_candy.ipynb). Candy data that can be downloaded from the datacamp course [here](https://www.datacamp.com/courses/forecasting-using-arima-models-in-python), which is a great course. 
 - **FFT**: [fourier_ts.ipynb](fourier_ts.ipynb) contains FFT extrapolation + filtering for time series prediction with synthetic data.
 - **Dynamical Systems** (SINDy - Sparse Identification of Nonlinear Dynamics): [sindy_cubicmodel.ipynb](sindy_cubicmodel.ipynb) Based on the [Paper](https://www.pnas.org/content/113/15/3932). The python code is much simpler (as opposed to the MATLAB code that comes with the paper) because of [scikit-learn](https://github.com/scikit-learn/scikit-learn).

## TODO
 - Use the same time series and model it using RNNs, CNNs, SARIMA, FFT, SINDy so that the comparison is more direct.
 - Add LASSO, Random Forests, xgboost.
 - Apply models to more datasets.

**Acknowledgements**
This repo borrows heavily from multiple sources that are listed below:
 - [SINDy](https://www.youtube.com/watch?v=gSCa78TIldg&t=1114s): Video tutorial on SINDy. See the paper reference above.
 - [multivariate_tf2_keras_cnn_rnn.ipynb](multivariate_tf2_keras_cnn_rnn.ipynb) borrows heavily from [Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow](https://github.com/ageron/handson-ml2/blob/master/15_processing_sequences_using_rnns_and_cnns.ipynb). See the [LICENCE](https://github.com/ageron/handson-ml2/blob/master/LICENSE).
 - [mlcourse.ai](https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python)
 - [Datacamp course by James Fulton](https://www.datacamp.com/courses/forecasting-using-arima-models-in-python)
 - [Fourier extrapolation article](https://www.kdnuggets.com/2016/11/combining-different-methods-create-advanced-time-series-prediction.html)
 
