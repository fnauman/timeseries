# timeseries

[![Binder][binder-badge]][binder-url]
[![Google Colab][colab-badge]][colab-url]

[binder-badge]: https://mybinder.org/badge.svg
[binder-url]: https://mybinder.org/v2/gh/fnauman/timeseries/master
[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
[colab-url]: https://colab.research.google.com/github/fnauman/timeseries/blob/master/

(Click the *Binder* or *Colab* links to open the notebooks and work with them in the cloud.)

Time series modeling using:
 - **Machine Learning** (XGBoost, Lasso, Random Forests): [xgboost_pipeline_candy.ipynb](candydata/xgboost_pipeline_candy.ipynb) does univariate forecasting for time series data. Hyperparameter optimization is done using the scikit-learn GridSearchCV funtion. Conclusion: Lasso does better!
 - **Deep Learning** (TensorFlow, Keras): 
   - [keras_tuner_candy.ipynb](candydata/keras_tuner_candy.ipynb): Hyperparameter optimization using [keras-tuner](https://github.com/keras-team/keras-tuner).
   - [tf2_multivariate_rnn_cnn.ipynb](tf2_multivariate_rnn_cnn.ipynb) 
 This is the multivariate generalization of the univariate notebook on time series:
[Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow](https://github.com/ageron/handson-ml2/blob/master/15_processing_sequences_using_rnns_and_cnns.ipynb)
 - **Econometrics** approach (SARIMA - Seasonal Autoregressive Integrated Moving Average): [sarima_candy.ipynb](candydata/sarima_candy.ipynb). Candy data that can be downloaded from the datacamp course [here](https://www.datacamp.com/courses/forecasting-using-arima-models-in-python). 
 - **FFT**: [fourier_ts.ipynb](fourier_ts.ipynb) contains FFT extrapolation + filtering for time series prediction with synthetic periodic data.
 - **Dynamical Systems** 
   - Reconstruction of dynamical systems using delay coordinate embeddings: For many chaotic dynamical systems, one can only observe one variable. Using delay coordinate embedding (embedding into a higher dimensional space), one can reconstruct a topologically equivalent system to the original one: [Python](delayembedding/python_delayembeddings_lorenz.ipynb), [Julia](delayembedding/julia_delayembeddings_lorenz.ipynb). **NOTE**: Julia notebooks are currently not supported in google colab. Use [juliabox](https://www.juliabox.com/) to do cloud computing using Julia for free.
   - SINDy - Sparse Identification of Nonlinear Dynamics: [sindy_cubicmodel.ipynb](dynamicalsystems/sindy_cubicmodel.ipynb) Based on the [Paper](https://www.pnas.org/content/113/15/3932). The python code is much simpler (as opposed to the MATLAB code that comes with the paper) because of [scikit-learn](https://github.com/scikit-learn/scikit-learn). SINDy can be used both to discover dynamical system equations and forecasting. See also: [**Blog**](https://fnauman.github.io/sindy-dynamical-systems/)

## TODO
 - Metrics: MSE/MAE, AIC/BIC (ARIMA), QQ plots, error distributions, ...
 - NN: Transformers, Attention, Seq2Seq models.
 - Different cross validation strategies: One train/test split vs the progressively bigger training dataset used with [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

**Acknowledgements**
This repo borrows heavily from multiple sources, please refer to the notebooks.
