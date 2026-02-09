Report and code for an analysis of a simple forward rate. See "ASimpleForwardRateAnalysis.pdf" for an overview of the findings and the methods used in the code.

*****************************************************
********************** SUMMARY **********************

Constant volatility of forward rates is a key assumption underlying common benchmark pricing formulas for interest rate derivatives such as caps/floors and swaptions. We analyse the drift and volatility of a simple forward rate using ten years of federal reserve data with daily sampling. Empirically, we find a drift consistent with zero and fluctuations incompatible either with normal or log normal distributions of constant volatility, as is widely acknowledged in the literature. Using a GARCH analysis, we show the data exhibit volatility clustering. We train neural networks of various architectures to perform volatility forecasting over three month timeframes and compare to the benchmark GARCH forecast. Quantitatively, we find small improvements on the GARCH benchmark with the dataset used here, and with interesting qualitative differences in the predicted time evolution of the volatility.  


*****************************************************

The python code report_normality_test_1.py performs standard statistical tests, showing the daily changes in the forward rate are incompatible with a normal or log normal distribution, assuming constant volatility. 

The python code report_GARCH_1.py quantifies the time dependent volatility through a GARCH(1,1) analysis.

The python code report_running_GARCH_1.py calculates a daily running GARCH(1,1) after the 1000th day. This is to prevent leakage between the eventual training/validation/test datasets when doing the machine learning. It writes a file  "./data/Running_Garch_Vol.csv" of its output which is used as input for the machine learning.

The python code report_NN_1.py trains a standard neural network on the data and compares to the GARCH forecasts. It reads the files from NN_hyperparameters (if present - if not it generates them - which may take ~1-2 hours on a typical laptop using CPUs) which come from a hyper-parameter optimization search.

The python code report_LSTM_1.py trains an LSTM neural network on the data and compares to the GARCH forecasts. It reads the files from LSTM_hyperparameters (if present - if not it generates them - which may take ~10 hours on a typical laptop using CPUs) which come from a hyper-parameter optimization search.

The data folder contains the bond yield data files downloaded from the US federal reserve upon which the analysis is based.

The plots folder is where the generated plots are stored.
