Report and code for an analysis of a simple forward rate. See "ASimpleForwardRateReport.pdf" for an overview of the findings and the methods used in the code.

The python code report_normality_test_1.py performs standard statistical tests, showing the daily changes in the forward rate are incompatible with a normal or log normal distribution, assuming constant volatility. 

The python code report_GARCH_1.py quantifies the time dependent volatility through a GARCH(1,1) analysis.

The python code report_running_GARCH_1.py calculates a daily running GARCH(1,1) after the 1000th day. This is to prevent leakage between the eventual training/validation/test datasets when doing the machine learning. It writes a file  "./data/Running_Garch_Vol.csv" of its output which is used as input for the machine learning.

The python code report_NN_1.py trains a standard neural network on the data and compares to the GARCH forecasts. It reads the files from NN_hyperparameters (if present - if not it generates them - which may take ~1-2 hours on a typical laptop using CPUs) which come from a hyper-parameter optimization search.

The python code report_LSTM_1.py trains an LSTM neural network on the data and compares to the GARCH forecasts. It reads the files from LSTM_hyperparameters (if present - if not it generates them - which may take ~10 hours on a typical laptop using CPUs) which come from a hyper-parameter optimization search.

The data folder contains the bond yield data files downloaded from the US federal reserve upon which the analysis is based.

The plots folder is where the generated plots are stored.
