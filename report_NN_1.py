import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats as stats
import warnings
from pandas.plotting import register_matplotlib_converters
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from sklearn.multioutput import RegressorChain
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pandas.errors import PerformanceWarning
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner import BayesianOptimization

plt.rcParams.update({'font.size': 12})

keras.utils.set_random_seed(1700)

###### Careful: runtimewarning and performancewarning disabled here - for cleaner output ########
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=PerformanceWarning)


####### days to lag and to look into the future

n_lags = 63
n_future = 63


#### read in the ten years of bond yields from the fed

yields1MO = pd.read_csv('./data/DGS1MO.csv')
yields3MO = pd.read_csv('./data/DGS3MO.csv')
yields6MO = pd.read_csv('./data/DGS6MO.csv')
yields1 = pd.read_csv('./data/DGS1.csv')
yields2 = pd.read_csv('./data/DGS2.csv')
yields3 = pd.read_csv('./data/DGS3.csv')
yields5 = pd.read_csv('./data/DGS5.csv')
yields7 = pd.read_csv('./data/DGS7.csv')
yields10 = pd.read_csv('./data/DGS10.csv')
yields20 = pd.read_csv('./data/DGS20.csv')
yields30 = pd.read_csv('./data/DGS30.csv')

##### make a dataframe of all the bond yields

yieldsfull = pd.concat([yields1MO['observation_date'], yields1MO['DGS1MO'], yields3MO['DGS3MO'], yields6MO['DGS6MO'], yields1['DGS1'], yields2['DGS2'], yields3['DGS3'], yields5['DGS5'], yields7['DGS7'], yields10['DGS10'], yields20['DGS20'], yields30['DGS30']], axis=1)


yieldsfullold = yieldsfull.copy()
yieldsfull = yieldsfull.dropna()

##### make a dataframe of all the zero coupon bond prices

zcbonds = pd.DataFrame()

#zcbonds['observation_date'] = yieldsfull['observation_date']
zcbonds['observation_date'] = pd.to_datetime(yieldsfull['observation_date'])
zcbonds['DGS0'] = 1
zcbonds['DGS1MO'] = np.exp(-yieldsfull['DGS1MO']/12*1/100)
zcbonds['DGS3MO'] = np.exp(-yieldsfull['DGS1MO']/4*1/100)
zcbonds['DGS6MO'] = np.exp(-yieldsfull['DGS1MO']/2*1/100)
zcbonds['DGS1'] = np.exp(-yieldsfull['DGS1']*1/100)
zcbonds['DGS2'] = np.exp(-yieldsfull['DGS2']*2*1/100)
zcbonds['DGS3'] = np.exp(-yieldsfull['DGS3']*3*1/100)
zcbonds['DGS5'] = np.exp(-yieldsfull['DGS5']*5*1/100)
zcbonds['DGS7'] = np.exp(-yieldsfull['DGS7']*7*1/100)
zcbonds['DGS10'] = np.exp(-yieldsfull['DGS10']*10*1/100)
zcbonds['DGS20'] = np.exp(-yieldsfull['DGS20']*20*1/100)
zcbonds['DGS30'] = np.exp(-yieldsfull['DGS30']*30*1/100)


#print('\nThe Zero Coupon Bond price dataframe is:')
#print(zcbonds)


###################

#### define a function which calculates the three month forward rate between 2025-9-22 and 2025-12-22
### first define a function which returns the yield curve

def yieldcurveapprox(args):
	obsdate = args.iloc[0,0]
	
	maturities = np.array([1/12, 1/4, 1/2, 1, 2, 3, 5, 7, 10, 20, 30])
	tempyields = args.iloc[0,1:]
	
	interpolating_function = interp1d(maturities, tempyields, kind='cubic')
	
	x_new = np.linspace(1/12, 30, 1000) 
	y_new = interpolating_function(x_new)

	return(interpolating_function(4.4))

### Now define a function which returns the discount bond curve

def discountcurveapprox(T1, T2, args):
	obsdate = args.iloc[0,0]
	
	maturities = np.array([0, 1/12, 1/4, 1/2, 1, 2, 3, 5, 7, 10, 20, 30])
	tempzcbs = args.iloc[0,1:]

	interpolating_function = interp1d(maturities, tempzcbs, kind='cubic')
		
	return(interpolating_function(T1), interpolating_function(T2))

###### Fix the period of the Forward Rate Agreement we are studying

Tfix1 = pd.to_datetime('2025-09-22')
Tfix2 = pd.to_datetime('2025-12-22')


#### t should be in datetime format, will come from reading in column 0 of zcbonds dataframe

def Ffix(t):
	delta = Tfix2 - Tfix1
	delta = float(delta.days/365)

	tinterval1 = Tfix1 - t
	tinterval2 = Tfix2 - t
	
	tinterval1 = float(tinterval1.days/365)
	tinterval2 = float(tinterval2.days/365)
	
	argstemp = zcbonds[zcbonds['observation_date'] == t]
	
	zcb1, zcb2 = discountcurveapprox(tinterval1, tinterval2, argstemp)
	
	simpleforwardrate = 1/delta*(zcb1/zcb2 - 1)

	return(simpleforwardrate)

##### make dataframe with simple forward rates

sfr_df = pd.DataFrame()
index1 = zcbonds.index[zcbonds['observation_date'] == Tfix1]
sfr_df['observation_date'] = zcbonds.observation_date.loc[0:index1[0]]
sfr_df['sfr'] = sfr_df['observation_date'].apply(Ffix)
sfr_df['logsfr'] = sfr_df['sfr'].apply(np.log)

sfr_df = sfr_df.reset_index(drop=True)



##### make a df with the changes of the SFR

def sfrfunc(t):
	indextemp = sfr_df.index[sfr_df['observation_date'] == t]
	sfrtemp = sfr_df.loc[indextemp[0],'sfr']
	return(sfrtemp)	

def sfrchangefunc(t):
	indextemp = sfr_df.index[sfr_df['observation_date'] == t]
	change = sfr_df.loc[indextemp[0],'sfr'] - sfr_df.loc[indextemp[0]-1,'sfr']
	return(change)	

sfrchange = pd.DataFrame()
sfrchange['observation_date'] = zcbonds.observation_date.loc[1:index1[0]]
sfrchange['sfr'] = sfrchange['observation_date'].apply(sfrfunc)
sfrchange['sfr_change'] = sfrchange['observation_date'].apply(sfrchangefunc)
sfrchange['sfr_change (bps)'] = sfrchange['sfr_change']*10**4


##### make a column in the dataframe with the changes to the log of the SFR

def logsfrchangefunc(t):
	indextemp = sfr_df.index[sfr_df['observation_date'] == t]
	change = sfr_df.loc[indextemp[0],'logsfr'] - sfr_df.loc[indextemp[0]-1,'logsfr']
	return(change)

sfrchange['logsfr_change'] = sfrchange['observation_date'].apply(logsfrchangefunc)
sfrchange['logsfr_change (%)'] = sfrchange['logsfr_change']*100

######## REMOVE POSSIBLY TROUBLESOME DATES - currently DISABLED #############################################
######## Find dates where the gap to the previous entry is greater than 1 day and remove


def daydeltafunc(t):
	indextemp = sfr_df.index[sfr_df['observation_date'] == t]
	change = sfr_df.loc[indextemp[0],'observation_date'] - sfr_df.loc[indextemp[0]-1,'observation_date']
	return(int(change.days))
	

sfrchange['daydelta'] = sfrchange['observation_date'].apply(daydeltafunc)

#print(sfrchange.head(30))

'''
condition1day = (sfrchange['daydelta'] == 1)
sfrchange1day = sfrchange[condition1day]

sfrchangeold = sfrchange.copy()
sfrchange = sfrchange1day
'''

sfrchange = sfrchange.reset_index(drop=True)

####### READ IN THE PREVIOUSLY CALCULATED RUNNING VOLATILITY 

sfrchange['GARCH_volatility_logsfr_change (%)'] = 0
sfrchange['GARCH_volatility_logsfr_change'] = 0

sfrchange['GARCH_volatility_logsfr_change (%)'] = pd.read_csv('./data/Running_Garch_Vol.csv')
sfrchange['GARCH_volatility_logsfr_change'] = sfrchange['GARCH_volatility_logsfr_change (%)']/100

#print(sfrchange)


##### make a new dataframe with some key columns

mlsfr = pd.DataFrame()

mlsfr['observation_date'] = sfrchange['observation_date']
mlsfr['daydelta'] = sfrchange['daydelta']
mlsfr['sfr (%)'] = sfrchange['sfr']
mlsfr['sfr (%)'] = sfrchange['sfr']*100
mlsfr['log sfr (%)'] = mlsfr['sfr (%)'].apply(np.log)
mlsfr['log sfr change (%)'] =  sfrchange['logsfr_change (%)']
mlsfr['volatility (%)'] = sfrchange['GARCH_volatility_logsfr_change (%)']



###### Introduce a rather smoothed EWMA

sfrchange['EWMA_sigma_0.99'] = float(0)

##### only use the eventual training data volatility to average for first estimate
datatoavg = mlsfr.iloc[0:1400]
firstavg = datatoavg['volatility (%)'].mean()

sfrchange.loc[0,'EWMA_sigma_0.99'] = firstavg

EWMAlambda = 0.99

for i in range(1,len(sfrchange)):
	sfrchange.loc[i,'EWMA_sigma_0.99'] = np.sqrt( EWMAlambda*(sfrchange.loc[i-1,'EWMA_sigma_0.99']**2)+(1-EWMAlambda)*(sfrchange.loc[i-1,'logsfr_change (%)']**2) )


mlsfr['EWMA volatility (%)'] = sfrchange['EWMA_sigma_0.99']

############

n_lags = 63
n_future = 63


lag_periods = list(range(1, n_lags + 1))
for lag in lag_periods:
	mlsfr[f'Vol Lag {lag}'] = mlsfr['volatility (%)'].shift(lag)

future_periods = list(range(1, n_future+1))
for fut in future_periods:
	mlsfr[f'Vol Fut {fut}'] = mlsfr['volatility (%)'].shift(-fut)


#print(mlsfr)


##### OPTIONAL FOR EXPERIMENTATION: drop rows around the big jump in March 2020 ##########################

'''
bigjumpindex = mlsfr.index[mlsfr['observation_date'] == '2020-03-17']
print(bigjumpindex)

print(mlsfr.loc[bigjumpindex-n_future])
print(mlsfr.loc[bigjumpindex])
print(mlsfr.loc[bigjumpindex+n_lags])

mlsfr.drop(mlsfr.index[(bigjumpindex[0]-n_future):(bigjumpindex[0]+n_lags+1)], inplace=True) 
'''

############################################################################

#### we want to predict the volatility for the next n_future days - drop nan values at the end of the data
y = mlsfr.loc[:, 'Vol Fut 1':f'Vol Fut {n_future}'].dropna()


#Make n_lags lag features - drop the nan values for the start of the data
X = mlsfr.loc[:, 'volatility (%)':f'Vol Lag {n_lags}'].dropna()

X = X.drop(columns=['EWMA volatility (%)'])

##### make daydelta column for X - experiment to see if this feature helps at all ##########
#X['daydelta'] = mlsfr['daydelta']

###### Align the target and the features as we dropped the nan values 
y, X = y.align(X, join='inner', axis=0)

###### SPLIT INTO TRAIN VALIDATION AND TEST DATASETS WITH  60/20/20 INITIAL PROPORTION ##############################

X_train_and_validation, X_test, y_train_and_validation, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

X_train, X_validation, y_train, y_validation = train_test_split(X_train_and_validation, y_train_and_validation, test_size=0.25, shuffle=False)

#################################### BUFFER TO AVOID POSSIBLE LEAKAGE ########################################################
###### Remove some n_future and n_lags rows appropriately to create buffer between the training, validation, and test datasets. 

#### remove  n_lags + n_future rows  from training dataset  
X_train = X_train.iloc[:-(n_future+n_lags)]
y_train = y_train.iloc[:-(n_future+n_lags)]

#### remove last n_future rows from validation dataset  
X_validation = X_validation.iloc[:-n_future]
y_validation = y_validation.iloc[:-n_future]

#### remove first n_lag rows from test dataset  
X_test = X_test.iloc[n_lags:]
y_test = y_test.iloc[n_lags:]

# print length of the datasets 

print('Length of X_train and y_train is:', len(X_train), len(y_train))

print('Length of X_validation and y_validation is:', len(X_validation), len(y_validation))

print('Length of X_test and y_test is:', len(X_test), len(y_test))



#################### PERFORM GARCH ON LOG SFR CHANGE UP TO LAST X_TRAIN INDEX and use it to make feature of predicted volatility behaviour - to be used as a benchmark ###################

first_index = X_train.index[0]
last_index = X_train.index[-1]


ui_array = sfrchange.loc[:(last_index),'logsfr_change (%)'].to_numpy()
vi_array = np.zeros(len(ui_array))

def viupdatefunc(omega,alpha,beta):
	vi_array[0] = omega
	vi_array[0] = omega/(1-alpha-beta)
	for i in range(1,len(vi_array)):
		vi_array[i] =  omega + alpha*ui_array[i-1]**2 + beta*vi_array[i-1]	
	return(vi_array)

def objectivefunc(q):

	omega = q[0]
	alpha = q[1]
	beta = q[2]
	
	vitemp = viupdatefunc(omega,alpha,beta)
	terms = np.log(vitemp) + ui_array**2/vitemp
	return(np.sum(terms))

q0 = [0.01,0.01,0.9]

cons = [{'type':'ineq', 'fun': lambda q: q - 0.000001},
	{'type':'ineq', 'fun': lambda q: 0.9999 - q[1] - q[2] }]

res1 = minimize(objectivefunc, q0, method='SLSQP', constraints=cons, bounds=None, tol=1e-10)



omegafit = res1.x[0]
alphafit = res1.x[1]
betafit = res1.x[2]

vi_array = viupdatefunc(omegafit,alphafit,betafit)

sigmai_array = np.sqrt(vi_array)

sigmafit = np.sqrt(omegafit/(1-alphafit-betafit))
#print(sigmafit)


## use the Volatility column to predict the future volatility following the GARCH extrapolation in a seperate dataframe - this will act as a benchmark   ###

def futuregarchpredict(volnow,daysinfuture):
	return( np.sqrt(sigmafit**2 + (alphafit+betafit)**daysinfuture*( volnow**2 - sigmafit**2) ) )
	
X_train_supplemental = X_train.copy()
X_validation_supplemental = X_validation.copy()
X_test_supplemental = X_test.copy()

future_periods = list(range(0, n_future+1))
for fut in future_periods:
	X_train_supplemental[f'Vol GARCH Pred Fut {fut}'] = X_train_supplemental['volatility (%)'].apply( futuregarchpredict, args=(fut,) )
	
future_periods = list(range(0, n_future+1))
for fut in future_periods:
	X_validation_supplemental[f'Vol GARCH Pred Fut {fut}'] = X_validation_supplemental['volatility (%)'].apply( futuregarchpredict, args=(fut,) )

future_periods = list(range(0, n_future+1))
for fut in future_periods:
	X_test_supplemental[f'Vol GARCH Pred Fut {fut}'] = X_test_supplemental['volatility (%)'].apply( futuregarchpredict, args=(fut,) )


####### simple estimate of the long run volatility from the training set, to be used in plots below

sigmafit = X_train['volatility (%)'].mean()


####### Scale the data ####################

#print(X_train)
#print(y_train)

#print(X_validation)
#print(y_validation)

#print(X_test)
#print(y_test)

X_train_unscaled = X_train
y_train_unscaled = y_train

X_validation_unscaled = X_validation
y_validation_unscaled = y_validation

X_test_unscaled = X_test
y_test_unscaled = y_test


scalerX = StandardScaler().fit(X_train)
scalery = StandardScaler().fit(y_train)

X_train = scalerX.transform(X_train)
y_train = scalery.transform(y_train)

X_validation = scalerX.transform(X_validation)
y_validation = scalery.transform(y_validation)

X_test = scalerX.transform(X_test)
y_test = scalery.transform(y_test)



############################# TRAIN NEURAL NETWORK ##########################################################################
#################################################################################################################################

# Define the model for the neural network #######


# Model-building function
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(shape=(X_train.shape[1],)))
#    model.add(layers.BatchNormalization())

    # Tune the number of layers
    for i in range(hp.Int('num_layers', min_value=1, max_value=3)):
        model.add(keras.layers.Dense(units=hp.Int('units_' + str(i),
                                     min_value=64,
                                     max_value=320,
                                     step=64),
                        activation='relu',
                        kernel_initializer='he_normal',
    			bias_initializer='zeros'))


        # Tune whether to include Batch Normalization after the dense layer (and before activation is common practice)
        if hp.Boolean(f'batch_norm_{i}'):
            model.add(layers.BatchNormalization())
            
        # Tune dropout rate (optional, but often used with or instead of BN)
        if hp.Boolean(f'dropout_{i}'):
            model.add(layers.Dropout(hp.Float(f'dropout_rate_{i}', min_value=0.0, max_value=0.5, step=0.1)))
    
    model.add(keras.layers.Dense(n_future))

    # Tune the learning rate
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='log')
    
    batch_size = hp.Int('batch_size', min_value=8, max_value=32, step=8)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mae',
        metrics=['mse'],
    )
    
    return model

########

early_stopping = keras.callbacks.EarlyStopping(
    patience=100,
    min_delta=0.0001,
    restore_best_weights=True,
)

tuner = BayesianOptimization(
    build_model,
    objective='val_loss', # Monitor the validation loss
    max_trials=150,            # Total number of configurations to try
    executions_per_trial=1,   # Number of times to train each configuration (with different initial weights)
    directory='NN_hyperparameters',       # Directory to store results
    project_name='keras_hyp_tune_10'
)

#hp_batch_size = hp.Choice('batch_size', values=[8, 16, 32]) 

#### search
tuner.search(x=X_train,
             y=y_train,
             epochs=500,
    	     callbacks=[early_stopping],             
             validation_data=(X_validation, y_validation)
             ) # Specify batch sizes here) # Explicitly provide your validation set



# Fetching the best hyperparameters
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

print('\nThe Best Hyperparameters found are:')
print(tuner.results_summary(num_trials=1))

# Building the model with the best hyperparameters
model = tuner.hypermodel.build(best_hp)

print("\nModel summary is:")
print(model.summary())

# Training the final model
history = model.fit(
    X_train, y_train,
    validation_data=(X_validation, y_validation),
    epochs=500,
    callbacks=[early_stopping],
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="NN Mean absolute error")
plt.ylim(0, 1)
plt.xlabel('Training epoch')
plt.ylabel('MAE loss')
plt.grid(True)
plt.savefig("./plots/NN_Training_and_Validation_Loss.jpg")
#plt.show()
plt.clf()


#### FINAL PREDICTION ON THE TRAINING SET
predictions_train = model.predict(X_train)

y_train_pred = pd.DataFrame(
	predictions_train,
	index=y_train_unscaled.index,
	columns=y_train_unscaled.columns,
).clip(0.0)

#### Invert the scaling

predictions_train_unscaled = scalery.inverse_transform(predictions_train)

y_train_pred_unscaled = pd.DataFrame(
	predictions_train_unscaled,
	index=y_train_unscaled.index,
	columns=y_train_unscaled.columns,
).clip(0.0)

#### FINAL PREDICTION ON THE VALIDATION SET
predictions_validation = model.predict(X_validation)

y_validation_pred = pd.DataFrame(
	predictions_validation,
	index=y_validation_unscaled.index,
	columns=y_validation_unscaled.columns,
).clip(0.0)

#### Invert the scaling

predictions_validation_unscaled = scalery.inverse_transform(predictions_validation)

y_validation_pred_unscaled = pd.DataFrame(
	predictions_validation_unscaled,
	index=y_validation_unscaled.index,
	columns=y_validation_unscaled.columns,
).clip(0.0)

#### PREDICT ON THE TEST SET
predictions = model.predict(X_test)

y_pred = pd.DataFrame(
	predictions,
	index=y_test_unscaled.index,
	columns=y_test_unscaled.columns,
).clip(0.0)

#### Invert the scaling

predictions_unscaled = scalery.inverse_transform(predictions)

y_pred_unscaled = pd.DataFrame(
	predictions_unscaled,
	index=y_test_unscaled.index,
	columns=y_test_unscaled.columns,
).clip(0.0)

# Evaluate the mean squared error on the test set. Also training and validation sets to get a complete picture. 


############ CALCULATE THE MSE ON THE UNSCALED OBSERVED TEST AND TEST PREDICTION SETS
mse = mean_squared_error(y_test_unscaled, y_pred_unscaled)
print(f"\nMean Squared Error on Test Data: {mse}")

mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
print(f"\nMean Absolute Error on Test Data: {mae}")


##### COMPARE TO MSE ON THE GARCH BENCHMARK ######
y_garch_pred = X_test_supplemental.loc[:,'Vol GARCH Pred Fut 1':f'Vol GARCH Pred Fut {n_future}']

mse = mean_squared_error(y_test_unscaled, y_garch_pred)
print(f"\nMean Squared Error GARCH predictions on the Test: {mse}")

mae = mean_absolute_error(y_test_unscaled, y_garch_pred)
print(f"\nMean Absolute Error GARCH predictions on the Test: {mae}")

#### check we get the same MSE and MAE
y_garch_pred_short = y_garch_pred.iloc[37:]
y_test_unscaled_short = y_test_unscaled.iloc[37:]

mse = mean_squared_error(y_test_unscaled_short, y_garch_pred_short)
print(f"\nMean Squared Error GARCH predictions on the LSTM Test Set: {mse}")

mae = mean_absolute_error(y_test_unscaled_short, y_garch_pred_short)
print(f"\nMean Absolute Error GARCH predictions on the LSTM Test Set: {mae}")


#print('\n y_test unscaled_short:')
#print(y_test_unscaled_short)

#print('\n y_garch_pred_short:')
#print(y_garch_pred_short)


########## manipulations for plotting

X_test_supplemental = X_test_supplemental.reset_index(drop=True)

X_test = X_test_unscaled.reset_index(drop=True)
y_pred = y_pred_unscaled.reset_index(drop=True)
y_test = y_test_unscaled.reset_index(drop=True)

#print(X_test)
#print(y_pred)
#print(y_test)

lastrowindex = len(y_test) - 1

day_index = np.array( list(range(-n_lags, n_future+1)) )

#print(X_test.columns)
#print(y_test.columns)


###############################

rowindex = lastrowindex-360

y_test_row = np.concatenate( ( X_test.loc[rowindex, 'volatility (%)':f'Vol Lag {n_lags}'].to_numpy()[::-1], y_test.loc[rowindex].to_numpy() ) )
y_garch_row = np.concatenate( ( X_test.loc[rowindex, 'volatility (%)':f'Vol Lag {n_lags}'].to_numpy()[::-1], X_test_supplemental.loc[rowindex, 'Vol GARCH Pred Fut 1':f'Vol GARCH Pred Fut {n_future}'].to_numpy() ) )
y_pred_row = np.concatenate( ( X_test.loc[rowindex, 'volatility (%)':f'Vol Lag {n_lags}'].to_numpy()[::-1], y_pred.loc[rowindex].to_numpy() ) )

plt.plot(day_index, y_test_row, '-', color='tab:blue', label='Observed')
plt.plot(day_index, y_pred_row, '--', color='tab:red', label='Extrapolated from $t = 0$  using NN')
plt.plot(day_index, y_garch_row, ':', color='tab:orange', label='Extrapolated from  $t = 0$ using GARCH')
plt.axhline(y=sigmafit, linestyle='--', color='tab:blue', label = 'Long term $\sigma$ from $X_{train}$') 
plt.legend()
plt.xlabel('t (days)')
plt.ylabel('Volatility $\sigma$ (%)')
plt.grid(True)
plt.savefig("./plots/NN_Volatility_Prediction_1.jpg")
#plt.show()
plt.clf()

###############################

rowindex = lastrowindex-240

y_test_row = np.concatenate( ( X_test.loc[rowindex, 'volatility (%)':f'Vol Lag {n_lags}'].to_numpy()[::-1], y_test.loc[rowindex].to_numpy() ) )
y_garch_row = np.concatenate( ( X_test.loc[rowindex, 'volatility (%)':f'Vol Lag {n_lags}'].to_numpy()[::-1], X_test_supplemental.loc[rowindex, 'Vol GARCH Pred Fut 1':f'Vol GARCH Pred Fut {n_future}'].to_numpy() ) )
y_pred_row = np.concatenate( ( X_test.loc[rowindex, 'volatility (%)':f'Vol Lag {n_lags}'].to_numpy()[::-1], y_pred.loc[rowindex].to_numpy() ) )

plt.plot(day_index, y_test_row, '-', color='tab:blue', label='Observed')
plt.plot(day_index, y_pred_row, '--', color='tab:red', label='Extrapolated from $t = 0$  using NN')
plt.plot(day_index, y_garch_row, ':', color='tab:orange', label='Extrapolated from  $t = 0$ using GARCH')
plt.axhline(y=sigmafit, linestyle='--', color='tab:blue', label = 'Long term $\sigma$ from $X_{train}$') 
plt.legend()
plt.xlabel('t (days)')
plt.ylabel('Volatility $\sigma$ (%)')
plt.grid(True)
plt.savefig("./plots/NN_Volatility_Prediction_2.jpg")
#plt.show()
plt.clf()

###############################

rowindex = lastrowindex-120

y_test_row = np.concatenate( ( X_test.loc[rowindex, 'volatility (%)':f'Vol Lag {n_lags}'].to_numpy()[::-1], y_test.loc[rowindex].to_numpy() ) )
y_garch_row = np.concatenate( ( X_test.loc[rowindex, 'volatility (%)':f'Vol Lag {n_lags}'].to_numpy()[::-1], X_test_supplemental.loc[rowindex, 'Vol GARCH Pred Fut 1':f'Vol GARCH Pred Fut {n_future}'].to_numpy() ) )
y_pred_row = np.concatenate( ( X_test.loc[rowindex, 'volatility (%)':f'Vol Lag {n_lags}'].to_numpy()[::-1], y_pred.loc[rowindex].to_numpy() ) )

plt.plot(day_index, y_test_row, '-', color='tab:blue', label='Observed')
plt.plot(day_index, y_pred_row, '--', color='tab:red', label='Extrapolated from $t = 0$  using NN')
plt.plot(day_index, y_garch_row, ':', color='tab:orange', label='Extrapolated from  $t = 0$ using GARCH')
plt.axhline(y=sigmafit, linestyle='--', color='tab:blue', label = 'Long term $\sigma$ from $X_{train}$') 
plt.legend()
plt.xlabel('t (days)')
plt.ylabel('Volatility $\sigma$ (%)')
plt.grid(True)
plt.savefig("./plots/NN_Volatility_Prediction_3.jpg")
#plt.show()
plt.clf()


###############################

rowindex = lastrowindex

y_test_row = np.concatenate( ( X_test.loc[rowindex, 'volatility (%)':f'Vol Lag {n_lags}'].to_numpy()[::-1], y_test.loc[rowindex].to_numpy() ) )
y_garch_row = np.concatenate( ( X_test.loc[rowindex, 'volatility (%)':f'Vol Lag {n_lags}'].to_numpy()[::-1], X_test_supplemental.loc[rowindex, 'Vol GARCH Pred Fut 1':f'Vol GARCH Pred Fut {n_future}'].to_numpy() ) )
y_pred_row = np.concatenate( ( X_test.loc[rowindex, 'volatility (%)':f'Vol Lag {n_lags}'].to_numpy()[::-1], y_pred.loc[rowindex].to_numpy() ) )

plt.plot(day_index, y_test_row, '-', color='tab:blue', label='Observed')
plt.plot(day_index, y_pred_row, '--', color='tab:red', label='Extrapolated from $t = 0$  using NN')
plt.plot(day_index, y_garch_row, ':', color='tab:orange', label='Extrapolated from  $t = 0$ using GARCH')
plt.axhline(y=sigmafit, linestyle='--', color='tab:blue', label = 'Long term $\sigma$ from $X_{train}$') 
plt.legend()
plt.xlabel('t (days)')
plt.ylabel('Volatility $\sigma$ (%)')
plt.grid(True)
plt.savefig("./plots/NN_Volatility_Prediction_4.jpg")
#plt.show()
plt.clf()




