import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats as stats
import warnings
import keras_tuner as kt
from pandas.plotting import register_matplotlib_converters
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers
from pandas.errors import PerformanceWarning

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


####### plot the running garch and the EWMA

plt.figure(figsize=(8, 6))
plt.plot(sfrchange['observation_date'], mlsfr['volatility (%)'], linestyle='-', label = '$\sigma(t)$')
plt.plot(sfrchange['observation_date'], mlsfr['EWMA volatility (%)'], linestyle='-', label = '$\sigma(t)$ EWMA')
plt.axhline(y=mlsfr['volatility (%)'].mean(), color='r', linestyle='--', label = 'Long term $\sigma$') 
plt.legend()
plt.title(f'Log SFR running change GARCH(1,1) volatility')
plt.xlabel('Date')
plt.ylabel('Volatility $\sigma$ (%)')
#plt.ylim(-0.05, 0.16)
plt.xticks(rotation=45, ha='right') # Optional: Rotate x-axis labels to prevent overcrowding
ax = plt.gca() # get current axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) # Optional: Format the date labels concisely (especially for large datasets)
plt.tight_layout() # Adjust layout to make room for x-axis labels
plt.grid(True)
plt.savefig("./plots/GARCH_volatility_log_change_with_EWMA.jpg")
#plt.show()
plt.clf()

##### make some lag and future features we will use at the end for plotting 

lag_periods = list(range(1, n_lags + 1))
for lag in lag_periods:
	mlsfr[f'Vol Lag {lag}'] = mlsfr['volatility (%)'].shift(lag)

future_periods = list(range(1, n_future+1))
for fut in future_periods:
	mlsfr[f'Vol Fut {fut}'] = mlsfr['volatility (%)'].shift(-fut)

#######

observed_df = mlsfr.loc[:,'Vol Lag 1':f'Vol Lag {n_lags}']
observed_df = observed_df[observed_df.columns[::-1]]

observed_df['volatility (%)'] = mlsfr['volatility (%)']

future_periods = list(range(1, n_future+1))

for fut in future_periods:
	observed_df[f'Vol Fut {fut}'] = mlsfr[f'Vol Fut {fut}']


############################################################################

# Make dataframe of the volatility - will use this to create y later as well.
#X = mlsfr.loc[:, 'volatility (%)']

X = mlsfr[['volatility (%)']]
X['EWMA volatility (%)'] = mlsfr['EWMA volatility (%)']


###### SPLIT INTO TRAIN VALIDATION AND TEST DATASETS WITH  60/20/20 INITIAL PROPORTION ##############################

X_train_and_validation, X_test = train_test_split(X, test_size=0.2, shuffle=False)

X_train, X_validation = train_test_split(X_train_and_validation, test_size=0.25, shuffle=False)

#################################### BUFFER TO AVOID POSSIBLE LEAKAGE ########################################################
###### Remove some n_future and n_lags rows appropriately to create buffer between the training, validation, and test datasets. 


print('Length of X_train is:', len(X_train))

print('Length of X_validation is:', len(X_validation))

print('Length of X_test is:', len(X_test))


#################### GARCH BENCHMARK ON LOG SFR CHANGE UP TO LAST X_TRAIN INDEX and use it to make feature of predicted volatility behaviour - to be used as a benchmark ###################

first_index = X_train.index[0]
last_index = X_train.index[-1]

#print(first_index, last_index)

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



## use the Volatility column to predict the future volatility following the GARCH extrapolation in a seperate dataframe - this will act as a benchmark   ###

def futuregarchpredict(volnow,daysinfuture):
	return( np.sqrt(sigmafit**2 + (alphafit+betafit)**daysinfuture*( volnow**2 - sigmafit**2) ) )
	
X_train_supplemental = X_train.copy()
X_validation_supplemental = X_validation.copy()
X_test_supplemental = X_test.copy()

future_periods = list(range(1, n_future+1))
for fut in future_periods:
	X_train_supplemental[f'Vol GARCH Pred Fut {fut}'] = X_train_supplemental['volatility (%)'].apply( futuregarchpredict, args=(fut,) )
	
future_periods = list(range(1, n_future+1))
for fut in future_periods:
	X_validation_supplemental[f'Vol GARCH Pred Fut {fut}'] = X_validation_supplemental['volatility (%)'].apply( futuregarchpredict, args=(fut,) )

future_periods = list(range(1, n_future+1))
for fut in future_periods:
	X_test_supplemental[f'Vol GARCH Pred Fut {fut}'] = X_test_supplemental['volatility (%)'].apply( futuregarchpredict, args=(fut,) )


####### simple estimate of the long run volatility from the training set, to be used in plots below

sigmafit = X_train['volatility (%)'].mean()


###### PREPARE DATA IN FORMAT FOR FOR LSTM ##########################
###### GET X_train data  #######################

raw_data = X_train[['volatility (%)','EWMA volatility (%)']].to_numpy()

#### apply a scaler - we use RobustScaler()

#scaler = MinMaxScaler(feature_range=(0, 1))
#scaler = StandardScaler()
scaler = RobustScaler()
scaled_data = scaler.fit_transform(raw_data)


# 3. Create Sliding Windows
# We use sequence_length=63 to match your '63 time lags' requirement
# We use the same data as targets, shifted by your prediction horizon
lookback = n_lags
horizon = n_future

# Define input windows: Shape (samples, 63, 2)
# Define targets: Shape (samples, 63) - Using index [:, 0] for 1 output feature
# Note: We align targets so they represent the 63 steps FOLLOWING the input window
X_raw = scaled_data[:-horizon] 
y_raw = scaled_data[lookback:, 0] # Extract only the 1st feature for prediction


#  Manual alignment for multi-step output (63 predictions)
def create_multistep_data(data, lookback, horizon):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i : i + lookback, :])        # 63 lags, 2 features
        y.append(data[i + lookback : i + lookback + horizon, 0]) # 63 future steps, 1 feature
    return np.array(X), np.array(y)


X_train_lstm_unscaled, y_train_lstm_unscaled = create_multistep_data(raw_data, lookback, horizon)
X_train_lstm, y_train_lstm = create_multistep_data(scaled_data, lookback, horizon)

y_train_lstm_unscaled = y_train_lstm_unscaled.reshape((y_train_lstm_unscaled.shape[0], y_train_lstm_unscaled.shape[1], 1)) 
y_train_lstm = y_train_lstm.reshape((y_train_lstm.shape[0], y_train_lstm.shape[1], 1)) 

print(f"\nShape of X_train_lstm: {X_train_lstm.shape}")   # Expected: (len - 2*63 + 1, 63, 2)
print(f"Shape of y_train_lstm: {y_train_lstm.shape}")  # Expected: (len - 2*63 + 1, 63,1)




###### GET X_validation into shape #######################


raw_data = X_validation[['volatility (%)','EWMA volatility (%)']].to_numpy()

#scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = scaler.transform(raw_data)


# 3. Create Sliding Windows
# We use sequence_length=63 to match your '63 time lags' requirement
# We use the same data as targets, shifted by your prediction horizon
lookback = n_lags
horizon = n_future

# Define input windows: Shape (samples, 63, 2)
# Define targets: Shape (samples, 63) - Using index [:, 0] for 1 output feature
# Note: We align targets so they represent the 63 steps FOLLOWING the input window
X_raw = scaled_data[:-horizon] 
y_raw = scaled_data[lookback:, 0] # Extract only the 1st feature for prediction


#  Manual alignment for multi-step output (63 predictions)
def create_multistep_data(data, lookback, horizon):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i : i + lookback, :])        # 63 lags, 2 features
        y.append(data[i + lookback : i + lookback + horizon, 0]) # 63 future steps, 1 feature
    return np.array(X), np.array(y)


X_validation_lstm_unscaled, y_validation_lstm_unscaled = create_multistep_data(raw_data, lookback, horizon)
X_validation_lstm, y_validation_lstm = create_multistep_data(scaled_data, lookback, horizon)

y_validation_lstm_unscaled = y_validation_lstm_unscaled.reshape((y_validation_lstm_unscaled.shape[0], y_validation_lstm_unscaled.shape[1], 1)) 
y_validation_lstm = y_validation_lstm.reshape((y_validation_lstm.shape[0], y_validation_lstm.shape[1], 1)) 

print(f"\nShape of X_validation_lstm: {X_validation_lstm.shape}")   # Expected: (len - 2*63 + 1, 63, 2)
print(f"Shape of y_validation_lstm: {y_validation_lstm.shape}")  # Expected: (len - 2*63 + 1, 63,1)




###### GET X_test into shape #######################

raw_data = X_test[['volatility (%)','EWMA volatility (%)']].to_numpy()

#scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = scaler.transform(raw_data)


# 3. Create Sliding Windows
# We use sequence_length=63 to match your '63 time lags' requirement
# We use the same data as targets, shifted by your prediction horizon
lookback = n_lags
horizon = n_future

# Define input windows: Shape (samples, 63, 2)
# Define targets: Shape (samples, 63) - Using index [:, 0] for 1 output feature
# Note: We align targets so they represent the 63 steps FOLLOWING the input window
X_raw = scaled_data[:-horizon] 
y_raw = scaled_data[lookback:, 0] # Extract only the 1st feature for prediction


# Manual alignment for multi-step output (63 predictions)
def create_multistep_data(data, lookback, horizon):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i : i + lookback, :])        # 63 lags, 2 features
        y.append(data[i + lookback : i + lookback + horizon, 0]) # 63 future steps, 1 feature
    return np.array(X), np.array(y)


X_test_lstm_unscaled, y_test_lstm_unscaled = create_multistep_data(raw_data, lookback, horizon)
X_test_lstm, y_test_lstm = create_multistep_data(scaled_data, lookback, horizon)

y_test_lstm_unscaled = y_test_lstm_unscaled.reshape((y_test_lstm_unscaled.shape[0], y_test_lstm_unscaled.shape[1], 1)) 
y_test_lstm = y_test_lstm.reshape((y_test_lstm.shape[0], y_test_lstm.shape[1], 1)) 


print(f"\nShape of X_test_lstm: {X_test_lstm.shape}") # Expected: (len - 2*63 + 1, 63, 2)
print(f"Shape of y_test_lstm: {y_test_lstm.shape}") # Expected: (len - 2*63 + 1, 63,1)


############################# TRAIN LSTM NEURAL NETWORK ##########################################################################
#################################################################################################################################


class Seq2SeqHyperModel(kt.HyperModel):
    def build(self, hp):
        n_future, n_lags, n_features = 63, 63, 2
        
        # Hyperparameters for structural depth
        # This will test having 1, 2, or 3 LSTM layers in the Encoder
        num_encoder_layers = hp.Int('num_layers', min_value=1, max_value=3)
        hp_units = hp.Int('units', min_value=32, max_value=256, step=32)
        hp_lr = hp.Float('learning_rate', 5e-6, 5e-4, sampling='log')
        hp_dropout = hp.Float('dropout', 0.1, 0.4, step=0.1)

        # --- ENCODER ---
        encoder_inputs = layers.Input(shape=(n_lags, n_features))
        x = encoder_inputs

        # Loop to create stacked layers
        for i in range(num_encoder_layers):
            # The last layer needs return_state=True to feed the decoder bridge
            is_last_layer = (i == num_encoder_layers - 1)
            
            if is_last_layer:
                # Last layer: get sequences for attention AND states for bridge
                encoder_outputs, state_h, state_c = layers.LSTM(
                    hp_units, return_sequences=True, return_state=True, dropout=hp_dropout
                )(x)
            else:
                # Intermediate layers: just pass sequences to the next LSTM
                x = layers.LSTM(hp_units, return_sequences=True, dropout=hp_dropout)(x)

        # --- DECODER ---
        # bridge the encoder's last hidden state to the decoder's time steps
        decoder_input_bridge = layers.RepeatVector(n_future)(state_h)
        decoder_lstm = layers.LSTM(hp_units, return_sequences=True, dropout=hp_dropout)(decoder_input_bridge)

        # --- ATTENTION ---
        # decoder_lstm: (batch, 63, units) | encoder_outputs: (batch, 63, units)
        attention_layer = layers.AdditiveAttention()
        context_vector = attention_layer([decoder_lstm, encoder_outputs])
        decoder_combined_context = layers.Concatenate()([decoder_lstm, context_vector])

        # --- OUTPUT ---
        outputs = layers.TimeDistributed(layers.Dense(1))(decoder_combined_context)

        model = keras.Model(inputs=encoder_inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp_lr),
            loss='mae',
            metrics=['mse']
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        # Tune batch size alongside architecture
        return model.fit(
            *args,
            batch_size=hp.Choice('batch_size', values=[16, 32, 64, 128]),
            **kwargs,
        )

# Initialize Tuner
tuner = kt.BayesianOptimization(
    Seq2SeqHyperModel(),
    objective='val_loss',
    max_trials=50, # Increased trials to account for larger search space
    directory='LSTM_hyperparameters',
    project_name='deep_seq2seq_volatility2'
)

# Run the Search using aligned arrays
tuner.search(
    x=X_train_lstm, 
    y=y_train_lstm,
    epochs=500,
    validation_data=(X_validation_lstm, y_validation_lstm),
    callbacks=[keras.callbacks.EarlyStopping('val_loss', patience=50)]
)

#  Get the results
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"\nBest LSTM hyperparameters found:")
print(f"Best Number of Layers: {best_hps.get('num_layers')}")
print(f"Best Units: {best_hps.get('units')}")
print(f"Best Dropout: {best_hps.get('dropout')}")
print(f"Best Learning Rate: {best_hps.get('learning_rate')}")
print(f"Best Batch Size: {best_hps.get('batch_size')}")



###### use the best hypers to train on.

model = tuner.hypermodel.build(best_hps)

print("\nModel summary is:")
print(model.summary())

early_stopping = keras.callbacks.EarlyStopping(
    patience=100,
#    min_delta=0.0001,
    restore_best_weights=True,
)

# Training the final model
history = model.fit(
    X_train_lstm, y_train_lstm,
    validation_data=(X_validation_lstm, y_validation_lstm),
    epochs=500,
    callbacks=[early_stopping],
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="LSTM Mean absolute error")
plt.ylim(0, 1) 
plt.xlabel('Training epoch')
plt.ylabel('MAE loss')
plt.grid(True)
plt.savefig("./plots/LSTM_Training_and_Validation_Loss.jpg")
#plt.show()
plt.clf()


#### FINAL PREDICTION ON THE TRAINING SET
predictions_train = model.predict(X_train_lstm)

flat_predictions = predictions_train.flatten().reshape(-1, 1)
dummy_array = np.zeros((len(flat_predictions), 2))
dummy_array[:, 0] = flat_predictions[:, 0]

#### Invert the scaling
unscaled_full = scaler.inverse_transform(dummy_array)
predictions_train_unscaled = unscaled_full[:, 0].reshape(predictions_train.shape)

predictions_train_unscaled = np.squeeze(predictions_train_unscaled)
predictions_train = np.squeeze(predictions_train)

print('\npredictions_train shape is:')
print(predictions_train.shape)
print(predictions_train_unscaled.shape)


#### FINAL PREDICTION ON THE VALIDATION SET
predictions_validation = model.predict(X_validation_lstm)


flat_predictions = predictions_validation.flatten().reshape(-1, 1)
dummy_array = np.zeros((len(flat_predictions), 2))
dummy_array[:, 0] = flat_predictions[:, 0]

#### Invert the scaling
unscaled_full = scaler.inverse_transform(dummy_array)
predictions_validation_unscaled = unscaled_full[:, 0].reshape(predictions_validation.shape)

predictions_validation_unscaled = np.squeeze(predictions_validation_unscaled)
predictions_validation = np.squeeze(predictions_validation)

print('\npredictions_validation shape is:')
print(predictions_validation.shape)
print(predictions_validation_unscaled.shape)



#### PREDICT ON THE TEST SET
predictions_test = model.predict(X_test_lstm)

flat_predictions = predictions_test.flatten().reshape(-1, 1)
dummy_array = np.zeros((len(flat_predictions), 2))
dummy_array[:, 0] = flat_predictions[:, 0]

#### Invert the scaling
unscaled_full = scaler.inverse_transform(dummy_array)
predictions_test_unscaled = unscaled_full[:, 0].reshape(predictions_test.shape)

predictions_test_unscaled = np.squeeze(predictions_test_unscaled)
predictions_test = np.squeeze(predictions_test)

print('\npredictions_test_unscaled shape is:')
print(predictions_test.shape)
print(predictions_test_unscaled.shape)


y_train_lstm = np.squeeze(y_train_lstm)
y_validation_lstm = np.squeeze(y_validation_lstm)
y_test_lstm = np.squeeze(y_test_lstm)

y_train_lstm_unscaled = np.squeeze(y_train_lstm_unscaled)
y_validation_lstm_unscaled = np.squeeze(y_validation_lstm_unscaled)
y_test_lstm_unscaled = np.squeeze(y_test_lstm_unscaled)

# Evaluate the mean squared error on the test set. Also training and validation sets to get a complete picture. 
mse = mean_squared_error(y_train_lstm, predictions_train)
print(f"\nMean Squared Error on Scaled Training Data: {mse}")

mse = mean_squared_error(y_validation_lstm, predictions_validation)
print(f"\nMean Squared Error on Scaled Validation Data: {mse}")

mse = mean_squared_error(y_test_lstm, predictions_test)
print(f"\nMean Squared Error on Scaled Test Data: {mse}")

mse = mean_squared_error(y_train_lstm_unscaled, predictions_train_unscaled)
print(f"\nMean Squared Error on Training Data: {mse}")

mse = mean_squared_error(y_validation_lstm_unscaled, predictions_validation_unscaled)
print(f"\nMean Squared Error on Validation Data: {mse}")

############ CALCULATE THE MSE ON THE UNSCALED OBSERVED TEST AND TEST PREDICTION SETS
mse = mean_squared_error(y_test_lstm_unscaled, predictions_test_unscaled)
print(f"\nMean Squared Error on Test Data: {mse}")

mae = mean_absolute_error(y_test_lstm_unscaled, predictions_test_unscaled)
print(f"\nMean Absolute Error on Test Data: {mae}")


##### We want to compare to the GARCH prediction - create some dataframes allowing easy comparison ######

#### This is a little trick to find where the original index matches the volatility of the test set #######
findindex = X_test_supplemental.index[X_test_supplemental['volatility (%)'] == y_test_lstm_unscaled[0,0]][0]
findindexmin1 = findindex-1

###### Reducing the X_test_supplemental dataframe -- which contains the GARCH predictions --- to the relevant rows matching y_test_unscales ########
X_test_supplemental = X_test_supplemental.loc[findindexmin1:]
X_test_supplemental = X_test_supplemental.iloc[:-n_future]
X_test_supplemental.drop('volatility (%)', axis=1, inplace=True)
X_test_supplemental.drop('EWMA volatility (%)', axis=1, inplace=True)


###### note the column names here (Future GARCH volatility predictions) are just a placeholder - these are actually the predictions from the LSTM #### 
y_pred_unscaled = pd.DataFrame(
	predictions_test_unscaled,
	index=X_test_supplemental.index,
	columns=X_test_supplemental.columns,
)

#print(y_pred_unscaled)

###### note the column names here (Future GARCH volatility predictions) are just a placeholder - these are actually the unscaled observations from the test dataset #### 
y_test_unscaled = pd.DataFrame(
	y_test_lstm_unscaled,
	index=X_test_supplemental.index,
	columns=X_test_supplemental.columns,
)

#print(y_test_unscaled)

############ uncomment to check it matches the previous calculation above so that the new dataframes are acting as expected #############
#mse = mean_squared_error(y_test_unscaled, y_pred_unscaled)
#print(f"\nMean Squared Error on Test Data: {mse}")

#mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
#print(f"\nMean Absolute Error on Test Data: {mae}")


########## Calculate the MSE and MAE of the GARCH predictions #####################

y_garch_pred = X_test_supplemental
#print(y_garch_pred)

mse = mean_squared_error(y_test_unscaled, y_garch_pred)
print(f"\nMean Squared Error GARCH predictions on the Test: {mse}")

mae = mean_absolute_error(y_test_unscaled, y_garch_pred)
print(f"\nMean Absolute Error GARCH predictions on the Test: {mae}")

#print('\n y_test unscaled:')
#print(y_test_unscaled)

#print('\n y_garch_pred:')
#print(y_garch_pred)

########## manipulations for plotting ###################

#print(observed_df.loc[findindexmin1])

observed_df = observed_df.loc[findindexmin1:]
observed_df = observed_df.iloc[:-n_future]

observed_df = observed_df.reset_index(drop=True)
y_garch_pred = y_garch_pred.reset_index(drop=True)
y_pred_unscaled = y_pred_unscaled.reset_index(drop=True)
y_test_unscaled = y_test_unscaled.reset_index(drop=True)


day_index = np.array( list(range(-n_lags, n_future+1)) )

lastrowindex = len(y_test_unscaled) - 1
#print(lastrowindex)

##################################

rowindex = lastrowindex-360

y_test_row = np.concatenate( ( observed_df.loc[rowindex, f'Vol Lag {n_lags}':'volatility (%)'].to_numpy(), y_test_unscaled.loc[rowindex].to_numpy() ) )
y_test_check = observed_df.loc[rowindex].to_numpy()
y_garch_row = np.concatenate( ( observed_df.loc[rowindex, f'Vol Lag {n_lags}':'volatility (%)'].to_numpy(), y_garch_pred.loc[rowindex].to_numpy() ) )
y_pred_row = np.concatenate( ( observed_df.loc[rowindex, f'Vol Lag {n_lags}':'volatility (%)'].to_numpy(), y_pred_unscaled.loc[rowindex].to_numpy() ) )


plt.plot(day_index, y_test_row, '-', color='tab:blue', label='Observed')
plt.plot(day_index, y_pred_row, '--', color='tab:red', label='Extrapolated from $t = 0$  using LSTM')
plt.plot(day_index, y_garch_row, ':', color='tab:orange', label='Extrapolated from  $t = 0$ using GARCH')
plt.axhline(y=sigmafit, linestyle='--', color='tab:blue', label = 'Long term $\sigma$ from $X_{train}$') 
plt.legend()
plt.xlabel('t (days)')
plt.ylabel('Volatility $\sigma$ (%)')
plt.grid(True)
plt.savefig("./plots/LSTM_Volatility_Prediction_1.jpg")
#plt.show()
plt.clf()


##################################

rowindex = lastrowindex-240

y_test_row = np.concatenate( ( observed_df.loc[rowindex, f'Vol Lag {n_lags}':'volatility (%)'].to_numpy(), y_test_unscaled.loc[rowindex].to_numpy() ) )
y_test_check = observed_df.loc[rowindex].to_numpy()
y_garch_row = np.concatenate( ( observed_df.loc[rowindex, f'Vol Lag {n_lags}':'volatility (%)'].to_numpy(), y_garch_pred.loc[rowindex].to_numpy() ) )
y_pred_row = np.concatenate( ( observed_df.loc[rowindex, f'Vol Lag {n_lags}':'volatility (%)'].to_numpy(), y_pred_unscaled.loc[rowindex].to_numpy() ) )


plt.plot(day_index, y_test_row, '-', color='tab:blue', label='Observed')
plt.plot(day_index, y_pred_row, '--', color='tab:red', label='Extrapolated from $t = 0$  using LSTM')
plt.plot(day_index, y_garch_row, ':', color='tab:orange', label='Extrapolated from  $t = 0$ using GARCH')
plt.axhline(y=sigmafit, linestyle='--', color='tab:blue', label = 'Long term $\sigma$ from $X_{train}$') 
plt.legend()
plt.xlabel('t (days)')
plt.ylabel('Volatility $\sigma$ (%)')
plt.grid(True)
plt.savefig("./plots/LSTM_Volatility_Prediction_2.jpg")
#plt.show()
plt.clf()

##################################

rowindex = lastrowindex-120

y_test_row = np.concatenate( ( observed_df.loc[rowindex, f'Vol Lag {n_lags}':'volatility (%)'].to_numpy(), y_test_unscaled.loc[rowindex].to_numpy() ) )
y_test_check = observed_df.loc[rowindex].to_numpy()
y_garch_row = np.concatenate( ( observed_df.loc[rowindex, f'Vol Lag {n_lags}':'volatility (%)'].to_numpy(), y_garch_pred.loc[rowindex].to_numpy() ) )
y_pred_row = np.concatenate( ( observed_df.loc[rowindex, f'Vol Lag {n_lags}':'volatility (%)'].to_numpy(), y_pred_unscaled.loc[rowindex].to_numpy() ) )


plt.plot(day_index, y_test_row, '-', color='tab:blue', label='Observed')
plt.plot(day_index, y_pred_row, '--', color='tab:red', label='Extrapolated from $t = 0$  using LSTM')
plt.plot(day_index, y_garch_row, ':', color='tab:orange', label='Extrapolated from  $t = 0$ using GARCH')
plt.axhline(y=sigmafit, linestyle='--', color='tab:blue', label = 'Long term $\sigma$ from $X_{train}$') 
plt.legend()
plt.xlabel('t (days)')
plt.ylabel('Volatility $\sigma$ (%)')
plt.grid(True)
plt.savefig("./plots/LSTM_Volatility_Prediction_3.jpg")
#plt.show()
plt.clf()

##################################

rowindex = lastrowindex

y_test_row = np.concatenate( ( observed_df.loc[rowindex, f'Vol Lag {n_lags}':'volatility (%)'].to_numpy(), y_test_unscaled.loc[rowindex].to_numpy() ) )
y_test_check = observed_df.loc[rowindex].to_numpy()
y_garch_row = np.concatenate( ( observed_df.loc[rowindex, f'Vol Lag {n_lags}':'volatility (%)'].to_numpy(), y_garch_pred.loc[rowindex].to_numpy() ) )
y_pred_row = np.concatenate( ( observed_df.loc[rowindex, f'Vol Lag {n_lags}':'volatility (%)'].to_numpy(), y_pred_unscaled.loc[rowindex].to_numpy() ) )


plt.plot(day_index, y_test_row, '-', color='tab:blue', label='Observed')
plt.plot(day_index, y_pred_row, '--', color='tab:red', label='Extrapolated from $t = 0$  using LSTM')
plt.plot(day_index, y_garch_row, ':', color='tab:orange', label='Extrapolated from  $t = 0$ using GARCH')
plt.axhline(y=sigmafit, linestyle='--', color='tab:blue', label = 'Long term $\sigma$ from $X_{train}$') 
plt.legend()
plt.xlabel('t (days)')
plt.ylabel('Volatility $\sigma$ (%)')
plt.grid(True)
plt.savefig("./plots/LSTM_Volatility_Prediction_4.jpg")
#plt.show()
plt.clf()




