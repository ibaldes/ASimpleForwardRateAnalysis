import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats as stats
import statsmodels.api as sm
import warnings
from pandas.plotting import register_matplotlib_converters
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from statsmodels.graphics.tsaplots import acf, pacf, plot_acf, plot_pacf
from arch import arch_model

plt.rcParams.update({'font.size': 12})

###### Careful: runtimewarning disabled here - seems to be harmless issue in viupdatefunc and objectivefunc ########
warnings.filterwarnings("ignore", category=RuntimeWarning)


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


print('\nThe Zero Coupon Bond price dataframe is:')
print(zcbonds)


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



#### REMOVE TROUBLESOME DATES ##############################################
######## Find dates where the calendar gap to the previous entry is greater than 1 day


def daydeltafunc(t):
	indextemp = sfr_df.index[sfr_df['observation_date'] == t]
	change = sfr_df.loc[indextemp[0],'observation_date'] - sfr_df.loc[indextemp[0]-1,'observation_date']
	return(int(change.days))
	

sfrchange['daydelta'] = sfrchange['observation_date'].apply(daydeltafunc)

##### Only keep the dates with a calendar gap equal to one. But also save the copy of the original dataframe with all dates as sfrchangeold


sfrchangeold = sfrchange.copy()


condition1day = (sfrchange['daydelta'] == 1)
sfrchange1day = sfrchange[condition1day]
sfrchange = sfrchange1day
sfrchange = sfrchange.reset_index(drop=True)

print('\nThe dataframe of simple forward rate changes, keeping only data from consecutive trading days (daydelta == 1), is:') 
print(sfrchange)


############################### GARCH(1,1) ANALYSIS #############################
############################# DO ANALYIS WITH SFR CHANGE (BPS) ##################


ui_array = sfrchange['sfr_change (bps)'].to_numpy()
vi_array = np.zeros(len(ui_array))


def viupdatefunc(omega,alpha,beta):
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

#### initial guess ####
q0 = [0.01,0.01,0.9]

print(objectivefunc(q0))

##### constraints ########
cons = [{'type':'ineq', 'fun': lambda q: q - 0.000001},
	{'type':'ineq', 'fun': lambda q: 0.9999 - q[1] - q[2] }]

res1 = minimize(objectivefunc, q0, method='SLSQP', constraints=cons, bounds=None, tol=1e-10)


omegafit = res1.x[0]
alphafit = res1.x[1]
betafit = res1.x[2]

vi_array = viupdatefunc(omegafit,alphafit,betafit)

# As a check: we can recover the EWMA result for lambda = 0.94 by uncommenting the below.
# vi_array = viupdatefunc(0,0.06,0.94)

#print(vi_array)

print('\nThe fitted values for the GARCH(1,1) analysis of Delta F are:')
print(f'omega = {omegafit}')
print(f'alpha = {alphafit}')
print(f'beta = {betafit}')
print(f'alpha + beta = {alphafit+betafit}')

sigmai_array = np.sqrt(vi_array)

sigmafit = np.sqrt(omegafit/(1-alphafit-betafit))

print(f'The long term average volatility is: {sigmafit} (bps)')

sfrchange['GARCH_volatility_sfr_change'] = sigmai_array



######################### USE ARCH LIBRARY AS A CHECK ########

returns = sfrchange['sfr_change (bps)']

model = arch_model(returns, vol='Garch', p=1, q=1, mean='Zero')
res = model.fit(disp='off')

#print('\nThe summary of the GARCH(1,1) fit using the arch library is:')
#print(res.summary())

conditional_vol = res.conditional_volatility

plt.figure(figsize=(12, 6))
plt.plot(sfrchange['observation_date'], returns, color='grey', alpha=0.5, label='Returns')
#plt.plot(conditional_vol.index, conditional_vol, color='red', label='GARCH Conditional Volatility (%)')
plt.plot(sfrchange['observation_date'], conditional_vol, color='red', label='$\sigma(t)$ (bps) arch package')
plt.plot(sfrchange['observation_date'], sfrchange['GARCH_volatility_sfr_change'], color='blue', linestyle='-', label = '$\sigma(t)$ (bps) own code')
plt.axhline(y=sigmafit, color='r', linestyle='--', label = 'Long term $\sigma$ (bps)') 
plt.title('GARCH(1,1) Conditional Volatility for $\Delta F(t,T_0,T_1)$ (bps)')
plt.legend()
plt.savefig("./plots/GARCH_volatility_sfr_change_using_library.jpg")
#plt.show()

plt.close('all')

################## Partial Autocorrelation Function Analysis (PACF) ################################

'''
Note: A time series with small autocorrelation but large autocorrelation in its squared values exhibits
strong non-linear dependencies, specifically a characteristic known as volatility clustering. This is a common feature of financial data
'''

#pacf_values = pacf(ui_array, nlags=10)
#print('autorcorrelations values of ui are:', pacf_values)

fig, ax = plt.subplots(figsize=(10, 5))
plot_pacf(ui_array, lags=10, ax=ax) # 'lags' parameter specifies the number of lags to display
plt.title('Partial Autocorrelation Function (PACF) Plot for $\Delta F(t,T_0,T_1) $')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation Coefficient')
plt.savefig("./plots/Partial_Autocorrelations_sfr_change_ui.jpg")
#plt.show()
plt.close('all')

#pacf_values = pacf(ui_array**2, nlags=10)
#print('autorcorrelations values of ui^2 are:', pacf_values)

fig, ax = plt.subplots(figsize=(10, 5))
plot_pacf(ui_array**2, lags=10, ax=ax) # 'lags' parameter specifies the number of lags to display
plt.title('Partial Autocorrelation Function (PACF) Plot for $(\Delta F(t,T_0,T_1) )^2$')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation Coefficient')
plt.savefig("./plots/Partial_Autocorrelations_sfr_change_ui2.jpg")
#plt.show()
plt.close('all')

#pacf_values = pacf(ui_array**2/sigmai_array**2, nlags=10)
#print('partial autorcorrelations values of ui**2/sigmai**2:', pacf_values)

fig, ax = plt.subplots(figsize=(10, 5))
plot_pacf(ui_array**2/sigmai_array**2, lags=10, ax=ax) # 'lags' parameter specifies the number of lags to display
plt.title('Partial Autocorrelation Function (PACF) Plot for $(\Delta F(t,T_0,T_1) )^2/\sigma_i^2$')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation Coefficient')
plt.savefig("./plots/Partial_Autocorrelations_sfr_change_ui2onsigmai2.jpg")
#plt.show()
plt.close('all')

##### DO GARCH(1,1) ANALYSIS ON SHORTER DATASETS AND THEN PROJECT VOLATILITY INTO THE FUTURE ######
##### First use half the dataset -- for clarity we redefine the viupdatefunc and objectivefunce even though they are the same as above ######

ui_array_full = sfrchange['sfr_change (bps)'].to_numpy()

ui_array_short = ui_array_full[0:-951]

ui_array = ui_array_short
vi_array = np.zeros(len(ui_array))

def viupdatefunc(omega,alpha,beta):
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

sigmai_array_short = np.sqrt(vi_array)

sigmafit_short = np.sqrt(omegafit/(1-alphafit-betafit))


#### extrapolation step

sigmai_array_full = np.concatenate( (sigmai_array_short, np.zeros( len(ui_array_full)-len(ui_array_short) ) ) )

for i in range(len(ui_array_short),len(ui_array_full)):
	sigmai_array_full[i] = np.sqrt( omegafit/(1-alphafit-betafit) + (alphafit+betafit)**(i+1-len(ui_array_short))*(sigmai_array_full[len(ui_array_short)-1]**2 - omegafit/(1-alphafit-betafit)) )

sfrchange['GARCH_volatility_sfr_change_prediction'] = sigmai_array_full

############# ESTIMATE VOLATILITY TERM STRUCTURE - save values as y for future plotting #######################

tempindex = 952

aparameter = -np.log( alphafit + betafit )

Vlong = sigmafit_short**2
Vzero = sfrchange.loc[tempindex, 'GARCH_volatility_sfr_change_prediction']**2

def sigmaterm(T):
	sigmaestimate2 = 252*(Vlong + (1 - np.exp(-aparameter*T) )/(aparameter*T)*(Vzero - Vlong))
	return(np.sqrt(sigmaestimate2))

x = np.linspace(0.0001, 2, 1000)
y = sigmaterm(365*x)



############## NOW USING FIRST 3/4 of DATA
##### for clarity we redefine the viupdatefunc and objectivefunce even though they are the same as above ######

ui_array_full = sfrchange['sfr_change (bps)'].to_numpy()

ui_array_short = ui_array_full[0:-476]


ui_array = ui_array_short
vi_array = np.zeros(len(ui_array))

def viupdatefunc(omega,alpha,beta):
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

sigmai_array_short = np.sqrt(vi_array)

sigmafit_short_old = sigmafit_short
sigmafit_short = np.sqrt(omegafit/(1-alphafit-betafit))


#### extrapolation step

sigmai_array_full = np.concatenate( (sigmai_array_short, np.zeros( len(ui_array_full)-len(ui_array_short) ) ) )

for i in range(len(ui_array_short),len(ui_array_full)):
	sigmai_array_full[i] = np.sqrt( omegafit/(1-alphafit-betafit) + (alphafit+betafit)**(i+1-len(ui_array_short))*(sigmai_array_full[len(ui_array_short)-1]**2 - omegafit/(1-alphafit-betafit)) )

sfrchange['GARCH_volatility_sfr_change_prediction_2'] = sigmai_array_full

##### Plot of GARCH(1,1) for the change in the SFR (bps), together with forecast predictions using 1/2 and 3/4 of the dataset.

plt.figure(figsize=(8, 6))
plt.plot(sfrchange['observation_date'], sfrchange['GARCH_volatility_sfr_change'], linestyle='-', label = '$\sigma(t)$ (bps)')
plt.plot(sfrchange.loc[(len(sfrchange)-951):,'observation_date'], sfrchange.loc[(len(sfrchange)-951):,'GARCH_volatility_sfr_change_prediction'], linestyle='-', linewidth=2, color = 'green', label = '$\sigma(t)$ (bps)  extrapolated from first 1/2 of data')
plt.plot(sfrchange.loc[(len(sfrchange)-476):,'observation_date'], sfrchange.loc[(len(sfrchange)-476):,'GARCH_volatility_sfr_change_prediction_2'], linestyle='-', linewidth=2, color = 'red', label = '$\sigma(t)$ (bps)  extrapolated from first 3/4 of data')
plt.axhline(y=sigmafit, color='tab:blue', linestyle='--', label = 'Long term $\sigma$ (bps)') 
plt.axhline(y=sigmafit_short_old, color='g', linestyle='--', label = 'Long term $\sigma$ (bps) first 1/2 of data') 
plt.legend()
plt.title(f'SFR change GARCH(1,1) volatility')
plt.xlabel('Date')
plt.ylabel('Volatility $\sigma$ (bps)')
plt.ylim(-0.5, 20)
plt.xticks(rotation=45, ha='right') # Optional: Rotate x-axis labels to prevent overcrowding
ax = plt.gca() # get current axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) # Optional: Format the date labels concisely (especially for large datasets)
plt.tight_layout() # Adjust layout to make room for x-axis labels
plt.grid(True)
plt.savefig("./plots/GARCH_volatility_sfr_change_with_prediction.jpg")
#plt.show()
plt.clf()


##################### ESTIMATE VOLATILITY TERM STRUCTURE FOR THE 3/4 DATASET AND PLOT  ############


tempindex = 1427

aparameter = -np.log( alphafit + betafit )
Vlong = sigmafit_short**2
Vzero = sfrchange.loc[tempindex, 'GARCH_volatility_sfr_change_prediction_2']**2

def sigmaterm(T):
	sigmaestimate2 = 252*(Vlong + ( 1 - np.exp(-aparameter*T) )/(aparameter*T)*(Vzero - Vlong))
	return(np.sqrt(sigmaestimate2))

x = np.linspace(0.0001, 2, 1000)
ynew = sigmaterm(365*x)

plt.plot(x, y, label='$\overline{\sigma} (T-t)$ for t = 2020-11-15', color = 'tab:blue') 
plt.plot(x, ynew, label='$\overline{\sigma} (T-t)$ for t = 2023-04-12', color = 'green')
plt.axhline(y=sigmafit_short_old*np.sqrt(252), color='tab:blue', linestyle='--', label = 'Long term $\sigma$ (bps) from first 1/2 of data')
plt.axhline(y=sigmafit_short*np.sqrt(252), color='g', linestyle='--', label = 'Long term $\sigma$  (bps) from first 3/4 of data') 
plt.title('Annualized Volatility Term Structure For $\Delta F(t,T_0,T_1)$ (bps)')
plt.xlabel('T - t  (years)')
plt.ylabel('$\overline{\sigma} (T-t)$ (bps)')
plt.grid(True)
plt.legend()
#plt.ylim(0.4, 0.8)
plt.xlim(0, 2)  
plt.savefig("./plots/Volatility_term_structure_sfr.jpg")

############################# NOW WE REDO THE WHOLE ANALYSIS ANALYIS WITH LOG(SFR) CHANGE ##################
##### for clarity we redefine the viupdatefunc and objectivefunce even though they are the same as above ######

ui_array = sfrchange['logsfr_change'].to_numpy()*100
vi_array = np.zeros(len(ui_array))

def viupdatefunc(omega,alpha,beta):
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

print('\nThe fitted values for the GARCH(1,1) analysis of Delta Log F are:')
print(f'omega = {omegafit}')
print(f'alpha = {alphafit}')
print(f'beta = {betafit}')
print(f'alpha + beta = {alphafit+betafit}')

sigmai_array = np.sqrt(vi_array)

sigmafit = np.sqrt(omegafit/(1-alphafit-betafit))

print(f'The long term average volatility for Delta Log F is: {sigmafit} (%)')

######## We multiplied by 100 above so this is in "percentage" (%) #######
sfrchange['GARCH_volatility_log_change'] = sigmai_array


##### use arch as a check ######

returns = sfrchange['logsfr_change']*100

model = arch_model(returns, vol='Garch', p=1, q=1, mean='Zero')
res = model.fit(disp='off')


#print('\nThe summary of the GARCH(1,1) fit using the arch library is:')
#print(res.summary())

conditional_vol = res.conditional_volatility


plt.figure(figsize=(12, 6))
plt.plot(sfrchange['observation_date'], returns, color='grey', alpha=0.5, label='Returns')
plt.plot(sfrchange['observation_date'], conditional_vol, color='red', label='$\sigma(t)$ arch package')
plt.plot(sfrchange['observation_date'], sfrchange['GARCH_volatility_log_change'], color='blue', linestyle='-', label = '$\sigma(t)$ own code')
plt.axhline(y=sigmafit, color='r', linestyle='--', label = 'Long term $\sigma$') 
plt.title('GARCH(1,1) Conditional Volatility for the Change in Log(SFR)')
plt.legend()
plt.savefig("./plots/GARCH_volatility_log_change_using_library.jpg")
#plt.show()

##### 


'''
Note: A time series with small autocorrelation but large autocorrelation in its squared values exhibits
strong non-linear dependencies, specifically a characteristic known as volatility clustering. This is a common feature of financial data
'''


################## PACF FOR THE DELTA LOG(SFR) DATASET ################################


plt.close('all')

pacf_values = pacf(ui_array, nlags=10)
#print('Partial Autorcorrelations values of ui are:', pacf_values)

fig, ax = plt.subplots(figsize=(10, 5))
plot_pacf(ui_array, lags=10, ax=ax) # 'lags' parameter specifies the number of lags to display
plt.title('Partial Autocorrelation Function (PACF) Plot for $\Delta \log F(t,T_0,T_1)$')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation Coefficient')
plt.savefig("./plots/Partial_Autocorrelations_logsfr_change_ui.jpg")
#plt.show()


plt.close('all')


pacf_values = pacf(ui_array**2, nlags=10)
#print('Partial Autorcorrelations values of ui^2 are:', pacf_values)

fig, ax = plt.subplots(figsize=(10, 5))
plot_pacf(ui_array**2, lags=10, ax=ax) # 'lags' parameter specifies the number of lags to display
plt.title('Partial Autocorrelation Function (PACF) Plot for $(\Delta \log F(t,T_0,T_1))^2$')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation Coefficient')
plt.savefig("./plots/Partial_Autocorrelations_logsfr_change_ui2.jpg")
#plt.show()

plt.close('all')

pacf_values = pacf(ui_array**2/sigmai_array**2, nlags=10)
#print('Partial autorcorrelations values are of ui**2/sigmai**2:', pacf_values)

fig, ax = plt.subplots(figsize=(10, 5))
plot_pacf(ui_array**2/sigmai_array**2, lags=10, ax=ax) # 'lags' parameter specifies the number of lags to display
plt.title('Partial Autocorrelation Function (PACF) Plot for $(\Delta \log F(t,T_0,T_1))^2/\sigma_i^2$')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation Coefficient')
plt.savefig("./plots/Partial_Autocorrelations_logsfr_change_ui2onsigmai2.jpg")
#plt.show()

plt.close('all')


##### DO GARCH(1,1) ANALYSIS ON SHORTER DATASET AND THEN PROJECT VOLATILITY INTO THE FUTURE ######
##### USING HALF OF THE DATASET HERE ########################################################
##### for clarity we redefine the viupdatefunc and objectivefunce even though they are the same as above ######

ui_array_full = sfrchange['logsfr_change'].to_numpy()*100

ui_array_short = ui_array_full[0:-951]
ui_array = ui_array_short
vi_array = np.zeros(len(ui_array))

def viupdatefunc(omega,alpha,beta):
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


sigmai_array_short = np.sqrt(vi_array)

sigmafit_short = np.sqrt(omegafit/(1-alphafit-betafit))


sigmai_array_full = np.concatenate( (sigmai_array_short, np.zeros( len(ui_array_full)-len(ui_array_short) ) ) )


for i in range(len(ui_array_short),len(ui_array_full)):
	sigmai_array_full[i] = np.sqrt( omegafit/(1-alphafit-betafit) + (alphafit+betafit)**(i+1-len(ui_array_short))*(sigmai_array_full[len(ui_array_short)-1]**2 - omegafit/(1-alphafit-betafit)) )

sfrchange['GARCH_volatility_log_change_prediction'] = sigmai_array_full

############# ESTIMATE VOLATILITY TERM STRUCTURE - to be plotted later #######################

tempindex = 952

aparameter = -np.log( alphafit + betafit )

Vlong = sigmafit_short**2
Vzero = sfrchange.loc[tempindex, 'GARCH_volatility_log_change_prediction']**2

def sigmaterm(T):
	sigmaestimate2 = 252*(Vlong + (1 - np.exp(-aparameter*T) )/(aparameter*T)*(Vzero - Vlong))
	return(np.sqrt(sigmaestimate2))

x = np.linspace(0.0001, 2, 1000)
y = sigmaterm(365*x)


########################################################################
##### Now with 3/4 of the dataset ##################################
##### for clarity we redefine the viupdatefunc and objectivefunce even though they are the same as above ######

ui_array_full = sfrchange['logsfr_change'].to_numpy()*100
ui_array_short = ui_array_full[0:-471]

ui_array = ui_array_short
vi_array = np.zeros(len(ui_array))

def viupdatefunc(omega,alpha,beta):
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

sigmai_array_short = np.sqrt(vi_array)

sigmafit_short_old = sigmafit_short
sigmafit_short = np.sqrt(omegafit/(1-alphafit-betafit))

sigmai_array_full = np.concatenate( (sigmai_array_short, np.zeros( len(ui_array_full)-len(ui_array_short) ) ) )


for i in range(len(ui_array_short),len(ui_array_full)):
	sigmai_array_full[i] = np.sqrt( omegafit/(1-alphafit-betafit) + (alphafit+betafit)**(i+1-len(ui_array_short))*(sigmai_array_full[len(ui_array_short)-1]**2 - omegafit/(1-alphafit-betafit)) )

sfrchange['GARCH_volatility_log_change_prediction_2'] = sigmai_array_full

##### plot the log change and the predictions using 1/2 and 3/4 of the datasets - convert units to "percentage" ########
plt.figure(figsize=(8, 6))
plt.plot(sfrchange['observation_date'], sfrchange['GARCH_volatility_log_change'], linestyle='-', color = 'tab:blue', label = '$\sigma(t)$ (%)')
plt.plot(sfrchange.loc[(len(sfrchange)-951):,'observation_date'], sfrchange.loc[(len(sfrchange)-951):,'GARCH_volatility_log_change_prediction'], linestyle='-', linewidth=2, color = 'green', label = '$\sigma(t)$ (%) extrapolated from first 1/2 of data')
plt.plot(sfrchange.loc[(len(sfrchange)-476):,'observation_date'], sfrchange.loc[(len(sfrchange)-476):,'GARCH_volatility_log_change_prediction_2'], linestyle='-', linewidth=2, color = 'red', label = '$\sigma(t)$ (%) extrapolated from first 3/4 of data')
plt.axhline(y=sigmafit, color='tab:blue', linestyle='--', label = 'Long term $\sigma$ (%)') 
plt.axhline(y=sigmafit_short_old, color='g', linestyle='--', label = 'Long term $\sigma$ (%) from first 1/2 of data') 
plt.legend()
plt.title(f'Log SFR change GARCH(1,1) volatility')
plt.xlabel('Date')
plt.ylabel('Volatility $\sigma$ (%)')
plt.xticks(rotation=45, ha='right') # Optional: Rotate x-axis labels to prevent overcrowding
ax = plt.gca() # get current axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) # Optional: Format the date labels concisely (especially for large datasets)
plt.tight_layout() # Adjust layout to make room for x-axis labels
plt.grid(True)
plt.savefig("./plots/GARCH_volatility_log_change_with_prediction.jpg")
#plt.show()
plt.clf()


##### Volatility term structure for the two GARCH predictions with plot ################


#print(sfrchange.loc[(len(sfrchange)-951),'observation_date'])
#print(sfrchange.loc[(len(sfrchange)-476),'observation_date'])

tempindex = 1427

aparameter = -np.log( alphafit + betafit )

Vlong = sigmafit_short**2
Vzero = sfrchange.loc[tempindex, 'GARCH_volatility_log_change_prediction_2']**2

def sigmaterm(T):
	sigmaestimate2 = 252*(Vlong + ( 1 - np.exp(-aparameter*T) )/(aparameter*T)*(Vzero - Vlong))
	return(np.sqrt(sigmaestimate2))

x = np.linspace(0.0001, 2, 1000)
ynew = sigmaterm(365*x)

plt.plot(x, y, label='$\overline{\sigma} (T-t)$ for t = 2020-11-15', color = 'tab:blue') 
plt.plot(x, ynew, label='$\overline{\sigma} (T-t)$ for t = 2023-04-12', color = 'green')
plt.axhline(y=sigmafit_short_old*np.sqrt(252), color='tab:blue', linestyle='--', label = 'Long term $\sigma$ from first 1/2 of data')
plt.axhline(y=sigmafit_short*np.sqrt(252), color='g', linestyle='--', label = 'Long term $\sigma$ from first 3/4 of data') 
plt.title('Annualized Volatility Term Structure For $\Delta \log F(t,T_0,T_1)$')
plt.xlabel('T - t  (years)')
plt.ylabel('$\overline{\sigma} (T-t)$ (%)')
plt.grid(True)
plt.legend()
plt.ylim(40, 80)
plt.xlim(0, 2)  
plt.savefig("./plots/Volatility_term_structure_logsfr.jpg")





