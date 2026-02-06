import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from scipy.optimize import minimize

plt.rcParams.update({'font.size': 12})


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

###### Make a function calculating GARCH(1,1) volatility using only the data up to the present day  --- if under 1000 days of data: use data up to 1000th day.
### could be made more efficient by only optimizing once for the first 1000 days. But we only need to run this once so the time saved in computation will be wasted in time lost coding.

def RunningGarch(t):
	##### find day index
	present_day_index = sfrchange.index[sfrchange['observation_date'] == t][0]
	
	print(f'{present_day_index/len(sfrchange)*100:.2f} % complete')
	
	data_index = np.maximum(present_day_index, 999)
	
	#### multiply by 100 to make the optimization algorithm run better. Results are in ("%").
	ui_array_temp = sfrchange.loc[0:(data_index),'logsfr_change'].to_numpy()*100
	vi_array_temp = np.zeros(len(ui_array_temp))
	
	##############################################
	def vitempupdatefunc(omega,alpha,beta):
		vi_array_temp[0] = omega
		vi_array_temp[0] = omega/(1-alpha-beta)
		for i in range(1,len(vi_array_temp)):
			vi_array_temp[i] =  omega + alpha*ui_array_temp[i-1]**2 + beta*vi_array_temp[i-1]	
		return(vi_array_temp)
	###############################################
	
	####################
	def objectivefunc2(q):
		omega = q[0]
		alpha = q[1]
		beta = q[2]
		
		vitemp = 0
		vitemp = vitempupdatefunc(omega,alpha,beta)
		terms = np.log(vitemp) + ui_array_temp**2/vitemp
		return(np.sum(terms))
	####################
	
	q0 = [0.01,0.01,0.9]

	cons = [{'type':'ineq', 'fun': lambda q: q - 0.0001},
		{'type':'ineq', 'fun': lambda q: 0.999 - q[1] - q[2] }]

	res2 = minimize(objectivefunc2, q0, method='SLSQP', constraints=cons, bounds=None, tol=1e-6)

	omegafit = res2.x[0]
	alphafit = res2.x[1]
	betafit = res2.x[2]

	vi_array_temp = vitempupdatefunc(omegafit,alphafit,betafit)
	
	sigmai_array_temp = np.sqrt(vi_array_temp)
	
#	print(len(sigmai_array_temp))
	
	#### return in the result (in '%') ####
	return(sigmai_array_temp[present_day_index])


##### Do the running GARCH computation

print('\nStarting the running GARCH computation')

sfrchange['GARCH_volatility_logsfr_change (%)'] = sfrchange['observation_date'].apply(RunningGarch)
sfrchange['GARCH_volatility_logsfr_change'] = sfrchange['GARCH_volatility_logsfr_change (%)']/100

print('\nThe dataframe of SFR changes with the running garch calculation is:')
print(sfrchange)

sigmafit = sfrchange['GARCH_volatility_logsfr_change (%)'].mean()


##### make a plot of the running GARCH
plt.figure(figsize=(8, 6))
plt.plot(sfrchange['observation_date'], sfrchange['GARCH_volatility_logsfr_change (%)'], linestyle='-', label = '$\sigma(t)$')
plt.axhline(y=sigmafit, color='r', linestyle='--', label = 'Long term $\sigma$') 
plt.legend()
plt.title(f'Running Log SFR change GARCH(1,1) volatility')
plt.xlabel('Date')
plt.ylabel('Volatility $\sigma$ (%)')
#plt.ylim(-0.05, 0.16)
plt.xticks(rotation=45, ha='right') # Optional: Rotate x-axis labels to prevent overcrowding
ax = plt.gca() # get current axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) # Optional: Format the date labels concisely (especially for large datasets)
plt.tight_layout() # Adjust layout to make room for x-axis labels
plt.grid(True)
plt.savefig("./plots/Running_GARCH_volatility_log_change.jpg")
#plt.show()
plt.clf()

################# WRITE OUT RUNNING GARCH(1,1) VOLATILITY #############################

sfrchange['GARCH_volatility_logsfr_change (%)'].to_csv('./data/Running_Garch_Vol.csv', index=False, header=True)



