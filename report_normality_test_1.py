import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats as stats
import statsmodels.api as sm
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from scipy.stats import kstest, norm, anderson
from statsmodels.stats.diagnostic import kstest_normal
from statsmodels.stats.diagnostic import lilliefors

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
	
	#print(obsdate)
	#print(tempyields.iloc[-1])
	
	x_new = np.linspace(1/12, 30, 1000) 
	y_new = interpolating_function(x_new)
	
	plt.figure(figsize=(8, 6))
	plt.plot(maturities, tempyields, 'o', label='Original Data Points')
	plt.plot(x_new, y_new, '-', label='Cubic Interpolated Function')
	plt.legend()
	plt.title('Interpolation using scipy.interpolate.interp1d')
	plt.xlabel('Maturity (years)')
	plt.ylabel('Yield (%)')
	plt.grid(True)
	plt.show()
	
	return(interpolating_function(4.4))

### Now define a function which returns the discount bond curve

def discountcurveapprox(T1, T2, args):
	obsdate = args.iloc[0,0]
	
	maturities = np.array([0, 1/12, 1/4, 1/2, 1, 2, 3, 5, 7, 10, 20, 30])
	tempzcbs = args.iloc[0,1:]
	
	#print(tempzcbs)
	
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

###### Now begin analysis of the Simple Forward Rate observations ######
### plot the behaviour of the simple forward rate

plt.figure(figsize=(8, 6))
plt.plot(sfr_df['observation_date'], sfr_df['sfr']*100, linestyle='-', label = 'F(t,T0,T1)')
plt.legend()
plt.title('Time series of SFR with $T_0$ = 2025-09-22 and $T_1$ = 2025-12-22')
plt.xlabel('Date')
plt.ylabel('Simple Forward Rate (%)')
plt.xticks(rotation=45, ha='right') # Optional: Rotate x-axis labels to prevent overcrowding
ax = plt.gca() # get current axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) # Optional: Format the date labels concisely (especially for large datasets)
plt.tight_layout() # Adjust layout to make room for x-axis labels
plt.grid(True)
plt.savefig("./plots/SFR_timeseries.jpg")
#plt.show()
plt.clf()

#### plot the time series of the log of the simple forward rate

plt.figure(figsize=(8, 6))
plt.plot(sfr_df['observation_date'], np.log(sfr_df['sfr']*100), linestyle='-', label = 'log[F(t,T0,T1) (%)]')
plt.legend()
plt.title('Time series of SFR with $T_0$ = 2025-09-22 and $T_1$ = 2025-12-22')
plt.xlabel('Date')
plt.ylabel('Logrithm of Simple Forward Rate log[F(t,T0,T1) (%)]')
plt.xticks(rotation=45, ha='right') # Optional: Rotate x-axis labels to prevent overcrowding
ax = plt.gca() # get current axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) # Optional: Format the date labels concisely (especially for large datasets)
plt.tight_layout() # Adjust layout to make room for x-axis labels
plt.grid(True)
plt.savefig("./plots/Log_SFR_timeseries.jpg")
#plt.show()
plt.clf()


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


######### TEST THE BACHELIER TYPE HYPOTHESIS WITH ONLY DELTA 1 DAY DATA  #################

print('\nWe now test the Bachelier type hypothesis:') 

### calculate mean etc.

mean_sfr_change = sfrchange['sfr_change (bps)'].mean()
std_sfr_sem = sfrchange['sfr_change (bps)'].sem()
std_sfr_change = sfrchange['sfr_change (bps)'].std()

print('Estimate mean from sample is (bps):', mean_sfr_change, '+/-', 1.645*std_sfr_sem, 'Range:', mean_sfr_change-1.645*std_sfr_sem, mean_sfr_change+1.645*std_sfr_sem )
print('Sample standard deviation is (bps):', std_sfr_change, '+/-', 1.645*std_sfr_change/np.sqrt(2*len(sfrchange)), 'Range:', std_sfr_change-1.645*std_sfr_change/np.sqrt(2*len(sfrchange)),  std_sfr_change+1.645*std_sfr_change/np.sqrt(2*len(sfrchange)) )

##### save the range in the standard deviation for comparison to the weekly volatility later ######
minrange1 = std_sfr_change-1.645*std_sfr_change/np.sqrt(2*len(sfrchange))
maxrange1 = std_sfr_change+1.645*std_sfr_change/np.sqrt(2*len(sfrchange))
#####

# Create the histogram of the results to compare to the Normal PDF
fig, ax = plt.subplots(figsize=(8, 5)) # Use ax=ax to plot on the same axes object

# Generate points for the theoretical normal distribution curve, Sort values for a smooth line plot 
x_values = np.sort(sfrchange['sfr_change (bps)']) 
pdf = norm.pdf(x_values, mean_sfr_change, std_sfr_change)

sfrchange['sfr_change (bps)'].plot.hist(bins=60, title='Histogram of the SFR changes', density=True, alpha=0.6, color='blue', ax=ax)
ax.plot(x_values, pdf, color='red', linewidth=2, label='Normal Distribution Fit') # Plot the PDF line
plt.xlabel('$\Delta$ F(t,$T_0$,$T_1$) (bps)')
plt.ylabel('Normalized Frequency')
plt.savefig("./plots/PDF_Bachelier_for_report.jpg")
#plt.show()
plt.clf()

### Now perform various statistical tests ###########

###### make a numpy array of the data we want to test ####
sfrchange_array = sfrchange['sfr_change (bps)'].to_numpy()

# Perform the Lilliefors test for a normal distribution

ks_statistic, p_value = lilliefors(sfrchange_array, dist='norm', pvalmethod='approx')
print(f"\nLillierfors test for Bachelier")
print(f"Lillierfors test KS Statistic: {ks_statistic}")
print(f"P-value: {p_value}")

# Perform the Shapiro-Wilk test for a normal distribution

shapiro_statistic, shapiro_p_value = stats.shapiro(sfrchange_array)
print(f"\nShapiro-Wilk test for Bachelier")
print(f"Shapiro-Wilk Statistic: {shapiro_statistic}")
print(f"P-value: {shapiro_p_value}")

# Perform the Anderson-Darling test
result = anderson(sfrchange_array)

# Print the results
print(f"\nAnderson-Darling test for Bachelier")
print(f"Test Statistic: {result.statistic}")
print("Critical Values and Significance Levels:")
for i in range(len(result.critical_values)):
    sl = result.significance_level[i]
    cv = result.critical_values[i]
    print(f"  Significance Level: {sl}% | Critical Value: {cv:.4f}")

#### make Q-Q plot for Bachelier #######

plt.close('all')

stats.probplot(sfrchange_array, dist="norm", plot=plt)

# Add plot titles and labels
plt.title("Q-Q Plot of $\Delta F$ Data vs Normal Distribution")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Observed Data Quantiles $\Delta F$ (bps)")
plt.grid(True)

# Display or save the plot
#plt.show()
plt.savefig("./plots/QQ_Bachelier_for_report.jpg")

######### TEST THE BLACK TYPE HYPOTHESIS #################
### calculate mean etc.

print('\nWe now test the Black type hypothesis:') 

mean_logsfr_change = sfrchange['logsfr_change'].mean()
std_logsfr_sem = sfrchange['logsfr_change'].sem()
std_logsfr_change = sfrchange['logsfr_change'].std()

print('Estimate mean from sample is:', mean_logsfr_change, '+/-', 1.645*std_logsfr_sem, 'Range:', mean_logsfr_change-1.645*std_logsfr_sem, mean_logsfr_change+1.645*std_logsfr_sem )

print('Sample standard deviation is:', std_logsfr_change, '+/-', 1.645*std_logsfr_change/np.sqrt(2*len(sfrchange)), 'Range:', std_logsfr_change-1.645*std_logsfr_change/np.sqrt(2*len(sfrchange)),  std_logsfr_change+1.645*std_logsfr_change/np.sqrt(2*len(sfrchange)) )


##### save the range in the standard deviation for comparison to the weekly volatility later ######
minrange2 = std_logsfr_change-1.645*std_logsfr_change/np.sqrt(2*len(sfrchange))
maxrange2 = std_logsfr_change+1.645*std_logsfr_change/np.sqrt(2*len(sfrchange))
#####

# Create the histogram of the results to compare to the Normal PDF
fig, ax = plt.subplots(figsize=(8, 5)) # Use ax=ax to plot on the same axes object

# Generate points for the theoretical normal distribution curve, Sort values for a smooth line plot 
x_values = np.sort(sfrchange['logsfr_change']) 
pdf = norm.pdf(x_values, mean_logsfr_change, std_logsfr_change)

sfrchange['logsfr_change'].plot.hist(bins=60, title='Histogram of the log SFR changes', density=True, alpha=0.6, color='blue', ax=ax)
ax.plot(x_values, pdf, color='red', linewidth=2, label='Normal Distribution Fit') # Plot the PDF line
plt.xlabel('$\Delta$ log$_e$ F(t,$T_0$,$T_1$)')
plt.ylabel('Normalized Frequency')
plt.savefig("./plots/PDF_Black_for_report.jpg")
#plt.show()
plt.clf()

### Now perform various statistical tests ###########

logsfrchange_array = sfrchange['logsfr_change'].to_numpy()
# Perform the Lilliefors test for a normal distribution

ks_statistic, p_value = lilliefors(logsfrchange_array, dist='norm', pvalmethod='approx')
print(f"\nLillierfors test for Black")
print(f"Lillierfors test KS Statistic: {ks_statistic}")
print(f"P-value: {p_value}")

# Perform the Shapiro-Wilk test for a normal distribution

shapiro_statistic, shapiro_p_value = stats.shapiro(logsfrchange_array)
print(f"\nShapiro-Wilk test for Black")
print(f"Shapiro-Wilk Statistic: {shapiro_statistic}")
print(f"P-value: {shapiro_p_value}")

# Perform the Anderson-Darling test
result = anderson(logsfrchange_array)

# Print the results
print(f"\nAnderson-Darling test for Black")
print(f"Test Statistic: {result.statistic}")
print("Critical Values and Significance Levels:")
for i in range(len(result.critical_values)):
    sl = result.significance_level[i]
    cv = result.critical_values[i]
    print(f"  Significance Level: {sl}% | Critical Value: {cv:.4f}")

#### make Q-Q plot #######

plt.close('all')

stats.probplot(logsfrchange_array, dist="norm", plot=plt)

# Add plot titles and labels
plt.title("Q-Q Plot of $\Delta \log F$ Data vs Normal Distribution")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Observed Data Quantiles $\Delta \log F$")
plt.grid(True)

# Display the plot
#plt.show()
plt.savefig("./plots/QQ_Black_for_report.jpg")



################### ANALYSIS OF LARGE JUMPS #####################################
#### show the five largest positive and negative jumps for the  Bachelier and  Black analysis


print('\nLargest positive jumps for Bachelier (bps)')
print(sfrchange['sfr_change (bps)'].nlargest(5))

print('\nLargest negative jumps for Bachelier (bps)')
print(sfrchange['sfr_change (bps)'].nsmallest(5))

print('\nLargest positive jumps for Bachelier (z-scopres)')
print((sfrchange['sfr_change (bps)'].nlargest(5)-mean_sfr_change)/std_sfr_change)

print('\nLargest negative jumps for Bachelier (z-scopres)')
print((sfrchange['sfr_change (bps)'].nsmallest(5)-mean_sfr_change)/std_sfr_change)




print('\nLargest positive jumps for Black (raw data)')
print(sfrchange['logsfr_change'].nlargest(5))

print('\nLargest negative jumps for Black (raw data)')
print(sfrchange['logsfr_change'].nsmallest(5))


print('\nLargest positive jumps for Black (z-scores)')
print((sfrchange['logsfr_change'].nlargest(5)-mean_logsfr_change)/std_logsfr_change)

print('\nLargest negative jumps for Black (z-scores)')
print((sfrchange['logsfr_change'].nsmallest(5)-mean_logsfr_change)/std_logsfr_change)


#####################################################################################
#### Plot the largest jumps ##################################################

###### plot of the largest negative value for Delta F #################################

rowcheck1 = sfrchange[sfrchange['sfr_change'] == sfrchange['sfr_change'].min()]
tcheck1 = rowcheck1['observation_date'].iloc[0]
tcheckmin1 = tcheck1 - timedelta(days=1)



delta = Tfix2 - Tfix1
delta = float(delta.days/365)

tinterval1 = Tfix1 - tcheck1
tinterval2 = Tfix2 - tcheck1
	
tinterval1 = float(tinterval1.days/365)
tinterval2 = float(tinterval2.days/365)

xcoords = [tinterval1, tinterval2]

argstemp1 = zcbonds[zcbonds['observation_date'] == tcheck1]
argstempmin1 = zcbonds[zcbonds['observation_date'] == tcheckmin1]
	
#zcb1, zcb2 = discountcurveapprox2(tinterval1, tinterval2, argstemp)
#simpleforwardrate = 1/delta*(zcb1/zcb2 - 1)

maturities = np.array([0, 1/12, 1/4, 1/2, 1, 2, 3, 5, 7, 10, 20, 30])

obsdate = argstemp1.iloc[0,0]	
tempzcbs = argstemp1.iloc[0,1:]

obsdatemin1 = argstempmin1.iloc[0,0]	
tempzcbsmin1 = argstempmin1.iloc[0,1:]


interpolating_function = interp1d(maturities, tempzcbs, kind='cubic')
interpolating_function_min1 = interp1d(maturities, tempzcbsmin1, kind='cubic')
	
x_new = np.linspace(0, 30, 1000) 
y_new = interpolating_function(x_new)
y_new_min1 = interpolating_function_min1(x_new)
	
plt.figure(figsize=(8, 6))
plt.plot(maturities, tempzcbs, 'o', label='Data Points at t', color='blue')
plt.plot(maturities, tempzcbsmin1, 'o', label='Data Points at t - 1 day', color='green')

plt.plot(x_new, y_new, '-', label='Interpolated Function at t', color='blue')
plt.plot(x_new, y_new_min1, '-', label='Interpolated Function at t - 1 day', color='green')

# Draw vertical lines at T1fix and T2fix
plt.axvline(xcoords[0], color='r', linestyle='--', label='T0')
plt.axvline(xcoords[1], color='r', linestyle='--', label='T1')
	
tcheckstr = tcheck1.strftime("%Y-%m-%d")

plt.legend()
plt.title(f'ZCB prices with t = {tcheck1.strftime("%Y-%m-%d")}')
plt.xlabel('Time to Maturity (Years)')
plt.ylabel('ZCB price')
plt.grid(True)
plt.xlim(0, 6) 
plt.ylim(0.8, 1) 	
plt.savefig("./plots/Min_sfr_jump_ZCBs.jpg") 	

###### plot of the largest positive value for Delta F #################################

rowcheck1 = sfrchange[sfrchange['sfr_change'] == sfrchange['sfr_change'].max()]
tcheck1 = rowcheck1['observation_date'].iloc[0]
tcheckmin1 = tcheck1 - timedelta(days=1)

delta = Tfix2 - Tfix1
delta = float(delta.days/365)

tinterval1 = Tfix1 - tcheck1
tinterval2 = Tfix2 - tcheck1
	
tinterval1 = float(tinterval1.days/365)
tinterval2 = float(tinterval2.days/365)

xcoords = [tinterval1, tinterval2]

argstemp1 = zcbonds[zcbonds['observation_date'] == tcheck1]
argstempmin1 = zcbonds[zcbonds['observation_date'] == tcheckmin1]

maturities = np.array([0, 1/12, 1/4, 1/2, 1, 2, 3, 5, 7, 10, 20, 30])

obsdate = argstemp1.iloc[0,0]	
tempzcbs = argstemp1.iloc[0,1:]

obsdatemin1 = argstempmin1.iloc[0,0]	
tempzcbsmin1 = argstempmin1.iloc[0,1:]


interpolating_function = interp1d(maturities, tempzcbs, kind='cubic')
interpolating_function_min1 = interp1d(maturities, tempzcbsmin1, kind='cubic')
	
x_new = np.linspace(0, 30, 1000) 
y_new = interpolating_function(x_new)
y_new_min1 = interpolating_function_min1(x_new)
	
plt.figure(figsize=(8, 6))
plt.plot(maturities, tempzcbs, 'o', label='Data Points at t', color='blue')
plt.plot(maturities, tempzcbsmin1, 'o', label='Data Points at t - 1 day', color='green')

plt.plot(x_new, y_new, '-', label='Interpolated Function at t', color='blue')
plt.plot(x_new, y_new_min1, '-', label='Interpolated Function at t - 1 day', color='green')

# Draw vertical lines at T1fix and T2fix
plt.axvline(xcoords[0], color='r', linestyle='--', label='T0')
plt.axvline(xcoords[1], color='r', linestyle='--', label='T1')
	
tcheckstr = tcheck1.strftime("%Y-%m-%d")

plt.legend()
plt.title(f'ZCB prices with t = {tcheck1.strftime("%Y-%m-%d")}')
plt.xlabel('Time to Maturity (Years)')
plt.ylabel('ZCB price')
plt.grid(True)
plt.xlim(0, 8) 
plt.ylim(0.9, 1.01) 	
plt.savefig("./plots/Max_sfr_jump_ZCBs.jpg") 	

###### plot of the largest negative value for Delta log F #################################

rowcheck1 = sfrchange[sfrchange['logsfr_change'] == sfrchange['logsfr_change'].min()]
tcheck1 = rowcheck1['observation_date'].iloc[0]
tcheckmin1 = tcheck1 - timedelta(days=1)

delta = Tfix2 - Tfix1
delta = float(delta.days/365)

tinterval1 = Tfix1 - tcheck1
tinterval2 = Tfix2 - tcheck1
	
tinterval1 = float(tinterval1.days/365)
tinterval2 = float(tinterval2.days/365)

xcoords = [tinterval1, tinterval2]

argstemp1 = zcbonds[zcbonds['observation_date'] == tcheck1]
argstempmin1 = zcbonds[zcbonds['observation_date'] == tcheckmin1]
	
#zcb1, zcb2 = discountcurveapprox2(tinterval1, tinterval2, argstemp)
#simpleforwardrate = 1/delta*(zcb1/zcb2 - 1)

maturities = np.array([0, 1/12, 1/4, 1/2, 1, 2, 3, 5, 7, 10, 20, 30])

obsdate = argstemp1.iloc[0,0]	
tempzcbs = argstemp1.iloc[0,1:]

obsdatemin1 = argstempmin1.iloc[0,0]	
tempzcbsmin1 = argstempmin1.iloc[0,1:]


interpolating_function = interp1d(maturities, tempzcbs, kind='cubic')
interpolating_function_min1 = interp1d(maturities, tempzcbsmin1, kind='cubic')
	
x_new = np.linspace(0, 30, 1000) 
y_new = interpolating_function(x_new)
y_new_min1 = interpolating_function_min1(x_new)
	
plt.figure(figsize=(8, 6))
plt.plot(maturities, tempzcbs, 'o', label='Data Points at t', color='blue')
plt.plot(maturities, tempzcbsmin1, 'o', label='Data Points at t - 1 day', color='green')

plt.plot(x_new, y_new, '-', label='Interpolated Function at t', color='blue')
plt.plot(x_new, y_new_min1, '-', label='Interpolated Function at t - 1 day', color='green')

# Draw vertical lines at T1fix and T2fix
plt.axvline(xcoords[0], color='r', linestyle='--', label='T0')
plt.axvline(xcoords[1], color='r', linestyle='--', label='T1')
	
tcheckstr = tcheck1.strftime("%Y-%m-%d")

plt.legend()
plt.title(f'ZCB prices with t = {tcheck1.strftime("%Y-%m-%d")}')
plt.xlabel('Time to Maturity (Years)')
plt.ylabel('ZCB price')
plt.grid(True)
plt.xlim(0, 8) 
plt.ylim(0.9, 1.01)
plt.savefig("./plots/Min_log_sfr_jump_ZCBs.jpg")

###### plot of the largest positive value for Delta log F #################################

rowcheck1 = sfrchange[sfrchange['logsfr_change'] == sfrchange['logsfr_change'].max()]
tcheck1 = rowcheck1['observation_date'].iloc[0]
tcheckmin1 = tcheck1 - timedelta(days=1)

delta = Tfix2 - Tfix1
delta = float(delta.days/365)

tinterval1 = Tfix1 - tcheck1
tinterval2 = Tfix2 - tcheck1
	
tinterval1 = float(tinterval1.days/365)
tinterval2 = float(tinterval2.days/365)

xcoords = [tinterval1, tinterval2]

argstemp1 = zcbonds[zcbonds['observation_date'] == tcheck1]
argstempmin1 = zcbonds[zcbonds['observation_date'] == tcheckmin1]

maturities = np.array([0, 1/12, 1/4, 1/2, 1, 2, 3, 5, 7, 10, 20, 30])

obsdate = argstemp1.iloc[0,0]	
tempzcbs = argstemp1.iloc[0,1:]

obsdatemin1 = argstempmin1.iloc[0,0]	
tempzcbsmin1 = argstempmin1.iloc[0,1:]


interpolating_function = interp1d(maturities, tempzcbs, kind='cubic')
interpolating_function_min1 = interp1d(maturities, tempzcbsmin1, kind='cubic')
	
x_new = np.linspace(0, 30, 1000) 
y_new = interpolating_function(x_new)
y_new_min1 = interpolating_function_min1(x_new)
	
plt.figure(figsize=(8, 6))
plt.plot(maturities, tempzcbs, 'o', label='Data Points at t', color='blue')
plt.plot(maturities, tempzcbsmin1, 'o', label='Data Points at t - 1 day', color='green')

plt.plot(x_new, y_new, '-', label='Interpolated Function at t', color='blue')
plt.plot(x_new, y_new_min1, '-', label='Interpolated Function at t - 1 day', color='green')

# Draw vertical lines at T1fix and T2fix
plt.axvline(xcoords[0], color='r', linestyle='--', label='T0')
plt.axvline(xcoords[1], color='r', linestyle='--', label='T1')
	
tcheckstr = tcheck1.strftime("%Y-%m-%d")

plt.legend()
plt.title(f'ZCB prices with t = {tcheck1.strftime("%Y-%m-%d")}')
plt.xlabel('Time to Maturity (Years)')
plt.ylabel('ZCB price')
plt.grid(True)
plt.xlim(0, 8) 
plt.ylim(0.9, 1.01)
plt.savefig("./plots/Max_log_sfr_jump_ZCBs.jpg") 	


############# SPLIT DATAFRAME IN HALF AND TEST FOR STATIONARITY - WHETHER THE TWO HALVES ARE DESCRIBED BY THE SAME DISTRIBUTION #################


# Calculate the halfway index
half_index = len(sfrchange) // 2

# Create the two new DataFrames
sfrchange_first_half = sfrchange.iloc[:half_index]  # Select from start to the halfway index
sfrchange_second_half = sfrchange.iloc[half_index:] # Select from the halfway index to the end

#print(len(sfrchange_first_half))
#print(len(sfrchange_second_half))

############ first for the change in the SFR i.e. Delta F in bps ############


##### Lilliefors tests ############################

sfrchange_array1 = sfrchange_first_half['sfr_change (bps)'].to_numpy()
sfrchange_array2 = sfrchange_second_half['sfr_change (bps)'].to_numpy()

ks_statistic, p_value = lilliefors(sfrchange_array1, dist='norm', pvalmethod='approx')
print("\nLilliefors test for first half of the data (Bachelier)")
print(f"KS Statistic: {ks_statistic}")
print(f"P-value: {p_value}")

ks_statistic, p_value = lilliefors(sfrchange_array2, dist='norm', pvalmethod='approx')
print("\n Lilliefors test for second half of the data (Bachelier)")
print(f"KS Statistic: {ks_statistic}")
print(f"P-value: {p_value}")


##### Shapiro-Wilk tests ############################

shapiro_statistic, shapiro_p_value = stats.shapiro(sfrchange_array1)
print("\nShapiro-Wilk test for first half of the data (Bachelier)")
print(f"Shapiro-Wilk Statistic: {shapiro_statistic}")
print(f"P-value: {shapiro_p_value}")

shapiro_statistic, shapiro_p_value = stats.shapiro(sfrchange_array2)
print("\nShapiro-Wilk test for second half of the data (Bachelier)")
print(f"Shapiro-Wilk Statistic: {shapiro_statistic}")
print(f"P-value: {shapiro_p_value}")

# Perform the Anderson-Darling test first half
result = anderson(sfrchange_array1)

# Print the results
print("\nAnderson-Darling test for first half of the data (Bachelier)")
print(f"Test Statistic: {result.statistic}")
print("Critical Values and Significance Levels:")
for i in range(len(result.critical_values)):
    sl = result.significance_level[i]
    cv = result.critical_values[i]
    print(f"  Significance Level: {sl}% | Critical Value: {cv:.4f}")

# Perform the Anderson-Darling test second half
result = anderson(sfrchange_array2)

# Print the results
print("\nAnderson-Darling test for second half of the data (Bachelier)")
print(f"Test Statistic: {result.statistic}")
print("Critical Values and Significance Levels:")
for i in range(len(result.critical_values)):
    sl = result.significance_level[i]
    cv = result.critical_values[i]
    print(f"  Significance Level: {sl}% | Critical Value: {cv:.4f}")
    
    
# Perform the two-sample KS test
ks_statistic, p_value = stats.ks_2samp(sfrchange_array1, sfrchange_array2)

print("\nTwo sample Kolmogorov-Smirnov test. First half vs Second half of the data (Bachelier):")
print(f"KS Statistic: {ks_statistic}")
print(f"P-value: {p_value}")

mean_sfr_change1 = sfrchange_first_half['sfr_change (bps)'].mean()
std_sfr_sem1 = sfrchange_first_half['sfr_change (bps)'].sem()
std_sfr_change1 = sfrchange_first_half['sfr_change (bps)'].std()

print('\nEstimate mean from 1st half sample is (bps):', mean_sfr_change1, '+/-', 1.645*std_sfr_sem1, 'Range:', mean_sfr_change1-1.645*std_sfr_sem1, mean_sfr_change1+1.645*std_sfr_sem1 )
print('Sample standard deviation from 1st half is (bps):', std_sfr_change1, '+/-', 1.645*std_sfr_change1/np.sqrt(2*len(sfrchange_first_half)), 'Range:', std_sfr_change1-1.645*std_sfr_change1/np.sqrt(2*len(sfrchange_first_half)),  std_sfr_change1+1.645*std_sfr_change1/np.sqrt(2*len(sfrchange_first_half)) )


mean_sfr_change2 = sfrchange_second_half['sfr_change (bps)'].mean()
std_sfr_sem2 = sfrchange_second_half['sfr_change (bps)'].sem()
std_sfr_change2 = sfrchange_second_half['sfr_change (bps)'].std()

print('\nEstimate mean from 2nd half sample is (bps):', mean_sfr_change2, '+/-', 1.645*std_sfr_sem2, 'Range:', mean_sfr_change2-1.645*std_sfr_sem2, mean_sfr_change2+1.645*std_sfr_sem2 )
print('Sample standard deviation from 2nd half is (bps):', std_sfr_change2, '+/-', 1.645*std_sfr_change2/np.sqrt(2*len(sfrchange_second_half)), 'Range:', std_sfr_change2-1.645*std_sfr_change2/np.sqrt(2*len(sfrchange_second_half)),  std_sfr_change2+1.645*std_sfr_change2/np.sqrt(2*len(sfrchange_second_half)) )


#######################################################################################
############ COMPARING FIRST AND SECOND HALVES FOR THE BLACK TYPE HYPOTHESIS ##########


logsfrchange_array1 = sfrchange_first_half['logsfr_change'].to_numpy()
logsfrchange_array2 = sfrchange_second_half['logsfr_change'].to_numpy()


##### Lilliefors tests ############################

ks_statistic, p_value = lilliefors(logsfrchange_array1, dist='norm', pvalmethod='approx')
print("\nLilliefors test for first half of the data (Black)")
print(f"KS Statistic: {ks_statistic}")
print(f"P-value: {p_value}")
ks_statistic, p_value = lilliefors(logsfrchange_array2, dist='norm', pvalmethod='approx')
print(f"\nKS Statistic: {ks_statistic}")
print(f"P-value: {p_value}")

ks_statistic, p_value = kstest(logsfrchange_array1, 'norm', args=(mean_logsfr_change, std_logsfr_change))
print("\nLilliefors test for second half of the data (Black)")
print(f"KS Statistic: {ks_statistic}")
print(f"P-value: {p_value}")
ks_statistic, p_value = kstest(logsfrchange_array2, 'norm', args=(mean_logsfr_change, std_logsfr_change))
print(f"\nKS Statistic: {ks_statistic}")
print(f"P-value: {p_value}")

shapiro_statistic, shapiro_p_value = stats.shapiro(logsfrchange_array1)
print("\nShapiro-Wilk test for first half of the data (Black)")
print(f"Shapiro-Wilk Statistic: {shapiro_statistic}")
print(f"P-value: {shapiro_p_value}")

shapiro_statistic, shapiro_p_value = stats.shapiro(logsfrchange_array2)
print("\nShapiro-Wilk test for second half of the data (Black)")
print(f"Shapiro-Wilk Statistic: {shapiro_statistic}")
print(f"P-value: {shapiro_p_value}")

# Perform the Anderson-Darling test
result = anderson(logsfrchange_array1)

# Print the results
print("\nAnderson-Darling test for first half of the data (Black)")
print(f"Test Statistic: {result.statistic}")
print("Critical Values and Significance Levels:")
for i in range(len(result.critical_values)):
    sl = result.significance_level[i]
    cv = result.critical_values[i]
    print(f"  Significance Level: {sl}% | Critical Value: {cv:.4f}")

# Perform the Anderson-Darling test
result = anderson(logsfrchange_array2)

# Print the results
print("\nAnderson-Darling test for second half of the data (Black)")
print(f"Test Statistic: {result.statistic}")
print("Critical Values and Significance Levels:")
for i in range(len(result.critical_values)):
    sl = result.significance_level[i]
    cv = result.critical_values[i]
    print(f"  Significance Level: {sl}% | Critical Value: {cv:.4f}")
    
    
# Perform the two-sample KS test
ks_statistic, p_value = stats.ks_2samp(logsfrchange_array1, logsfrchange_array2)

print("\nTwo sample Kolmogorov Smirnov test: first vs second half of the data (Black)")
print(f"KS Statistic: {ks_statistic}")
print(f"P-value: {p_value}")

########### Q-Q plot of the two halves for Bachelier type hypothesis #######################


plt.close('all')

fig = sm.qqplot_2samples(sfrchange_array1, sfrchange_array2, line='45') 

# Add titles and labels for clarity
plt.title("Q-Q Plot of 1st Half vs 2nd Half Bachelier")
plt.xlabel("Quantiles of 1st Half Data $\Delta F$ (bps)")
plt.ylabel("Quantiles of 2nd Half Data $\Delta F$ (bps)")
plt.grid(True)

# Save/Display the plot
#plt.show()
plt.savefig("./plots/FigQQ2sampleBachelier.jpg")

plt.close('all')

########### Q-Q plot of the two halves for Black type hypothesis #######################

fig = sm.qqplot_2samples(logsfrchange_array1, logsfrchange_array2, line='45') 

#  Add titles and labels for clarity
plt.title("Q-Q Plot of 1st Half vs 2nd Half Black")
plt.xlabel("Quantiles of 1st Half $\Delta \log F$ Data")
plt.ylabel("Quantiles of 2nd Half $\Delta \log F$ Data")
plt.grid(True)

# Save/Display the plot
#plt.show()
plt.savefig("./plots/FigQQ2sampleBlack.jpg")

plt.close('all')

##### calculate means and std deviations of the seperate halves for black

mean_logsfr_change1 = sfrchange_first_half['logsfr_change'].mean()
std_logsfr_sem1 = sfrchange_first_half['logsfr_change'].sem()
std_logsfr_change1 = sfrchange_first_half['logsfr_change'].std()

print('\nEstimate mean from 1st half sample is (Black):', mean_logsfr_change1, '+/-', 1.645*std_logsfr_sem1, 'Range:', mean_logsfr_change1-1.645*std_logsfr_sem1, mean_logsfr_change1+1.645*std_logsfr_sem1 )
print('Sample standard deviation from 1st half is (Black):', std_logsfr_change1, '+/-', 1.645*std_logsfr_change1/np.sqrt(2*len(sfrchange_first_half)), 'Range:', std_logsfr_change1-1.645*std_logsfr_change1/np.sqrt(2*len(sfrchange_first_half)),  std_logsfr_change1+1.645*std_logsfr_change1/np.sqrt(2*len(sfrchange_first_half)) )


mean_logsfr_change2 = sfrchange_second_half['logsfr_change'].mean()
std_logsfr_sem2 = sfrchange_second_half['logsfr_change'].sem()
std_logsfr_change2 = sfrchange_second_half['logsfr_change'].std()

print('\nEstimate mean from 2nd half sample is (Black):', mean_logsfr_change2, '+/-', 1.645*std_logsfr_sem2, 'Range:', mean_logsfr_change2-1.645*std_logsfr_sem, mean_logsfr_change2+1.645*std_logsfr_sem2 )
print('Sample standard deviation from 2nd half is (Black):', std_logsfr_change2, '+/-', 1.645*std_logsfr_change2/np.sqrt(2*len(sfrchange_second_half)), 'Range:', std_logsfr_change2-1.645*std_logsfr_change2/np.sqrt(2*len(sfrchange_second_half)),  std_logsfr_change2+1.645*std_logsfr_change2/np.sqrt(2*len(sfrchange_second_half)) )



##############################################  WEEKLY CHANGE ANALYSIS ##############################################
############# reset things to include the days we dropped with larger than 1 calendar day change ####################
sfrchange = sfrchangeold
sfrchange = sfrchange.reset_index(drop=True)

##################### make a function for finding indices at the end of a full trading week ############################################
#### the trick is to find sequences in the data where four days in a row have changes in the calendar day ('daydelta') equal to 1 ######

def endoffulltradingweek(t):
	indextemp = sfrchange.index[sfrchange['observation_date'] == t][0]
	
	if indextemp < 3:
		return(False)
	
	indextempmin1 = indextemp - 1
	indextempmin2 = indextemp - 2
	indextempmin3 = indextemp - 3
	
	check0 = sfrchange.loc[indextemp,'daydelta']
	check1 = sfrchange.loc[indextempmin1,'daydelta']
	check2 = sfrchange.loc[indextempmin2,'daydelta']
	check3 = sfrchange.loc[indextempmin3,'daydelta']
	
	check = check0*check1*check2*check3
	
	if check == 1:
		return(True)
	else:
		return(False)

sfrchange['end_of_full_week'] = sfrchange['observation_date'].apply(endoffulltradingweek)


def weeklysfrchange(t):	
	indextemp = sfrchange.index[sfrchange['observation_date'] == t][0]
	
	if sfrchange.loc[indextemp,'end_of_full_week']:
		change = sfrchange.loc[indextemp,'sfr'] - sfrchange.loc[indextemp-4,'sfr']
		return(change*10**4)
	else:
		return(None)
		
def logweeklysfrchange(t):	
	indextemp = sfrchange.index[sfrchange['observation_date'] == t][0]
	
	if sfrchange.loc[indextemp,'end_of_full_week']:
		change = np.log(sfrchange.loc[indextemp,'sfr']) - np.log(sfrchange.loc[indextemp-4,'sfr'])
		return(change)
	else:
		return(None)	

sfrchange['weekly_sfr_change (bps)'] = sfrchange['observation_date'].apply(weeklysfrchange)
sfrchange['log_weekly_sfr_change'] = sfrchange['observation_date'].apply(logweeklysfrchange)


#### the functions above return NA if end_of_full_week is False. So we drop NA to get rid of undesired columns #####
sfrchange = sfrchange.dropna()

sfrchange = sfrchange.drop(columns=['daydelta','end_of_full_week', 'sfr', 'sfr_change', 'sfr_change (bps)', 'logsfr_change',])
sfrchange = sfrchange.reset_index(drop=True)

print('\nsfr change df after calculating weekly changes for full trading weeks and dropping unecessary columns and resetting index is:')
print(sfrchange)

########### BACHELIER ANALYSIS FOR THE WEEKLY CHANGES #################
### calculate mean etc.

mean_sfr_change = sfrchange['weekly_sfr_change (bps)'].mean()
std_sfr_sem = sfrchange['weekly_sfr_change (bps)'].sem()
std_sfr_change = sfrchange['weekly_sfr_change (bps)'].std()

print('\nEstimate mean for weakly change from sample is (bps):', mean_sfr_change, '+/-', 1.645*std_sfr_sem, 'Range:', mean_sfr_change-1.645*std_sfr_sem, mean_sfr_change+1.645*std_sfr_sem )
print('Sample standard deviation for weakly change is (bps):', std_sfr_change, '+/-', 1.645*std_sfr_change/np.sqrt(2*len(sfrchange)), 'Range:', std_sfr_change-1.645*std_sfr_change/np.sqrt(2*len(sfrchange)),  std_sfr_change+1.645*std_sfr_change/np.sqrt(2*len(sfrchange)) )


##### Various statistical tests below ################
# Perform the Lilliefors test for a normal distribution

sfrchange_array = sfrchange['weekly_sfr_change (bps)'].to_numpy()

ks_statistic, p_value = lilliefors(sfrchange_array, dist='norm', pvalmethod='approx')
print("\nLilliefors test for weekly change (Bachelier)") 
print(f"KS Statistic: {ks_statistic}")
print(f"P-value: {p_value}")


shapiro_statistic, shapiro_p_value = stats.shapiro(sfrchange_array)
print("\nShapiro-Wilk test for weekly change (Bachelier)") 
print(f"Shapiro-Wilk Statistic: {shapiro_statistic}")
print(f"P-value: {shapiro_p_value}")

# Perform the Anderson-Darling test
result = anderson(sfrchange_array)

# Print the results
print("\nAnderson-Darling test for weekly change (Bachelier)")
print(f"Test Statistic: {result.statistic}")
print("Critical Values and Significance Levels:")
for i in range(len(result.critical_values)):
    sl = result.significance_level[i]
    cv = result.critical_values[i]
    print(f"  Significance Level: {sl}% | Critical Value: {cv:.4f}")


print('\nThe volatility 90% CI range for the weekly change - (Bachelier in bps) is:', std_sfr_change-1.645*std_sfr_change/np.sqrt(2*len(sfrchange)),  std_sfr_change+1.645*std_sfr_change/np.sqrt(2*len(sfrchange)) )
print('The volatility 90% CI range range for the daily change - (Bachelier in bps) times two is:', (minrange1)*np.sqrt(4), (maxrange1)*np.sqrt(4))

######################### TEST THE BLACK SCENARIO FOR THE WEEKLY CHANGES ######################

mean_logsfr_change = sfrchange['log_weekly_sfr_change'].mean()
std_logsfr_sem = sfrchange['log_weekly_sfr_change'].sem()
std_logsfr_change = sfrchange['log_weekly_sfr_change'].std()

print('\nEstimate mean for weakly change sample is (Black):', mean_logsfr_change, '+/-', 1.645*std_logsfr_sem, 'Range:', mean_logsfr_change-1.645*std_logsfr_sem, mean_logsfr_change+1.645*std_logsfr_sem )
print('Sample standard deviation for weakly change is (Black):', std_logsfr_change, '+/-', 1.645*std_logsfr_change/np.sqrt(2*len(sfrchange)), 'Range:', std_logsfr_change-1.645*std_logsfr_change/np.sqrt(2*len(sfrchange)),  std_logsfr_change+1.645*std_logsfr_change/np.sqrt(2*len(sfrchange)) )

logsfrchange_array = sfrchange['log_weekly_sfr_change'].to_numpy()

##### Various statistical tests below ################

ks_statistic, p_value = lilliefors(logsfrchange_array, dist='norm', pvalmethod='approx')
print("\nAnderson-Darling test for weekly change (Black))")
print(f"KS Statistic: {ks_statistic}")
print(f"P-value: {p_value}")


shapiro_statistic, shapiro_p_value = stats.shapiro(logsfrchange_array)
print("\nAnderson-Darling test for weekly change (Black)")
print(f"Shapiro-Wilk Statistic: {shapiro_statistic}")
print(f"P-value: {shapiro_p_value}")


# Perform the Anderson-Darling test
result = anderson(logsfrchange_array)

# Print the results
print("\nAnderson-Darling test for weekly change (Black)")
print(f"Test Statistic: {result.statistic}")
print("Critical Values and Significance Levels:")
for i in range(len(result.critical_values)):
    sl = result.significance_level[i]
    cv = result.critical_values[i]
    print(f"  Significance Level: {sl}% | Critical Value: {cv:.4f}")

print(f'\nThe volatility 90% CI range for the weekly change - (Black) is:',  std_logsfr_change-1.645*std_logsfr_change/np.sqrt(2*len(sfrchange)),  std_logsfr_change+1.645*std_logsfr_change/np.sqrt(2*len(sfrchange)) )
print(f'The volatility 90% CI range range for the daily change (Black) times two is:', (minrange2)*np.sqrt(4), (maxrange2)*np.sqrt(4))


##### large jump z scores for the weekly change analysis ########################
print('\nLargest positive jumps for weekly change, Bachelier (z-scores)')
print((sfrchange['weekly_sfr_change (bps)'].nlargest(5)-mean_sfr_change)/std_sfr_change)

print('\nLargest negative jumps for weekly change, Bachelier (z-scores)')
print((sfrchange['weekly_sfr_change (bps)'].nsmallest(5)-mean_sfr_change)/std_sfr_change)

print('\nLargest positive jumps for weekly change, Black (z-scores)')
print((sfrchange['log_weekly_sfr_change'].nlargest(5)-mean_logsfr_change)/std_logsfr_change)

print('\nLargest negative jumps for weekly change, Black (z-scores)')
print((sfrchange['log_weekly_sfr_change'].nsmallest(5)-mean_logsfr_change)/std_logsfr_change)


