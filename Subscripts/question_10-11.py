# Import all libraries we need
import numpy as np
import pandas as pd
import pypfopt
import matplotlib.pyplot as plt

## In this block we define functions we call repeatedly later
raw_prices_d= pd.read_excel("data for exam 2023.xlsx", sheet_name="stocks daily")

## This function takes as input the dataframe as provided by the exam data and outputs a dataframe with stock tickers as column
## column labels and timestamps as row labels, as required by PyPfOpt
def clean_dataframe(df):
    df.rename(columns={np.nan:'Date'}, inplace=True)
    df.iloc[0,0] = 'Date'
    df.columns = df.iloc[0]
    df.drop([0,1], inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.astype(float)
    return df

## Takes a dataframe as input and returns a dataframe with the descriptive statistics (mean, var, std, skew, kurt)
def descriptive_stats(df):
    means = df.mean(axis=0)
    stds = df.std(axis=0)
    variance = df.var(axis=0)
    skew = df.skew(axis=0)
    kurt = df.kurt(axis=0)
    stats = {'Mean' : means, 'Standard Deviation': stds, 'Variance': variance, 'Skewness':skew, 'Kurtosis':kurt}
    stats_df = pd.DataFrame(data=stats)
    return stats_df

#Given a dataframe of returns and weights of stocks in the dataframe, it returns the weighted average (which is to say the
#portfolio returns for that specific allocation.
#port has to be a string, it's the name of the portfolio
def portfolio_returns(df, weights, port):
    df[port+' Returns'] = pd.Series(0,index=df.index,dtype='float64') 
    for i in weights.keys():
        df[port+' Returns']+= df[i]*weights.get(i)
    return pd.DataFrame(df[port+' Returns'])

#Load daily Italian stocks data
#The index should consist of dates or timestamps, and each column should represent the time series of prices for an asset
raw_prices_d= pd.read_excel("data for exam 2023.xlsx", sheet_name="stocks daily")
prices_d = clean_dataframe(raw_prices_d) #_d from now on indicates daily

raw_prices_m = pd.read_excel("data for exam 2023.xlsx", sheet_name="stocks monthly")
prices_m = clean_dataframe(raw_prices_m) #_m from now on indicates monthly

##Constructing the dataframe with daily and monthly returns to do some descriptive statistics on that
pct_returns_d = prices_d.pct_change(1)
pct_returns_m = prices_m.pct_change(1)

##Select a sample made of 10-12 securities.
stocks = ['I:ECK', 'I:CLT', 'I:ARN', 'I:HER', 'I:ELN', 'I:AMP', 'I:VIN', 'I:SSL', 'I:ITM', 'I:EDNR', 'I:FUL', 'I:MON']
select_d = prices_d.filter(stocks, axis=1)
select_m = prices_m.filter(stocks, axis=1)

## Given that data of question 7 follow from 5 and 6, we have grouped them together in this block.

##5-6. Mean variance optimal portfolio allocation for sample chosen
from pypfopt.expected_returns import mean_historical_return
mu_d = mean_historical_return(select_d) #daily expected returns
S_d = pypfopt.risk_models.sample_cov(select_d) #daily vcv matr

mu_m = mean_historical_return(select_m, frequency=12) #monthly expected returns
S_m = pypfopt.risk_models.sample_cov(select_m, frequency=12) #monthly vcv matr


from pypfopt.efficient_frontier import EfficientFrontier

ef_d_l = EfficientFrontier(mu_d, S_d) #efficient frontier, daily, constrained
w_d_l = ef_d_l.max_sharpe()
cw_d_l = ef_d_l.clean_weights()
ef_d_l.save_weights_to_file("ef_d_l.csv")
print(cw_d_l)
exp_ret_d_l, exp_vol_d_l, sr_d_l = ef_d_l.portfolio_performance(verbose = True)

ef_m_l = EfficientFrontier(mu_m, S_m) #efficient frontier, monthly, constrained
w_m_l = ef_m_l.max_sharpe()
cw_m_l = ef_m_l.clean_weights()
ef_m_l.save_weights_to_file("ef_m_l.csv")
print(cw_m_l)
exp_ret_m_l, exp_vol_m_l, sr_m_l = ef_m_l.portfolio_performance(verbose = True)

ef_d_s = EfficientFrontier(mu_d, S_d, weight_bounds=(-1,1)) #efficient frontier, daily, unconstrained
w_d_s = ef_d_s.max_sharpe() 
cw_d_s = ef_d_s.clean_weights()
ef_d_s.save_weights_to_file("ef_d_s.csv")
print(cw_d_s)
exp_ret_d_s, exp_vol_d_s, sr_d_s = ef_d_s.portfolio_performance(verbose = True)

ef_m_s = EfficientFrontier(mu_m, S_m, weight_bounds=(-1,1)) #efficient frontier, monthly, unconstrained
w_m_s = ef_m_s.max_sharpe()
cw_m_s = ef_m_s.clean_weights()
ef_m_s.save_weights_to_file("ef_m_s.csv")
print(cw_m_s)
exp_ret_m_s, exp_vol_m_s, sr_m_s = ef_m_s.portfolio_performance(verbose = True)

##Import and cleanup of index dataset
raw_index_d = pd.read_excel("data for exam 2023.xlsx", sheet_name="ftse italia all share daily")
raw_index_m = pd.read_excel("data for exam 2023.xlsx", sheet_name="ftse italia all share monthly")
index_prices_d = clean_dataframe(raw_index_d)
index_prices_m = clean_dataframe(raw_index_m)
index_returns_d = index_prices_d.pct_change(1)
index_returns_m = index_prices_m.pct_change(1)

stats_index_tot = pd.concat([stats_index_d, stats_index_m], axis=0)
stats_index_tot.to_excel('index statistics.xlsx')

##10. Compute the beta for each security included in your portfolio and the beta for your portfolio as well.

index_var_d = index_returns_d['FITASHE(RI)'].var() #index variance, daily
index_var_m = index_returns_m['FITASHE'].var() #index variance, monthly


beta_select = {} #daily betas
for i in w_m_s.keys():
        if w_d_s.get(i) != 0: #since we want those securities in the portfolio, we impose to compute for non zero weights
            beta= (select_ret_d[i].cov(index_returns_d['FITASHE(RI)'])/index_var_d)
            beta_select[i] = beta
            print('Daily '+i+' '+str(beta))
        else:
            continue

beta_select_m = {} #monthly betas
for i in w_m_s.keys():
        if w_m_s.get(i) != 0:
            beta= (select_ret_m[i].cov(index_returns_m['FITASHE'])/index_var_m)
            beta_select_m[i] = beta
            print('Monthly '+i+' '+str(beta))
        else:
            continue            
            
beta_d_l = (d_l_returns['Daily Long Portfolio Returns'].cov(index_returns_d['FITASHE(RI)'])/index_var_d)
beta_m_l = (m_l_returns['Monthly Long Portfolio Returns'].cov(index_returns_m['FITASHE'])/index_var_m)
beta_d_s = (d_s_returns['Daily Short Portfolio Returns'].cov(index_returns_d['FITASHE(RI)'])/index_var_d)
beta_m_s = (m_s_returns['Monthly Short Portfolio Returns'].cov(index_returns_m['FITASHE'])/index_var_m)


print('Daily constrained '+str(beta_d_l))
print('Monthly constrained '+str(beta_m_l))
print('Daily unconstrained '+str(beta_d_s))
print('Monthly unconstrained '+str(beta_m_s))

beta_select = pd.DataFrame(beta_select, index=['Beta'])
beta_ports = {'Daily Long':beta_d_l, 'Monthly Long':beta_m_l, 'Daily Short':beta_d_s, 'Monthly Short':beta_m_s}
beta_ports = pd.DataFrame(beta_ports, index=['Beta'])
all_betas = pd.concat([beta_select, beta_ports], axis=1)

##11. Given a return for a Risk-Free security equal to 0.3 per cent (0.003), compute the Security Market Line
## Calculate annualized expected returns to plot the SML

exp_ret_d = pypfopt.expected_returns.mean_historical_return(select_d)
exp_ret_m = pypfopt.expected_returns.mean_historical_return(select_m, frequency=12)
rm_d = pypfopt.expected_returns.mean_historical_return(index_prices_d['FITASHE(RI)'])
rm_m = pypfopt.expected_returns.mean_historical_return(index_prices_m['FITASHE'], frequency = 12)    
rf = 0.003 #risk free rate


def SML(rf,rm): #daily SML
    betas = [x/10 for x in range(21)]
    assetReturns = [rf+(rm-rf)*x for x in betas]
    plt.plot(betas,assetReturns)
    plt.xlabel("Asset Beta")
    plt.ylabel("Asset Return")
    plt.title("Security Market Line Daily")
    plt.plot(1,rm,"ro")
    plt.plot(all_betas['I:ITM']['Beta'], exp_ret_d['I:ITM'] ,".")
    plt.text(all_betas['I:ITM']['Beta'], exp_ret_d['I:ITM'] , 'I:ITM')
    plt.plot(all_betas['I:AMP']['Beta'], exp_ret_d['I:AMP'] ,".")
    plt.text(all_betas['I:AMP']['Beta'], exp_ret_d['I:AMP'] , 'I:AMP')    
    plt.plot(beta_ports['Daily Long']['Beta'], exp_ret_d_l, "v")
    plt.text(beta_ports['Daily Long']['Beta'], exp_ret_d_l, "D Constr")
    plt.plot(beta_ports['Daily Short']['Beta'], exp_ret_d_s, "v")
    plt.text(beta_ports['Daily Short']['Beta'], exp_ret_d_s, "D Unconstr")


SML(rf,rm_d)

def SML(rf,rm): #monthly SML
    betas = [x/10 for x in range(21)]
    assetReturns = [rf+(rm-rf)*x for x in betas]
    plt.plot(betas,assetReturns)
    plt.xlabel("Asset Beta")
    plt.ylabel("Asset Return")
    plt.title("Security Market Line Monthly")
    plt.plot(1,rm,"ro")
    plt.plot(all_betas['I:ITM']['Beta'], exp_ret_m['I:ITM'] ,".")
    plt.text(all_betas['I:ITM']['Beta'], exp_ret_m['I:ITM'] , 'I:ITM')
    plt.plot(all_betas['I:AMP']['Beta'], exp_ret_m['I:AMP'] ,".")
    plt.text(all_betas['I:AMP']['Beta'], exp_ret_m['I:AMP'] , 'I:AMP')
    plt.plot(beta_ports['Monthly Long']['Beta'], exp_ret_m_l, "v")
    plt.text(beta_ports['Monthly Long']['Beta'], exp_ret_m_l, "M Constr")
    plt.plot(beta_ports['Monthly Short']['Beta'], exp_ret_m_s, "v")
    plt.text(beta_ports['Monthly Short']['Beta'], exp_ret_m_s, "M Unconstr")

SML(rf,rm_m)        
