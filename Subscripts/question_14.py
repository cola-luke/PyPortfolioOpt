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

from pypfopt.expected_returns import mean_historical_return
mu_d = mean_historical_return(select_d) #daily expected returns
S_d = pypfopt.risk_models.sample_cov(select_d) #daily vcv matr

mu_m = mean_historical_return(select_m, frequency=12) #monthly expected returns
S_m = pypfopt.risk_models.sample_cov(select_m, frequency=12) #monthly vcv matr

##14. Global Min Variance portfolio and statistics

ef_gmv_d_l = EfficientFrontier(mu_d, S_d,weight_bounds=(0,1)) #daily, constrained
w_gmv_d_l = ef_gmv_d_l.min_volatility()
ef_gmv_d_l.save_weights_to_file("w_gmv_d_l.csv")
print(w_gmv_d_l)
ef_gmv_d_l.portfolio_performance(verbose = True)

ef_gmv_m_l = EfficientFrontier(mu_m, S_m,weight_bounds=(0,1)) #monthly, constrained
w_gmv_m_l = ef_gmv_m_l.min_volatility()
ef_gmv_m_l.save_weights_to_file("w_gmv_m_l.csv")
print(w_gmv_m_l)
ef_gmv_m_l.portfolio_performance(verbose = True)

ef_gmv_d_s = EfficientFrontier(mu_d, S_d,weight_bounds=(-1,1)) #daily, unconstrained
w_gmv_d_s = ef_gmv_d_s.min_volatility()
ef_gmv_d_s.save_weights_to_file("w_gmv_d_s.csv")
print(w_gmv_d_s)
ef_gmv_d_s.portfolio_performance(verbose = True)

ef_gmv_m_s = EfficientFrontier(mu_m, S_m,weight_bounds=(-1,1)) #monthly, unconstrained
w_gmv_m_s = ef_gmv_m_s.min_volatility()
ef_gmv_m_s.save_weights_to_file("w_gmv_m_s.csv")
print(w_gmv_m_s)
ef_gmv_m_s.portfolio_performance(verbose = True)


gmv_d_l_ret = portfolio_returns(select_ret_d, w_gmv_d_l, 'Daily Long GMV')
stats_gmv_port_d_l = descriptive_stats(gmv_d_l_ret)

gmv_m_l_ret = portfolio_returns(select_ret_m, w_gmv_m_l, 'Monthly Long GMV')
stats_gmv_port_m_l = descriptive_stats(gmv_m_l_ret)

gmv_d_s_ret = portfolio_returns(select_ret_d, w_gmv_d_s, 'Daily Short GMV')
stats_gmv_port_d_s = descriptive_stats(gmv_d_s_ret)

gmv_m_s_ret = portfolio_returns(select_ret_m, w_gmv_m_s, 'Monthly Short GMV')
stats_gmv_port_m_s = descriptive_stats(gmv_m_s_ret)

gmv_stats = pd.concat([stats_gmv_port_d_l, stats_gmv_port_m_l, stats_gmv_port_d_s, stats_gmv_port_m_s], axis=0)
gmv_stats.to_excel('gmv portfolio stats.xlsx')