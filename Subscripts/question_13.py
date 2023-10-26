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

##13. Pure Bayesian
## Building prior arrays
mu_arr_d = mu_d.to_numpy()
S_arr_d = S_d.to_numpy()
mu_pr_d = mu_arr_d + np.sqrt(np.diag(S_arr_d)) #daily mean prior
S_pr_d = S_arr_d * 2 #daily vcv prior

mu_arr_m = mu_m.to_numpy()
S_arr_m = S_m.to_numpy()
mu_pr_m = mu_arr_m + np.sqrt(np.diag(S_arr_m)) #monthly mean prior
S_pr_m = S_arr_m * 2 #monthly vcv prior

#Constructing predictive parameters
T=len(mu_d)
num_mu_po_d = np.matmul(T*np.linalg.inv(S_arr_d), mu_arr_d) + np.matmul(np.linalg.inv(S_pr_d), mu_pr_d)
sigma_po_d = np.linalg.inv(T*np.linalg.inv(S_arr_d) + np.linalg.inv(S_pr_d))
mu_po_d = np.matmul(sigma_po_d, num_mu_po_d) #daily predictive mean
sigma_pred_d = sigma_po_d + S_arr_d #daily predictive vcv

sigma_pred_d_df = pd.DataFrame(sigma_pred_d, index = S_d.index, columns=S_d.columns)
mu_po_d_df = pd.Series(mu_po_d, index=mu_d.index)

sigma_pred_d_df.to_excel('variance covariance predictive.xlsx', sheet_name='daily')
mu_po_d_df.to_excel('mean predictive.xlsx', sheet_name='daily')

num_m_po_m = np.matmul(T*np.linalg.inv(S_arr_m), mu_arr_m) + np.matmul(np.linalg.inv(S_pr_m), mu_pr_m)
sigma_po_m = np.linalg.inv(T*np.linalg.inv(S_arr_m) + np.linalg.inv(S_pr_m))
mu_po_m = np.matmul(sigma_po_m, num_m_po_m) #monthly predictive mean
sigma_pred_m = sigma_po_m + S_arr_m #monthly predictive vcv

sigma_pred_m_df = pd.DataFrame(sigma_pred_m, index = S_m.index, columns=S_m.columns)
mu_po_m_df = pd.Series(mu_po_m, index=mu_m.index)

sigma_pred_m_df.to_excel('variance covariance predictive.xlsx', sheet_name='monthly')
mu_po_m_df.to_excel('mean predictive.xlsx', sheet_name= 'monthly')

#Calculating efficient portfolios
efbay_d_l = EfficientFrontier(mu_po_d_df, sigma_pred_d_df) #daily, constrained
bayw_d_l = efbay_d_l.max_sharpe()
baycw_d_l = efbay_d_l.clean_weights()
efbay_d_l.save_weights_to_file("efbay_d_l.csv")
print(baycw_d_l)
efbay_d_l.portfolio_performance(verbose = True)

efbay_m_l = EfficientFrontier(mu_po_m_df, sigma_pred_m_df) #monthly, constrained
bayw_m_l = efbay_m_l.max_sharpe()
baycw_m_l = efbay_m_l.clean_weights()
efbay_m_l.save_weights_to_file("efbay_m_l.csv")
print(baycw_m_l)
efbay_m_l.portfolio_performance(verbose = True)

efbay_d_s = EfficientFrontier(mu_po_d_df, sigma_pred_d_df, weight_bounds=(-1,1)) #daily, unconstrained
bayw_d_s = efbay_d_s.max_sharpe()
baycw_d_s = efbay_d_s.clean_weights()
efbay_d_s.save_weights_to_file("efbay_d_s.csv")
print(baycw_d_s)
efbay_d_s.portfolio_performance(verbose = True)

efbay_m_s = EfficientFrontier(mu_po_m_df, sigma_pred_m_df, weight_bounds=(-1,1)) #monthly, unconstrained
bayw_m_s = efbay_m_s.max_sharpe()
baycw_m_s = efbay_m_s.clean_weights()
efbay_m_s.save_weights_to_file("efbay_m_s.csv")
print(baycw_m_s)
efbay_m_s.portfolio_performance(verbose = True)

bay_d_l_ret = portfolio_returns(select_ret_d, baycw_d_l, 'Daily Long Bay')
stats_bay_port_d_l = descriptive_stats(bay_d_l_ret)

bay_m_l_ret = portfolio_returns(select_ret_m, baycw_m_l, 'Monthly Long Bay')
stats_bay_port_m_l = descriptive_stats(bay_m_l_ret)

bay_d_s_ret = portfolio_returns(select_ret_d, baycw_d_s, 'Daily Short Bay')
stats_bay_port_d_s = descriptive_stats(bay_d_s_ret)

bay_m_s_ret = portfolio_returns(select_ret_m, baycw_m_s, 'Monthly Short Bay')
stats_bay_port_m_s = descriptive_stats(bay_m_s_ret)


bay_stats = pd.concat([stats_bay_port_d_l, stats_bay_port_m_l, stats_bay_port_d_s, stats_bay_port_m_s], axis=0)
bay_stats.to_excel('bayesian portfolio stats.xlsx')