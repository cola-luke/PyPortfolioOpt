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

##2. Compute the variance-covariance matrix and the correlation matrix
cov_mat = pypfopt.risk_models.sample_cov(prices_d)
corr_mat = pypfopt.risk_models.cov_to_corr(cov_mat)
cov_mat.to_excel('covariance matrix.xlsx')
corr_mat.to_excel('correlation matrix.xlsx')

## To filter correlations we devised:
exclusion = ['I:PIRL','I:ILLB','I:SPAC','I:EQUI','I:IG','I:ENAV','I:PST','I:GAMB','I:AST','I:DEA','I:BIM','I:BOR','I:CASS'] #stocks with nan or delisted

## Find pairs that have correlation below a certain threshold
threshold = 0.07 
for i in corr_mat.columns:
    for j in corr_mat.columns:
        if corr_mat[i][j] < threshold and i not in exclusion and j not in exclusion: #criteria that make the pair feasible to us
            print('Pair '+i+' '+j+' has correlation '+str(corr_mat[i][j]))
        else:
            continue

