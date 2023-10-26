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

##9. Compute all the statistics relative to that index [FTSE Italia All Market] 
##   (mean, standard deviation, variance, kurtosis and skewness)
##Import and cleanup of index dataset
raw_index_d = pd.read_excel("data for exam 2023.xlsx", sheet_name="ftse italia all share daily")
raw_index_m = pd.read_excel("data for exam 2023.xlsx", sheet_name="ftse italia all share monthly")
index_prices_d = clean_dataframe(raw_index_d)
index_prices_m = clean_dataframe(raw_index_m)
index_returns_d = index_prices_d.pct_change(1)
index_returns_m = index_prices_m.pct_change(1)
stats_index_d = descriptive_stats(index_returns_d)
stats_index_m = descriptive_stats(index_returns_m)

stats_index_tot = pd.concat([stats_index_d, stats_index_m], axis=0)
stats_index_tot.to_excel('index statistics.xlsx')