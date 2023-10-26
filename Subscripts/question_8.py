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

##8. Plot the efficient frontier for both daily and monthly frequency.

## The efficient frontier portfolio source code has been taken from the user guide of PyPortfolioOpt.
## Daily, constrained
from pypfopt import plotting
ef = EfficientFrontier(mu_d, S_d, weight_bounds=(0, 1))
fig, ax = plt.subplots()
ef_max_sharpe = ef.deepcopy()
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)

# Find the tangency portfolio
ef_max_sharpe.max_sharpe()
ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

# Generate random portfolios
n_samples = 10000
w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
rets = w.dot(ef.expected_returns)
stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
sharpes = rets / stds
ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

# Output
ax.set_title("Daily Constrained Efficient Frontier")
ax.legend()
plt.tight_layout()
plt.savefig("ef_scatter.png", dpi=200)
plt.show()

###############################

## Monhtly, constrained
ef1 = EfficientFrontier(mu_m, S_m, weight_bounds=(0, 1))
fig, ax = plt.subplots()
ef_max_sharpe_mo = ef1.deepcopy()
plotting.plot_efficient_frontier(ef1, ax=ax, show_assets=True)

# Find the tangency portfolio
ef_max_sharpe_mo.max_sharpe()
ret_tangent_mo, std_tangent_mo, _ = ef_max_sharpe_mo.portfolio_performance()
ax.scatter(std_tangent_mo, ret_tangent_mo, marker="*", s=100, c="r", label="Max Sharpe")

# Generate random portfolios
n_samples = 10000
w_mo = np.random.dirichlet(np.ones(ef1.n_assets), n_samples)
rets_mo = w.dot(ef1.expected_returns)
stds_mo = np.sqrt(np.diag(w @ ef1.cov_matrix @ w.T))
sharpes_mo = rets_mo / stds_mo
ax.scatter(stds_mo, rets_mo, marker=".", c=sharpes_mo, cmap="viridis_r")

# Output
ax.set_title("Monthly Constrained Efficient Frontier")
ax.legend()
plt.tight_layout()
plt.savefig("ef_scatter.png", dpi=200)
plt.show()

###############################

## Daily, unconstrained
ef2 = EfficientFrontier(mu_d, S_d, weight_bounds=(-1, 1))
fig, ax = plt.subplots()
ef2_max_sharpe = ef2.deepcopy()
plotting.plot_efficient_frontier(ef2, ax=ax, show_assets=True)

# Find the tangency portfolio
ef2_max_sharpe.max_sharpe()
ret2_tangent, std2_tangent, _ = ef2_max_sharpe.portfolio_performance()
ax.scatter(std2_tangent, ret2_tangent, marker="*", s=100, c="r", label="Max Sharpe")

# Generate random portfolios
n_samples = 10000
w2 = np.random.dirichlet(np.ones(ef2.n_assets), n_samples)
rets2 = w.dot(ef2.expected_returns)
stds2 = np.sqrt(np.diag(w @ ef2.cov_matrix @ w.T))
sharpes2 = rets2 / stds2
ax.scatter(stds2, rets2, marker=".", c=sharpes2, cmap="viridis_r")

# Output
ax.set_title("Daily Unconstrained Efficient Frontier")
ax.legend()
plt.tight_layout()
plt.savefig("ef_scatter.png", dpi=200)
plt.show()

###############################

## Monthly, unconstrained
ef3 = EfficientFrontier(mu_m, S_m, weight_bounds=(-1, 1))
fig, ax = plt.subplots()
ef3_max_sharpe = ef3.deepcopy()
plotting.plot_efficient_frontier(ef3, ax=ax, show_assets=True)

# Find the tangency portfolio
ef3_max_sharpe.max_sharpe()
ret3_tangent, std3_tangent, _ = ef3_max_sharpe.portfolio_performance()
ax.scatter(std3_tangent, ret3_tangent, marker="*", s=100, c="r", label="Max Sharpe")

# Generate random portfolios
n_samples = 10000
w3 = np.random.dirichlet(np.ones(ef3.n_assets), n_samples)
rets3 = w.dot(ef3.expected_returns)
stds3 = np.sqrt(np.diag(w @ ef3.cov_matrix @ w.T))
sharpes3 = rets3 / stds3
ax.scatter(stds3, rets3, marker=".", c=sharpes3, cmap="viridis_r")

# Output
ax.set_title("Monthly Unconstrained Efficient Frontier")
ax.legend()
plt.tight_layout()
plt.savefig("ef_scatter.png", dpi=200)
plt.show()