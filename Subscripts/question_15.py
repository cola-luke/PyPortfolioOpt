#############à### PREMISE ###############
## This question is entirely based off the data of most of previous points and scripts. We avoid copy-pastying the whole scripts to avoid the running of a heavy code. We suggest that the reader interested in running this script keeps in memory variables from previous points, or better yet runs it on the attached Jupyter notebook
#########################################

# Import all libraries we need
import numpy as np
import pandas as pd
import pypfopt
import matplotlib.pyplot as plt

##15. Mixed Portfolio
cw_d_l_ser=pd.Series(cw_d_l)
cw_m_l_ser=pd.Series(cw_m_l)
cw_d_s_ser=pd.Series(cw_d_s)
cw_m_s_ser=pd.Series(cw_m_s)

w_gmv_d_l_ser=pd.Series(w_gmv_d_l)
w_gmv_m_l_ser=pd.Series(w_gmv_m_l)
w_gmv_d_s_ser=pd.Series(w_gmv_d_s)
w_gmv_m_s_ser=pd.Series(w_gmv_m_s)

blcw_d_l_ser=pd.Series(blcw_d_l)
blcw_m_l_ser=pd.Series(blcw_m_l)
blcw_d_s_ser=pd.Series(blcw_d_s)
blcw_m_s_ser=pd.Series(blcw_m_s)

baycw_d_l_ser=pd.Series(baycw_d_l)
baycw_m_l_ser=pd.Series(baycw_m_l)
baycw_d_s_ser=pd.Series(baycw_d_s)
baycw_m_s_ser=pd.Series(baycw_m_s)


#Calculate all weights as average of previous weights
tot_port_columns = ['MV', 'BL', 'Bayes', 'GMV']
daily_long = pd.DataFrame([cw_d_l_ser, blcw_d_l_ser, baycw_d_l_ser, w_gmv_d_l_ser]).T
daily_long['Average of allocations'] = daily_long.mean(axis=1)
tw_d_l = daily_long['Average of allocations'].to_dict()
daily_long['Average of allocations'].to_excel('daily long weights.xlsx')

monthly_long = pd.DataFrame([cw_m_l_ser, blcw_m_l_ser, baycw_m_l_ser, w_gmv_m_l_ser]).T
monthly_long['Average of allocations'] = monthly_long.mean(axis=1)
tw_m_l = monthly_long['Average of allocations'].to_dict()
monthly_long['Average of allocations'].to_excel('monthly long weights.xlsx')

daily_short = pd.DataFrame([cw_d_s_ser, blcw_d_s_ser, baycw_d_s_ser, w_gmv_d_s_ser]).T
daily_short['Average of allocations'] = daily_short.mean(axis=1)
tw_d_s = daily_short['Average of allocations'].to_dict()
daily_short['Average of allocations'].to_excel('daily short weights.xlsx')

monthly_short = pd.DataFrame([cw_m_s_ser, blcw_m_s_ser, baycw_m_s_ser, w_gmv_m_s_ser]).T
monthly_short['Average of allocations'] = monthly_short.mean(axis=1)
tw_m_s = monthly_short['Average of allocations'].to_dict()
monthly_short['Average of allocations'].to_excel('monthly short weights.xlsx')

tef_d_l = EfficientFrontier(mu_d, S_d) #daily, constrained
tef_d_l.set_weights(tw_d_l)
tef_d_l.portfolio_performance(verbose = True)

tef_m_l = EfficientFrontier(mu_m, S_m) #monthly, constrained
tef_m_l.set_weights(tw_m_l)
tef_m_l.portfolio_performance(verbose = True)

tef_d_s = EfficientFrontier(mu_d, S_d, weight_bounds=(-1,1)) #daily, unconstrained
tef_d_s.set_weights(tw_d_s)
tef_d_s.portfolio_performance(verbose = True)

tef_m_s = EfficientFrontier(mu_m, S_m, weight_bounds=(-1,1)) #monthly, unconstrained
tef_m_s.set_weights(tw_m_s)
tef_m_s.portfolio_performance(verbose = True)

tot_d_l_ret = portfolio_returns(select_ret_d, tw_d_l, 'Daily Long Tot')
stats_tot_d_l = descriptive_stats(tot_d_l_ret)

tot_m_l_ret = portfolio_returns(select_ret_m, tw_m_l, 'Monthly Long Tot')
stats_tot_m_l = descriptive_stats(tot_m_l_ret)

tot_d_s_ret = portfolio_returns(select_ret_d, tw_d_s, 'Daily Short Tot')
stats_tot_d_s = descriptive_stats(tot_d_s_ret)

tot_m_s_ret = portfolio_returns(select_ret_m, tw_m_s, 'Monthly Short Tot')
stats_tot_m_s = descriptive_stats(tot_m_s_ret)

tot_stats = pd.concat([stats_tot_d_l, stats_tot_m_l, stats_tot_d_s, stats_tot_m_s], axis=0)
tot_stats.to_excel('mixed portfolio stats.xlsx')