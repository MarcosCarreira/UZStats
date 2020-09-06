# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # The Robert and Rosenbaum Uncertainty Zones model

# %% [markdown]
# ## Implementation by 
# ## Marcos Costa Santos Carreira
# ## École Polytechnique - CMAP
# ## Dec-2019

# %% [markdown]
# ## Import packages

# %% Python imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial
import scipy.stats as st
import seaborn as sns
from scipy import optimize

# %% Ruptures Import
# import ruptures as rpt

# %% Datetime import
# import datetime as dtt
import timeit

# %% UZ import
import uz as uz

# %%
#import cProfile
# useful to check bottlenecks
# But the biggest improvement is to parallelize \
#     the Monte Carlo simulations

# %% Pandas setup
pd.set_option('display.max_columns', 80)

# %% [markdown]
# ## File paths

# %% File paths
# pathdata='/Users/marcoscscarreira/Documents/X/CME project/data/'
# pathdfs='/Users/marcoscscarreira/Documents/X/CME project/dfs/'
pathdfs='/Users/marcoscscarreira/My Papers/UZModelUncertainty/dfs/'
# useful to store the raw paths, the processed paths and the statistics

# %% [markdown]
# ## Simulations

# %% [markdown]
# ### Parameters

# %%
# npaths = 50 # should be enough?
# npaths = 2
# Spot = 100.
# tm = 1. # one trading day
# dt = 0.005 # step 0.005s=5ms
# hours = 8 # a trading day for stocks
# dtosec = np.sqrt(hours*3600)
# nrsteps = int(np.round((dtosec**2)/dt))
# tmrst = uz.atrange(nrsteps,tm)

# %% Constants
npaths = 1
s = 100.
tm = 1. # one trading day
dt = 0.005 # step 0.005s=5ms
nrsteps = 2**23 # 2**23
hours = dt*(nrsteps)/3600 # a trading day for stocks
dtosec = np.sqrt(hours*3600)
tmrst = uz.atrange(nrsteps, tm)
hours = 9
tmrstsec = tmrst * hours * 3600

# %% Parameters
# vollist   = [0.00125, 0.0025, 0.005, 0.01, 0.02] # For the trading hours
# alphalist = [0.005, 0.01, 0.02, 0.04]
# etalist   = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
# mulist = [-0.03, -0.02, -0.01, -0.005, 0.0, +0.005, +0.01, +0.02, +0.03]
# spotlist = [97., 98., 99., 99.5, 99.75, 100., 100.25, 100.5, 101., 102., 103.]

# %% [markdown]
# The durations can be estimated by the formula:
#
# $2\cdot\eta\cdot\left(\frac{\alpha}{S}\cdot\frac{1}{\sigma}\right)^{2}$

# %%
# durlist = np.array([[v, a, e, uz.dur(Spot, a, e, v)] \
#                        for e in etalist \
#                        for a in alphalist \
#                        for v in vollist])

# %%
# durdf=pd.DataFrame(durlist,columns=['vol','alpha','eta','dur'])
# durdf['dursec']=durdf['dur']*dtosec**2

# %% [markdown]
# We want to make sure our simulation has steps small enough to capture
# the durations for $\eta$=0.1

# %%
# durdf[durdf['eta']==0.1].set_index(['vol','alpha','eta']).sort_index()

# %%
# durdf.set_index(['vol','alpha','eta']).sort_index()

# %% [markdown]
# ### Raw paths - Generation

# %% Seed
seed = 42

# %% Random numbers generation
rndns = uz.frndn(nrsteps, seed)

# %% Choices for trend and vol

σ = 0.01
μ = 0.
σ2 = 0.02
μ2 = 0.05
σ3 = 0.05
μ3 = 0.0004*(9*60)
v = np.full(nrsteps, σ)
mu = np.full(nrsteps, μ)
v1 = np.concatenate((np.full(nrsteps//2, σ), np.full(nrsteps//2, σ2)))
mu1 = np.concatenate((np.full(nrsteps//2, μ), np.full(nrsteps//2, μ2)))
v2 = np.concatenate((np.full(nrsteps//2-500000, σ),
                     np.full(1000000, σ3),
                     np.full(nrsteps//2-500000, σ)))
mu2 = np.concatenate((np.full(nrsteps//2-500000, μ),
                      np.full(1000000, μ3),
                      np.full(nrsteps//2-500000, μ)))

# %% Plot effective paths
# pd.Series(v, index=tmrst[:-1]).plot(figsize=(9, 6), color='k');
# pd.Series(v1, index=tmrst[:-1]).plot(figsize=(9, 6), color='g');
# pd.Series(v2, index=tmrst[:-1]).plot(figsize=(9, 6), color='b');

# %% Plot effective paths
# pd.Series(mu, index=tmrst[:-1]).plot(figsize=(9, 6), color='k');
# pd.Series(mu1, index=tmrst[:-1]).plot(figsize=(9, 6), color='g');
# pd.Series(mu2, index=tmrst[:-1]).plot(figsize=(9, 6), color='b');

# %%
# tmrst = uz.atrange(nrsteps, tm)
# tmrst = uz.atrange(nrsteps, tm)

# %% Effective paths generation
path0 = uz.MCPath(rndns, v, s, tm, mu)  # Black, trend 0 vol 0.01
path1 = uz.MCPath(rndns, v1, s, tm, mu)  # Green, vol from 0.01 to 0.02
path2 = uz.MCPath(rndns, v, s, tm, mu1)  # Blue, trend from 0 to 0.05
path3 = uz.MCPath(rndns, v1, s, tm, mu1)  # Magenta, both changes
path4 = uz.MCPath(rndns, v, s, tm, mu2)  # Red, burst
path5 = uz.MCPath(rndns, v2, s, tm, mu2)  # Yellow, burst with vol

# %% Plot effective paths
# pd.Series(path0, index=tmrst).plot(figsize=(9, 6), color='k');
# pd.Series(path1, index=tmrst).plot(figsize=(9, 6), color='g');
# pd.Series(path2, index=tmrst).plot(figsize=(9, 6), color='b');
# pd.Series(path3, index=tmrst).plot(figsize=(9, 6), color='m');
# pd.Series(path4, index=tmrst).plot(figsize=(9, 6), color='r');
# pd.Series(path5, index=tmrst).plot(figsize=(9, 6), color='y');

# %% signal

# signal = np.transpose(np.array([path0, path1, path2, path3]))

# %% Rupture model

# algo = rpt.Pelt(model='l2', min_size=100).fit(signal)
# my_bkps = algo.predict(pen=10)

# %% Display Pelt

# rpt.display(signal, my_bkps)

# %% Traded paths

α = 0.01
η = 0.3
trpath0 = uz.trprpath(α, η, path0)
trpath1 = uz.trprpath(α, η, path1)
trpath2 = uz.trprpath(α, η, path2)
trpath3 = uz.trprpath(α, η, path3)
trpath4 = uz.trprpath(α, η, path4)
trpath5 = uz.trprpath(α, η, path5)

# %% Plot traded paths
# pd.Series(trpath0, index=tmrst).plot(figsize=(9, 6), color='k');
# pd.Series(trpath1, index=tmrst).plot(figsize=(9, 6), color='g');
# pd.Series(trpath2, index=tmrst).plot(figsize=(9, 6), color='b');
# pd.Series(trpath3, index=tmrst).plot(figsize=(9, 6), color='m');
# pd.Series(trpath4, index=tmrst).plot(figsize=(9, 6), color='r');
# pd.Series(trpath5, index=tmrst).plot(figsize=(9, 6), color='y');

# %% signal

# signal_tr = np.transpose(np.array([trpath0, trpath1, trpath2, trpath3]))

# %% Rupture model

# algo_tr = rpt.Pelt(model='l2', min_size=100).fit(signal_tr)
# my_bkps_tr = algo_tr.predict(pen=3)

# %% Display Pelt

# rpt.display(signal_tr, my_bkps_tr)

# %% Reduce to price changes

pxchg0 = uz.diff_prices_df_nogroup(pd.Series(trpath0, index=tmrstsec), α)
pxchg1 = uz.diff_prices_df_nogroup(pd.Series(trpath1, index=tmrstsec), α)
pxchg2 = uz.diff_prices_df_nogroup(pd.Series(trpath2, index=tmrstsec), α)
pxchg3 = uz.diff_prices_df_nogroup(pd.Series(trpath3, index=tmrstsec), α)
pxchg4 = uz.diff_prices_df_nogroup(pd.Series(trpath4, index=tmrstsec), α)
pxchg5 = uz.diff_prices_df_nogroup(pd.Series(trpath5, index=tmrstsec), α)

# %%
uz.distance_dur(pxchg0, s, α, η, σ, μ, hours)

# %%
hour_range = range(1, hours + 1)
min_range = range(10, hours*60 + 10, 10)

# %%
# start = timeit.default_timer()
# res0de = uz.minde_dist(pxchg0, s, α, hours)
# stop = timeit.default_timer()
# print('Time Spent: ', round(stop - start), ' seconds')
# df0_res = pd.DataFrame([res0de], index=[0],
#                        columns=['H', 'η', 'σ', 'σXe', 'σP', 'μ', 'μmax', 'distance'])
# print(df0_res)

# %%
# start = timeit.default_timer()
# res0de = uz.minde_dist(pxchg0, s, α, hours, 0.01)
# stop = timeit.default_timer()
# print('Time Spent: ', round(stop - start), ' seconds')
# df0_res = pd.DataFrame([res0de], index=[0],
#                        columns=['H', 'η', 'σ', 'σXe', 'σP', 'μ', 'μmax', 'distance'])
# print(df0_res)

# %%
start = timeit.default_timer()
res0dmf = uz.multfit_dist(pxchg0, s, α, hours, 'All', 0)
stop = timeit.default_timer()
print('Time Spent: ', round(stop - start), ' seconds')

# %%
res0dmf.T

# %%
start = timeit.default_timer()
res0_tbl = [uz.multfit_dist(pxchg0.loc[:h*3600], s, α, hours, h)
             for h in hour_range]
stop = timeit.default_timer()
print('Time Spent: ', round(stop - start), ' seconds')

# %%
df0_rest = pd.concat(res0_tbl, axis=0)

# %%
df0_rest.T

# %%
start = timeit.default_timer()
res0_rtbl = [uz.multfit_dist(pxchg0.loc[(h-1)*3600:h*3600], s, α, hours, h)
             for h in hour_range]
stop = timeit.default_timer()
print('Time Spent: ', round(stop - start), ' seconds')
df0_rrest = pd.concat(res0_rtbl, axis=0)

# %%
df0_rrest.T

# %%
start = timeit.default_timer()
res1_rtbl = [uz.multfit_dist(pxchg1.loc[(h-1)*3600:h*3600], s, α, hours, h)
             for h in hour_range]
stop = timeit.default_timer()
print('Time Spent: ', round(stop - start), ' seconds')
df1_rrest = pd.concat(res1_rtbl, axis=0)

# %%
df1_rrest.T

# %%
start = timeit.default_timer()
res2_rtbl = [uz.multfit_dist(pxchg2.loc[(h-1)*3600:h*3600], s, α, hours, h)
             for h in hour_range]
stop = timeit.default_timer()
print('Time Spent: ', round(stop - start), ' seconds')
df2_rrest = pd.concat(res2_rtbl, axis=0)

# %%
df2_rrest.T

# %%
start = timeit.default_timer()
res4_rtbl = [uz.multfit_dist(pxchg4.loc[(h-1)*3600:h*3600], s, α, hours, h)
             for h in hour_range]
stop = timeit.default_timer()
print('Time Spent: ', round(stop - start), ' seconds')
df4_rrest = pd.concat(res4_rtbl, axis=0)

# %%
df4_rrest.T

# %%
start = timeit.default_timer()
res5_rtbl = [uz.multfit_dist(pxchg5.loc[(h-1)*3600:h*3600], s, α, hours, h)
             for h in hour_range]
stop = timeit.default_timer()
print('Time Spent: ', round(stop - start), ' seconds')
df5_rrest = pd.concat(res4_rtbl, axis=0)

# %%
df5_rrest.T

# %%
df0_rest['H'].plot(color='c');
df0_rrest['H'].plot(color='k');
df1_rrest['H'].plot(color='g');
df2_rrest['H'].plot(color='b');
df4_rrest['H'].plot(color='r');
df5_rrest['H'].plot(color='y');

# %%
df0_rest['η'].plot(color='c');
df0_rrest['η'].plot(color='k');
df1_rrest['η'].plot(color='g');
df2_rrest['η'].plot(color='b');
df4_rrest['η'].plot(color='r');
df5_rrest['η'].plot(color='y');

# %%
df_rest['σ'].plot(color='c');
df_rrest['σ'].plot(color='k');
df1_rrest['σ'].plot(color='g');
df2_rrest['σ'].plot(color='b');
df4_rrest['σ'].plot(color='r');
df5_rrest['σ'].plot(color='y');

# %%
df0_rrest['μ'].plot(color='k');
df1_rrest['μ'].plot(color='g');
df2_rrest['μ'].plot(color='b');
df4_rrest['μ'].plot(color='r');
df5_rrest['μ'].plot(color='y');

# %%
df4_rmrest['η'].plot(color='r');
df5_rmrest['η'].plot(color='y');

# %%
df4_rmrest['σ'].plot(color='r');
df5_rmrest['σ'].plot(color='y');

# %%
df4_rmrest['μ'].plot(color='r');
df5_rmrest['μ'].plot(color='y');

# %%
uz.qq_plots(pxchg0, s, α, η, σ, μ, hours)

# %%
uz.qq_plots(pxchg0, s, α, 0.327, 0.01, 0, hours)

# %%
uz.cum_H(pxchg0).plot(figsize=(9, 6), color='k');
uz.cum_H(pxchg1).plot(figsize=(9, 6), color='g');
uz.cum_H(pxchg2).plot(figsize=(9, 6), color='b');
uz.cum_H(pxchg3).plot(figsize=(9, 6), color='m');
uz.cum_H(pxchg4).plot(figsize=(9, 6), color='r');

# %%
uz.cum_H(pxchg0).loc[5000:].plot(figsize=(9, 6), color='k');
uz.cum_H(pxchg1).loc[5000:].plot(figsize=(9, 6), color='g');
uz.cum_H(pxchg2).loc[5000:].plot(figsize=(9, 6), color='b');
uz.cum_H(pxchg3).loc[5000:].plot(figsize=(9, 6), color='m');
uz.cum_H(pxchg4).loc[5000:].plot(figsize=(9, 6), color='r');


# %% Rolling indicators

def rolling_indic(df, window=100):
    dfc = df.copy()
    dfc['dPtj_r'] = dfc['dPtj'].rolling(window=window).sum()
    dfc['Co_r'] = dfc['Co'].rolling(window=window).sum()
    dfc['Al_r'] = dfc['Al'].rolling(window=window).sum()
    dfc['H_r'] = dfc['Co_r'] / (2 * dfc['Al_r'])
    dfc['dtj_Co'] = np.where(dfc['Co'], dfc['dtj'], 0)
    dfc['dtj_Al'] = np.where(dfc['Al'], dfc['dtj'], 0)
    dfc['dtj_Co_r'] = dfc['dtj_Co'].rolling(window=window).sum() / dfc['Co_r']
    dfc['dtj_Al_r'] = dfc['dtj_Al'].rolling(window=window).sum() / dfc['Al_r']
    dfc['dtj_Co/Al_r'] = dfc['dtj_Co_r'] / dfc['dtj_Al_r']
    dfc['eta_r'] = (np.sqrt(dfc['dtj_Co/Al_r']**2 - dfc['dtj_Co/Al_r'] + 1) +
                    (1 - dfc['dtj_Co/Al_r'])) / (2 * dfc['dtj_Co/Al_r'])
    return dfc

# %% Reduce to price changes

window_roll = 500

pxchg0_r = rolling_indic(pxchg0, window_roll)
pxchg1_r = rolling_indic(pxchg1, window_roll)
pxchg2_r = rolling_indic(pxchg2, window_roll)
pxchg3_r = rolling_indic(pxchg3, window_roll)
pxchg4_r = rolling_indic(pxchg4, window_roll)

# %% Plot dPtj

pxchg0_r['dPtj_r'].plot(figsize=(9, 6), color='k');
pxchg1_r['dPtj_r'].plot(figsize=(9, 6), color='g');
pxchg2_r['dPtj_r'].plot(figsize=(9, 6), color='b');
pxchg3_r['dPtj_r'].plot(figsize=(9, 6), color='m');
pxchg4_r['dPtj_r'].plot(figsize=(9, 6), color='r');

# %% Plot H

pxchg0_r['H_r'].plot(figsize=(9, 6), color='k');
pxchg1_r['H_r'].plot(figsize=(9, 6), color='g');
pxchg2_r['H_r'].plot(figsize=(9, 6), color='b');
pxchg3_r['H_r'].plot(figsize=(9, 6), color='m');
pxchg4_r['H_r'].plot(figsize=(9, 6), color='r');

# %% Plot Conditional durations - Alternations

pxchg0_r['dtj_Al_r'].plot(figsize=(9, 6), color='k');
pxchg1_r['dtj_Al_r'].plot(figsize=(9, 6), color='g');
pxchg2_r['dtj_Al_r'].plot(figsize=(9, 6), color='b');
pxchg3_r['dtj_Al_r'].plot(figsize=(9, 6), color='m');
pxchg4_r['dtj_Al_r'].plot(figsize=(9, 6), color='r');

# %% Plot Conditional durations - Continuations

pxchg0_r['dtj_Co_r'].plot(figsize=(9, 6), color='k');
pxchg1_r['dtj_Co_r'].plot(figsize=(9, 6), color='g');
pxchg2_r['dtj_Co_r'].plot(figsize=(9, 6), color='b');
pxchg3_r['dtj_Co_r'].plot(figsize=(9, 6), color='m');
pxchg4_r['dtj_Co_r'].plot(figsize=(9, 6), color='r');

# %% Plot Conditional durations ratio - Continuations / Alternations

(pxchg0_r['dtj_Co_r'] / pxchg0_r['dtj_Al_r']).plot(figsize=(9, 6), color='k');
(pxchg1_r['dtj_Co_r'] / pxchg1_r['dtj_Al_r']).plot(figsize=(9, 6), color='g');
(pxchg2_r['dtj_Co_r'] / pxchg2_r['dtj_Al_r']).plot(figsize=(9, 6), color='b');
(pxchg3_r['dtj_Co_r'] / pxchg3_r['dtj_Al_r']).plot(figsize=(9, 6), color='m');
(pxchg4_r['dtj_Co_r'] / pxchg4_r['dtj_Al_r']).plot(figsize=(9, 6), color='r');

# %% Plot eta

pxchg0_r['eta_r'].plot(figsize=(9, 6), color='k', linestyle=':');
pxchg1_r['eta_r'].plot(figsize=(9, 6), color='g');
pxchg2_r['eta_r'].plot(figsize=(9, 6), color='b');
pxchg3_r['eta_r'].plot(figsize=(9, 6), color='m');
pxchg4_r['eta_r'].plot(figsize=(9, 6), color='r', linestyle=':');

# %% Rupture model

algo_tr = rpt.Pelt(model='l2', min_size=100).fit(pxchg2['dtj'].dropna().values)
my_bkps_tr = algo_tr.predict(pen=3)

# %% Display Pelt

rpt.display(pxchg2['dtj'].dropna().values, my_bkps_tr)




