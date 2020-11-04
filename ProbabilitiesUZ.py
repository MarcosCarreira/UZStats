# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
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
# ## Feb-2020

# %% [markdown]
# ### Import packages

# %% Standard imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import seaborn as sns

# %% Import partial
from functools import partial

# %% Import optimize
from scipy import optimize

# %%
from scipy.special import kv

# %% Import dtt
import datetime as dtt

# %% Import numba
import numba as numba

# %% Import erfc
from math import erfc

# %% [markdown]
# ### Define functions

# %% Define adapt prob in numba

@numba.jit(nopython=True)
def nprobw(ud, s0, s, α, η, σ, μ, t):
    if t == 0:
        ans = 0
    else:
        ω = μ - 0.5 * σ ** 2
        x0 = np.log(s0)
        bu = np.log(s + (0.5 + η) * α)
        bd = np.log(s - (0.5 + η) * α)
        bpm = bu * (1 + ud) / 2 + bd * (1 - ud) / 2
        bmp = bu * (1 - ud) / 2 + bd * (1 + ud) / 2
        tyzn = 0
        n = 0
        tyznn = 1
        eps=1e-16
        while tyznn > eps:
            n += 1
            yn = ud * (2 * (n - 1) * bmp - (2 * n - 1) * bpm + x0)
            zn = ud * (2 * n * bmp - (2 * n - 1) * bpm - x0)
            tyznn =\
                (np.exp(ω * yn / (σ ** 2)) * erfc(
                -(yn + ω * t) / (σ * np.sqrt(2 * t))) -
                np.exp(ω * zn / (σ ** 2)) * erfc(
                -(zn + ω * t) / (σ * np.sqrt(2 * t))) +
                np.exp(-ω * yn / (σ ** 2)) * erfc(
                -(yn - ω * t) / (σ * np.sqrt(2 * t))) -
                np.exp(-ω * zn / (σ ** 2)) * erfc(
                -(zn - ω * t) / (σ * np.sqrt(2 * t)))) / 2
            tyzn += tyznn
        ans = np.exp(ω * (bpm - x0) / (σ ** 2)) * tyzn
    return ans

# %% nprobw loop
print('start')
currentDT1 = dtt.datetime.now()
print (str(currentDT1))

print(nprobw(1, 100, 100, 0.001, 0.25, 0.01, 0, 1))

print('end')
currentDT2 = dtt.datetime.now()
print (str(currentDT2))

print (str(currentDT2 - currentDT1))

# %% Define t_from_pud (for inverse CDF)
def t_from_pw(ud, s0, s, α, η, σ, μ):
    '''t_from_pw(ud, s0, s, α, η, σ, μ) solves the equation
    Pup(t) / Pup(1) (or Pdown(t) / Pdown(1))  = p;
    so for a random p such that 0 <= p <= 1 we find
    the expected time of a price change for the sign given'''
    mprobt = partial(nprobw, ud, s0, s, α, η, σ, μ)
    def groot0(t):
            return -mprobt(t)
    sol0 = optimize.minimize_scalar(groot0, bracket=[0, 1], method='golden')
    return sol0

# %% Define t_from_pud (for inverse CDF)
def drawfromp(ud, s0, s, α, η, σ, μ, n=10000):
    '''drawfromp(ud, s0, s, α, η, σ, μ, n=100) solves the equation
    Pup(t) / Pup(1) (or Pdown(t) / Pdown(1))  = p;
    so for a random p such that 0 <= p <= 1 we find
    the expected time of a price change for the sign given'''
    mprobt = partial(nprobw, ud, s0, s, α, η, σ, μ)
    ts = []
    rng = np.random.default_rng()
    vals = list(rng.random(n))
    for p in vals:
        def groot(t):
            return np.round(mprobt(t)/mprobt(1) - p, decimals=6)
        sol = optimize.root_scalar(groot, bracket=[0, 1], method='brentq')
        tp = sol.root
        ts = ts + [tp]
    return ts

# %% Example t_from_p
print('start')
currentDT1 = dtt.datetime.now()
print (str(currentDT1))

draws_A_Up = drawfromp(1, 100 + (0.5 - 0.5) * 0.01, 100, 0.01, 0.5, 0.01, 0)
pd.Series(draws_A_Up).hist(bins=100)

print('end')
currentDT2 = dtt.datetime.now()
print (str(currentDT2))

print (str(currentDT2 - currentDT1))

# %% Example t_from_p
print('start')
currentDT1 = dtt.datetime.now()
print (str(currentDT1))

print(st.geninvgauss.fit(draws_A_Up, floc=0))

print('end')
currentDT2 = dtt.datetime.now()
print (str(currentDT2))

print (str(currentDT2 - currentDT1))

# %% Define t_from_pud (for inverse CDF)
# def invprob(ud, s0, s, α, η, σ, μ, n=1000, printflag=False):
#     '''invprob(ud, s0, s, α, η, σ, μ, n) solves the equation
#     Pup(t) / Pup(1) (or Pdown(t) / Pdown(1))  = p;
#     so for a random p such that 0 <= p <= 1 we find
#     the expected time of a price change for the sign given'''
#     mprobt = partial(nprobw, ud, s0, s, α, η, σ, μ)
#     mprobt1 = mprobt(1)
#     ts = []
#     vals = np.linspace(0, (1 - 1 / n), n)
#     for p in vals:
#         def groot(t):
#             return np.round(mprobt(t)/mprobt1 - p, decimals=6)
#         sol = optimize.root_scalar(groot, bracket=[0, 1], method='brentq')
#         tp = sol.root
#         ts = ts + [tp]
#     ts = pd.Series(ts)
#     p, b, loc, scale = st.geninvgauss.fit(ts, scale=((α / s0) / σ) ** 2)
#     if printflag:
#         qtls = np.array([0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 0.999])
#         dfq = pd.DataFrame(ts.quantile(qtls))
#         dfq.columns = ['From P']
#         dfq['GIG'] = [st.geninvgauss.ppf(q, p, b, loc, scale) for q in qtls]
#         print(dfq)
#     return [p, b, loc, scale]

# %%
qtls_def = np.linspace(0, 0.90, 10)


# %% Define t_from_pud (for inverse CDF)
def invprob(ud, sud, s, α, η, σ, μ, qtls=qtls_def, print_flag=False):
    '''invprob(ud, s0, s, α, η, σ, μ, n) solves the equation
    Pup(t) / Pup(1) (or Pdown(t) / Pdown(1))  = p;
    so for a random p such that 0 <= p <= 1 we find
    the expected time of a price change for the sign given'''
    cont_flag =  - ud * sud
    s0 = s + sud * (0.5 - η) * α
    adjη = np.abs(ud + sud) * η + np.abs(ud - sud) / 2
    scl = ((α / s) / σ)**2
    adj = adjη * scl
    mprobt = partial(nprobw, ud, s0, s, α, η, σ, μ)
    mprobt1 = mprobt(1)
    ts = []
    for q in qtls:
        def groot(t):
            return np.round(mprobt(t)/mprobt1 - q, decimals=6)
        sol = optimize.root_scalar(groot, bracket=[0, 1], method='brentq')
        tp = sol.root
        ts = ts + [tp]
    ts = np.array(ts)
    x0 = np.array([-0.5, 1., 1.])
    def gig(x): #  x = [p, b, scale]; loc=0
        return np.array([st.geninvgauss.cdf(t, x[0], x[1], 0, x[2] * adj) for t in ts])
    def dist_gig(x):
        return np.linalg.norm(qtls - gig(x))
    fit_gig = optimize.minimize(fun=dist_gig, x0=x0, method='Nelder-Mead',
                                options={'disp': print_flag, 'maxiter': 2000, 'adaptive': True})
#     bounds_DE = [(-1, 1), (1e-6, 4), (0.1, 10)]
#     fit_gig_DE = optimize.differential_evolution(dist_gig, bounds=bounds_DE, popsize=50, tol=1e-2, disp=print_flag)
    p, b, scale_adj = fit_gig.x
    scale = scale_adj * adj
    dist = fit_gig.fun
    mean = scale * kv(1 + scale, b) / kv(scale, b)
    adj_mean = mean / scl
    fit_qtls = np.array([st.geninvgauss.ppf(q, p, b, 0, scale) for q in qtls])
    if print_flag:
        print(pd.DataFrame({'From P': ts, 'GIG': fit_qtls}, index=qtls))
    return pd.Series([ud, sud, cont_flag, s0, s, α, η, σ, μ, scl, adj, dist, p, b, scale, scale_adj, mean, adj_mean],
                        index=['ud', 'sud', 'Co/Al', 's0', 's', 'α', 'η', 'σ', 'μ', 'scl', 'adj', 'dist', 'p', 'b', 'scale', 'scale_adj', 'mean', 'mean_scl'])

# %% Define t_from_pud (for inverse CDF)
def invprobgig(ud, sud, s, α, p, b, scale, qtls=qtls_def, print_flag=False):
    '''invprob(ud, s0, s, α, η, σ, μ, n) solves the equation
    Pup(t) / Pup(1) (or Pdown(t) / Pdown(1))  = p;
    so for a random p such that 0 <= p <= 1 we find
    the expected time of a price change for the sign given'''
    fit_qtls = np.array([st.geninvgauss.ppf(q, p, b, 0, scale) for q in qtls])
    def pft(x): #  x = [η, σ, μ];
        return np.array([nprobw(ud, s + sud * (0.5 - x[0]) * α, s, α, x[0], x[1], x[2], t) /
                         nprobw(ud, s + sud * (0.5 - x[0]) * α, s, α, x[0], x[1], x[2], 1) for t in fit_qtls])
    def dist_gig(x):
        return np.linalg.norm(qtls - pft(x))
    x0 = np.array([0.3, 0.01, 0.])
    fit_gig = optimize.minimize(fun=dist_gig, x0=x0, method='Nelder-Mead',
                                options={'disp': print_flag, 'maxiter': 2000, 'adaptive': True})
#     bounds_DE = [(-1, 1), (1e-6, 4), (0.1, 10)]
#     fit_gig_DE = optimize.differential_evolution(dist_gig, bounds=bounds_DE, popsize=50, tol=1e-2, disp=print_flag)
    η, σ, μ = fit_gig.x
    cont_flag =  - ud * sud
    s0 = s + sud * (0.5 - η) * α
    scl = ((α / s) / σ)**2
    adjη = np.abs(ud + sud) * η + np.abs(ud - sud) / 2
    adj = adjη * scl 
    scale_adj = scale / adj
    dist = fit_gig.fun
    mean = scale * kv(1 + scale, b) / kv(scale, b)
    adj_mean = mean / scl
    if print_flag:
        mprobt = partial(nprobw, ud, s0, s, α, η, σ, μ)
        mprobt1 = mprobt(1)
        ts = []
        for q in qtls:
            def groot(t):
                return np.round(mprobt(t)/mprobt1 - q, decimals=6)
            sol = optimize.root_scalar(groot, bracket=[0, 1], method='brentq')
            tp = sol.root
            ts = ts + [tp]
        ts = np.array(ts)
        print(pd.DataFrame({'From P': ts, 'GIG': fit_qtls}, index=qtls))
    return pd.Series([ud, sud, cont_flag, s0, s, α, η, σ, μ, scl, adj, dist, p, b, scale, scale_adj, mean, adj_mean],
                        index=['ud', 'sud', 'Co/Al', 's0', 's', 'α', 'η', 'σ', 'μ', 'scl', 'adj', 'dist', 'p', 'b', 'scale', 'scale_adj', 'mean', 'mean_scl'])

# %% Example t_from_p
print('start')
currentDT1 = dtt.datetime.now()
print (str(currentDT1))

ud = +1
s = 100
α = 0.01
η = 0.5
σ = 0.01
μ = 0
sud = +1
df = invprob(ud, sud, s, α, η, σ, μ, print_flag=True)

print('end')
currentDT2 = dtt.datetime.now()
print (str(currentDT2))

print (str(currentDT2 - currentDT1))
df

# %% Example t_from_p
print('start')
currentDT1 = dtt.datetime.now()
print (str(currentDT1))

ud = +1
s = 100
μ = 0
sud = +1
df = invprobgig(ud, sud, s, α, 0.371050, 1.166569, 0.585022 * 0.0001, print_flag=True)

print('end')
currentDT2 = dtt.datetime.now()
print (str(currentDT2))

print (str(currentDT2 - currentDT1))
df

# %% Example t_from_p
print('start')
currentDT1 = dtt.datetime.now()
print (str(currentDT1))

ud = +1
s = 100
α = 0.01
η = 0.01
σ = 0.01
μ = 0
sud = +1
df = invprob(ud, sud, s, α, η, σ, μ, η)

print('end')
currentDT2 = dtt.datetime.now()
print (str(currentDT2))

print (str(currentDT2 - currentDT1))
df

# %%
height_def = 7
aspect_def = 1.5

# %% Example t_from_p
print('start')
currentDT1 = dtt.datetime.now()
print (str(currentDT1))

tbl_eta = np.linspace(0.01, 0.50, 50)
s = 100
α = 0.01
σ = 0.01
μ = 0

chg_eta_A_Up = pd.DataFrame({η: invprob(+1, +1, s, α, η, σ, μ, 2*η) for η in tbl_eta}).transpose()
print('chg_eta_A_Up')
chg_eta_A_Up.to_csv('chg_eta_A_Up.csv', index=False)
print (str(dtt.datetime.now()))
chg_eta_A_Down = pd.DataFrame({η: invprob(-1, -1, s, α, η, σ, μ, 2*η) for η in tbl_eta}).transpose()
print('chg_eta_A_Down')
chg_eta_A_Down.to_csv('chg_eta_A_Down.csv', index=False)
print (str(dtt.datetime.now()))
chg_eta_C_Up = pd.DataFrame({η: invprob(+1, -1, s, α, η, σ, μ, 1) for η in tbl_eta}).transpose()
print('chg_eta_C_Up')
chg_eta_C_Up.to_csv('chg_eta_C_Up.csv', index=False)
print (str(dtt.datetime.now()))
chg_eta_C_Down = pd.DataFrame({η: invprob(-1, +1, s, α, η, σ, μ, 1) for η in tbl_eta}).transpose()
print('chg_eta_C_Down')
chg_eta_C_Down.to_csv('chg_eta_C_Down.csv', index=False)
print (str(dtt.datetime.now()))

chg_eta = pd.concat([chg_eta_A_Up, chg_eta_A_Down, chg_eta_C_Up, chg_eta_C_Down])
print('chg_eta')

del(s)
del(α)
del(σ)
del(μ)

print('end')
currentDT2 = dtt.datetime.now()
print (str(currentDT2))

print (str(currentDT2 - currentDT1))

# %%
sns.relplot(x='η', y='dist', hue='Co/Al', data=chg_eta, kind='line', palette='Set1', height=height_def, aspect=aspect_def);

# %%
sns.relplot(x='η', y='p', hue='Co/Al', data=chg_eta, kind='line', palette='Set1', height=height_def, aspect=aspect_def);

# %%
sns.relplot(x='η', y='b', hue='Co/Al', data=chg_eta, kind='line', palette='Set1', height=height_def, aspect=aspect_def);

# %%
sns.relplot(x='η', y='scale_adj', hue='Co/Al', data=chg_eta, kind='line', palette='Set1', height=height_def, aspect=aspect_def);

# %%
sns.relplot(x='η', y='mean_scl', hue='Co/Al', data=chg_eta, kind='line', palette='Set1', height=height_def, aspect=aspect_def);

# %%
sns.relplot(x='p', y='b', hue='Co/Al', data=chg_eta, palette='Set1', height=height_def, aspect=1);

# %%
chg_eta_A_Up[['p', 'b', 'scale_adj', 'mean_scl']].plot(figsize=(12, 8));
chg_eta_A_Down[['p', 'b', 'scale_adj', 'mean_scl']].plot(figsize=(12, 8));
chg_eta_C_Up[['p', 'b', 'scale_adj', 'mean_scl']].plot(figsize=(12, 8));
chg_eta_C_Down[['p', 'b', 'scale_adj', 'mean_scl']].plot(figsize=(12, 8));

# %% Example t_from_p
print('start')
currentDT1 = dtt.datetime.now()
print (str(currentDT1))

s = 100
# tbl_alpha = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
# tbl_eta = np.linspace(0.05, 0.50, 10)
# tbl_vol = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
# tbl_mu = np.array([-1., -0.5, -0.1, -0.01, 0, +0.01, +0.1, +0.5, +1.])
tbl_alpha = np.array([0.0025, 0.005, 0.01, 0.02])
tbl_eta = np.array([0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
tbl_vol = np.array([0.0025, 0.005, 0.01, 0.02])
μ = 0

chg_A_Up = pd.DataFrame({(α, η, σ): invprob(+1, +1, s, α, η, σ, μ, 2*η) for α in tbl_alpha for η in tbl_eta for σ in tbl_vol}).transpose()
print('chg_A_Up')
chg_A_Up.to_csv('chg_A_Up.csv', index=False)
print (str(dtt.datetime.now()))
chg_A_Down = pd.DataFrame({(α, η, σ): invprob(-1, -1, s, α, η, σ, μ, 2*η) for α in tbl_alpha for η in tbl_eta for σ in tbl_vol}).transpose()
print('chg_A_Down')
chg_A_Down.to_csv('chg_A_Down.csv', index=False)
print (str(dtt.datetime.now()))
chg_C_Up = pd.DataFrame({(α, η, σ): invprob(+1, -1, s, α, η, σ, μ, 1) for α in tbl_alpha for η in tbl_eta for σ in tbl_vol}).transpose()
print('chg_C_Up')
chg_C_Up.to_csv('chg_C_Up.csv', index=False)
print (str(dtt.datetime.now()))
chg_C_Down = pd.DataFrame({(α, η, σ): invprob(-1, +1, s, α, η, σ, μ, 1) for α in tbl_alpha for η in tbl_eta for σ in tbl_vol}).transpose()
print('chg_C_Down')
chg_C_Down.to_csv('chg_C_Down.csv', index=False)
print (str(dtt.datetime.now()))

chgs = pd.concat([chg_A_Up, chg_A_Down, chg_C_Up, chg_C_Down])
print('chgs')

del(s)
del(μ)

print('end')
currentDT2 = dtt.datetime.now()
print (str(currentDT2))

print (str(currentDT2 - currentDT1))

# %%
sns.relplot(x='η', y='dist', hue='Co/Al', data=chgs, kind='line', palette='Set1', height=height_def, aspect=aspect_def);

# %%
sns.relplot(x='η', y='p', hue='Co/Al', data=chgs, kind='line', palette='Set1', height=height_def, aspect=aspect_def);

# %%
sns.relplot(x='η', y='b', hue='Co/Al', data=chgs, kind='line', palette='Set1', height=height_def, aspect=aspect_def);

# %%
sns.relplot(x='η', y='scale_adj', hue='Co/Al', data=chgs, kind='line', palette='Set1', height=height_def, aspect=aspect_def);

# %%
sns.relplot(x='η', y='mean_scl', hue='Co/Al', data=chgs, kind='line', palette='Set1', height=height_def, aspect=aspect_def);

# %% Example t_from_p
print('start')
currentDT1 = dtt.datetime.now()
print (str(currentDT1))

s = 100
α = 0.002
tbl_eta = np.linspace(0.05, 0.50, 50)
σ = 0.01
μ = 0.10* (12 * 9)

chg_eta_mu_A_Up = pd.DataFrame({η: invprob(+1, +1, s, α, η, σ, μ, 2*η) for η in tbl_eta}).transpose()
print('chg_eta_mu_A_Up')
chg_eta_mu_A_Up.to_csv('chg_eta_mu_A_Up.csv', index=False)
print (str(dtt.datetime.now()))
chg_eta_mu_A_Down = pd.DataFrame({η: invprob(-1, -1, s, α, η, σ, μ, 2*η) for η in tbl_eta}).transpose()
print('chg_eta_mu_A_Down')
chg_eta_mu_A_Down.to_csv('chg_eta_mu_A_Down.csv', index=False)
print (str(dtt.datetime.now()))
chg_eta_mu_C_Up = pd.DataFrame({η: invprob(+1, -1, s, α, η, σ, μ, 1) for η in tbl_eta}).transpose()
print('chg_eta_mu_C_Up')
chg_eta_mu_C_Up.to_csv('chg_eta_mu_C_Up.csv', index=False)
print (str(dtt.datetime.now()))
chg_eta_mu_C_Down = pd.DataFrame({η: invprob(-1, +1, s, α, η, σ, μ, 1) for η in tbl_eta}).transpose()
print('chg_eta_mu_C_Down')
chg_eta_mu_C_Down.to_csv('chg_eta_mu_C_Down.csv', index=False)
print (str(dtt.datetime.now()))

chg_eta_mu = pd.concat([chg_eta_mu_A_Up, chg_eta_mu_A_Down, chg_eta_mu_C_Up, chg_eta_mu_C_Down])
print('chg_mu')

del(s)
del(α)
del(σ)
del(μ)

print('end')
currentDT2 = dtt.datetime.now()
print (str(currentDT2))

print (str(currentDT2 - currentDT1))

# %%
sns.relplot(x='η', y='dist', hue='Co/Al', data=chg_eta, kind='line', palette='Set1', height=5, aspect=aspect_def);
sns.relplot(x='η', y='dist', hue='Co/Al', data=chg_eta_mu, kind='line', palette='Set1', height=5, aspect=aspect_def);

# %%
sns.relplot(x='η', y='p', row='sud', col='ud', data=chg_eta, kind='line', palette='Set1', height=5, aspect=aspect_def);

# %%
sns.relplot(x='η', y='p', row='sud', col='ud', data=chg_eta_mu, kind='line', palette='Set1', height=5, aspect=aspect_def);

# %%
sns.relplot(x='η', y='p', hue='Co/Al', data=chg_eta, kind='line', palette='Set1', height=5, aspect=aspect_def);
sns.relplot(x='η', y='p', hue='Co/Al', data=chg_eta_mu, kind='line', palette='Set1', height=5, aspect=aspect_def);

# %%
sns.relplot(x='η', y='b', hue='Co/Al', data=chg_eta, kind='line', palette='Set1', height=5, aspect=aspect_def);
sns.relplot(x='η', y='b', hue='Co/Al', data=chg_eta_mu, kind='line', palette='Set1', height=5, aspect=aspect_def);

# %%
sns.relplot(x='η', y='scale_adj', hue='Co/Al', data=chg_eta, kind='line', palette='Set1', height=5, aspect=aspect_def);
sns.relplot(x='η', y='scale_adj', hue='Co/Al', data=chg_eta_mu, kind='line', palette='Set1', height=5, aspect=aspect_def);

# %%
sns.relplot(x='η', y='mean_scl', hue='Co/Al', data=chg_eta, kind='line', palette='Set1', height=5, aspect=aspect_def);
sns.relplot(x='η', y='mean_scl', hue='Co/Al', data=chg_eta_mu, kind='line', palette='Set1', height=5, aspect=aspect_def);

# %% Example t_from_p
print('start')
currentDT1 = dtt.datetime.now()
print (str(currentDT1))

s = 100
α = 0.002
η= 0.50
σ = 0.01
tbl_mu = np.linspace(-0.10, +0.10, 21)
μt = (60 * 9) / 5

chg_mu_A_Up = pd.DataFrame({μ: invprob(+1, +1, s, α, η, σ, μ * μt, 2*η) for μ in tbl_mu}).transpose()
print('chg_mu_A_Up')
chg_mu_A_Up.to_csv('chg_mu_A_Up.csv', index=False)
print (str(dtt.datetime.now()))
chg_mu_A_Down = pd.DataFrame({μ: invprob(-1, -1, s, α, η, σ, μ * μt, 2*η) for μ in tbl_mu}).transpose()
print('chg_mu_A_Down')
chg_mu_A_Down.to_csv('chg_mu_A_Down.csv', index=False)
print (str(dtt.datetime.now()))
chg_mu_C_Up = pd.DataFrame({μ: invprob(+1, -1, s, α, η, σ, μ * μt, 1) for μ in tbl_mu}).transpose()
print('chg_mu_C_Up')
chg_mu_C_Up.to_csv('chg_mu_C_Up.csv', index=False)
print (str(dtt.datetime.now()))
chg_mu_C_Down = pd.DataFrame({μ: invprob(-1, +1, s, α, η, σ, μ * μt, 1) for μ in tbl_mu}).transpose()
print('chg_mu_C_Down')
chg_mu_C_Down.to_csv('chg_mu_C_Down.csv', index=False)
print (str(dtt.datetime.now()))

chg_mu = pd.concat([chg_mu_A_Up, chg_mu_A_Down, chg_mu_C_Up, chg_mu_C_Down])
print('chg_mu')

del(s)
del(η)
del(α)
del(σ)

print('end')
currentDT2 = dtt.datetime.now()
print (str(currentDT2))

print (str(currentDT2 - currentDT1))

# %%
sns.relplot(x='μ', y='dist', hue='Co/Al', data=chg_mu, kind='line', palette='Set1', height=height_def, aspect=2);

# %%
sns.relplot(x='μ', y='p', hue='Co/Al', data=chg_mu, kind='line', palette='Set1', height=height_def, aspect=2);

# %%
sns.relplot(x='μ', y='b', hue='Co/Al', data=chg_mu, kind='line', palette='Set1', height=height_def, aspect=2);

# %%
sns.relplot(x='μ', y='scale_adj', hue='Co/Al', data=chg_mu, kind='line', palette='Set1', height=height_def, aspect=2);

# %%
sns.relplot(x='μ', y='mean_scl', hue='Co/Al', data=chg_mu, kind='line', palette='Set1', height=height_def, aspect=2);


# %% Define t_from_p (for inverse CDF)
def t_from_p(s0, s, α, η, σ, μ, p):
    '''t_from_p(s0, s, α, η, σ, μ, p) solves the equation
    Pup(t) + Pdown(t) = p; so for a random p such that 0 <= p <= 1 we find
    the expected time of a price change'''
    mprobtup = partial(nprobw, 1, s0, s, α, η, σ, μ)
    mprobtdown = partial(nprobw, -1, s0, s, α, η, σ, μ)
    def groot(t):
        return np.round(mprobtup(t) + mprobtdown(t) - p, decimals=6)
    sol = optimize.root_scalar(groot, bracket=[0, 1], method='brentq')
    return sol.root

# %% Define t_from_pud (for inverse CDF)
def t_from_pud(s0, s, α, η, σ, μ, ud, p):
    '''t_from_pud(s0, s, α, η, σ, μ, ud, p) solves the equation
    Pup(t) / Pup(1) (or Pdown(t) / Pdown(1))  = p;
    so for a random p such that 0 <= p <= 1 we find
    the expected time of a price change for the sign given'''
    mprobt = partial(nprobw, ud, s0, s, α, η, σ, μ)
    def groot(t):
        return np.round(mprobt(t)/mprobt(1) - p, decimals=6)
    sol = optimize.root_scalar(groot, bracket=[0, 1], method='brentq')
    return sol.root

# %% Example t_from_p
print('start')
currentDT1 = dtt.datetime.now()
print (str(currentDT1))

t50 = t_from_p(100 + (0.5 - 0.2) * 0.01, 100, 0.01, 0.5, 0.01, 0, 0.999999)
print(t50)

print('end')
currentDT2 = dtt.datetime.now()
print (str(currentDT2))

print (str(currentDT2 - currentDT1))

# %% Define CDF
def cdf(s0, s, α, η, σ, μ, npts=100+1):
    grid = np.linspace(0., 0.999, npts)
    pts = pd.Series(grid, index=[t_from_p(s0, s, α, η, σ, μ, p)
                                 for p in grid])
    pts.index.name = 't'
    pts.name = 'CDF(t)'
    return pts

# %% Define CDFs
def cdfs(s0, s, α, η, σ, μ, npts=100):
    grid = np.linspace(0, 0.999, npts)
    ts = [t_from_p(s0, s, α, η, σ, μ, p) for p in grid]
    up = [nprobw(+1, s0, s, α, η, σ, μ, t) for t in ts]
    up = np.minimum(up, grid)
    pts = pd.DataFrame({'CDF(t)': grid, 'Up': up, 'Down': grid - up},
                       index=ts)
    pts.index.name = 't'
    # pts.loc[0] = [0., up[0], 1 - up[0]]
    return pts.sort_index()

# %% test

tcdf = cdfs(100 + (0.5 - 0.25) * 0.01, 100, 0.01, 0.25, 0.01, 0)

# %% Plot CDFs
def cdfsplot(s0, s, α, η, σ, μ, npts=100+1):
    df = cdfs(s0, s, α, η, σ, μ, npts)
    title = str([s0, s, α, η, σ, μ])
    df.plot(figsize=(9, 6), marker='o', title=title)

# %% test jupyter={"outputs_hidden": true}

cdfsplot(100 + (0.5 - 0.25) * 0.01, 100, 0.01, 0.25, 0.01, 0)
# cdfsplot(100 + (0.5 - 0.25) * 0.01, 100, 0.01, 0.25, 0.015, 0)
cdfsplot(100 + (0.5 - 0.25) * 0.01, 100, 0.01, 0.25, 0.01, 0.002*9*60)
cdfsplot(100 - (0.5 - 0.25) * 0.01, 100, 0.01, 0.25, 0.01, 0)
cdfsplot(100 - (0.5 - 0.25) * 0.01, 100, 0.01, 0.25, 0.01, 0.002*9*60)

# %% CDFs of simul jupyter={"outputs_hidden": true}

cdfsplot(100 + (0.5 - 0.25) * 0.01, 100, 0.01, 0.25, 0.01, 0)
cdfsplot(100 - (0.5 - 0.25) * 0.01, 100, 0.01, 0.25, 0.01, 0)

# %% Calculate CDF

print('start')
currentDT1 = dtt.datetime.now()
print (str(currentDT1))

pts0 = cdf(100 + (0.5 - 0.5) * 0.01, 100, 0.01, 0.5, 0.01, 0)
pts1 = cdf(100 + (0.5 - 0.2) * 0.01, 100, 0.01, 0.2, 0.01, 0)
pts2 = cdf(100 + (0.5 - 0.5) * 0.01, 100, 0.01, 0.5, 0.01, 0.10)
pts3 = cdf(100 + (0.5 - 0.5) * 0.01, 100, 0.01, 0.5, 0.015, 0)

print('end')
currentDT2 = dtt.datetime.now()
print (str(currentDT2))

print (str(currentDT2 - currentDT1))

# %% Plot CDF jupyter={"outputs_hidden": true}

pts0.plot(figsize=(9, 6), color='b', marker='o');
pts1.plot(figsize=(9, 6), color='r', marker='o');
pts2.plot(figsize=(9, 6), color='c', marker='o');
pts3.plot(figsize=(9, 6), color='k', marker='o');

# %% Define up_down
def up_down(s0, s, α, η, σ, μ, t, p):
    '''up_down(s0, s, α, η, σ, μ, t, p) finds Pup(t) and Pdown(t) with t
    given by solving t_from_p outside the function so that
    for a random p such that 0 <= p <= 1 we choose
    the expected sign (+1 or -1) of a price change'''
    pup = nprobw(1, s0, s, α, η, σ, μ, t)
    pdown = nprobw(-1, s0, s, α, η, σ, μ, t)
    return int(2 * np.heaviside(p - pdown / (pup + pdown), 1) - 1)


# %% Example up_down
print('start')
currentDT1 = dtt.datetime.now()
print (str(currentDT1))

print(
      np.array(
          [up_down(100 + (0.5 - 0.5) * 0.01, 100, 0.01, 0.5, 0.01, 0, t50,
           np.random.uniform(low=0., high=1.)) for j in range(20)]))

print('end')
currentDT2 = dtt.datetime.now()
print (str(currentDT2))

print (str(currentDT2 - currentDT1))

# %% Define next move
def next_move(s0, s, α, η, σ, μ):
    p = np.random.uniform(low=0., high=0.999999)
    t = t_from_p(s0, s, α, η, σ, μ, p)
    p = np.random.uniform(low=0., high=0.999999)
    ud = up_down(s0, s, α, η, σ, μ, t, p)
    return [t, ud]

# %% Example next_move
print('start')
currentDT1 = dtt.datetime.now()
print (str(currentDT1))

print(next_move(100 + (0.5 - 0.5) * 0.01, 100, 0.01, 0.5, 0.01, 0))

print('end')
currentDT2 = dtt.datetime.now()
print (str(currentDT2))

print (str(currentDT2 - currentDT1))

# %% Define next state t first
def next_state_t(α, η, σ, μ, state):
    sup = state[0]
    s = state[1]
    prev_s = s + sup * (0.5 - η) * α
    p = np.random.uniform(low=0., high=0.999999)
    t = t_from_p(prev_s, s, α, η, σ, μ, p)
    p = np.random.uniform(low=0., high=0.999999)
    up = up_down(prev_s, s, α, η, σ, μ, t, p)
    alt = sup == up
    cont = sup != up
    return [[t, s + up * α, up, alt, cont], [-up, s + up * α]]

# %% Define next state
def next_state(α, η, σ, μ, state):
    sup = state[0]
    s = state[1]
    prev_s = s + sup * (0.5 - η) * α
    p = np.random.uniform(low=0., high=0.999999)
    up = up_down(prev_s, s, α, η, σ, μ, 1, p)
    p = np.random.uniform(low=0., high=0.999999)
    t = t_from_pud(prev_s, s, α, η, σ, μ, up, p)
    alt = sup == up
    cont = sup != up
    return [[t, s + up * α, up, alt, cont], [-up, s + up * α]]


# %% Example next_state
print('start')
currentDT1 = dtt.datetime.now()
print (str(currentDT1))

print(next_state(0.01, 0.25, 0.01, 0, [+1, 100]))

print('end')
currentDT2 = dtt.datetime.now()
print (str(currentDT2))

print (str(currentDT2 - currentDT1))

# %% Full simulation with random 1st state

def simul(s, α, η, σ, μ, T):
    state = [np.random.choice([-1, +1]), s]
    df = pd.DataFrame({'dt': 0, 'P': state[1], 'Sign': 0,
                       'Al': True, 'Co': False}, index=[0])
    n = 1
    while df['dt'].sum() < T:
        row, state = next_state(α, η, σ, μ, state)
        df.loc[n] = row
        n += 1
    df['t'] = df['dt'].cumsum()
    df.set_index('t', inplace=True)
    return df

# %% Example simul
print('start')
currentDT1 = dtt.datetime.now()
print (str(currentDT1))

df_simul = simul(100, 0.01, 0.25, 0.01, 0, 1)

print('end')
currentDT2 = dtt.datetime.now()
print (str(currentDT2))

print (str(currentDT2 - currentDT1))

# %% Plot price

df_simul['P'].plot()

# %% Quantiles durations

def acup_qtl(df):
    df_aup = df[(df['Sign'] == +1) & (df['Al'])]['dt']
    df_cup = df[(df['Sign'] == +1) & (df['Co'])]['dt']
    df_ado = df[(df['Sign'] == -1) & (df['Al'])]['dt']
    df_cdo = df[(df['Sign'] == -1) & (df['Co'])]['dt']
    qtl_rng = np.linspace(0.01, 1.00, 100)
    df_aupq = df_aup.quantile(qtl_rng).reset_index().set_index('dt')
    df_cupq = df_cup.quantile(qtl_rng).reset_index().set_index('dt')
    df_adoq = df_ado.quantile(qtl_rng).reset_index().set_index('dt')
    df_cdoq = df_cdo.quantile(qtl_rng).reset_index().set_index('dt')
    df_aupq.columns = ['p_Al_Up']
    df_cupq.columns = ['p_Co_Up']
    df_adoq.columns = ['p_Al_Do']
    df_cdoq.columns = ['p_Co_Do']
    return [df_aupq, df_cupq, df_adoq, df_cdoq]

# %% Example quantiles

df_quant = acup_qtl(df_simul)
pd.concat(df_quant, axis=1).interpolate().plot()

# %% H

def roll_H(df):
    dfc = df.copy()
    dfc['Cum_Al'] = dfc['Al'].cumsum()
    dfc['Cum_Co'] = dfc['Co'].cumsum()
    dfc['H'] = dfc['Cum_Co'] / (2 * dfc['Cum_Al'])
    return dfc['H']

# %% Example H

print(df_simul[['Al', 'Co']].sum())
df_H = roll_H(df_simul)
df_H.plot()

# %% [markdown]
# ### Examples

# %% Lists of parameter values
# t_values = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
t_values = [1e-5, 1e-4, 1e-3, 1e-2]
alpha_values = [0.001, 0.01, 0.1, 1]
eta_values = [0.05, 0.2, 0.35, 0.5]
vol_values = [0.001, 0.01, 0.1]
mu_values = [0, 0.02, 0.10]

# %% Find stat_dist
stat_dist([[0.4, 0.6], [0.2, 0.8]])

# %% inita
inita = np.array([[0,0],[1,1]])
