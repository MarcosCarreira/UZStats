# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Latency and Tick

# %% [markdown]
# ## Python Imports

# %% Python imports
import numpy as np
import matplotlib
# import matplotlib.pyplot as plt

# %% Python imports
from scipy.optimize import minimize

# %% Python imports
from functools import partial

# %%
# from numba import njit

# %% [markdown]
# ## Tick Imports

# %%
# # %matplotlib inline

# %% Tick Imports
from tick.hawkes import SimuHawkes, SimuHawkesMulti
from tick.base import TimeFunction
from tick.hawkes import HawkesKernelTimeFunc, HawkesKernelExp
# from tick.hawkes import HawkesKernelSumExp, HawkesKernel0
# from tick.hawkes import HawkesEM, HawkesSumExpKern
# from tick.hawkes import SimuPoissonProcess, SimuInhomogeneousPoisson
# from tick.hawkes import HawkesBasisKernels, HawkesSumGaussians
# from tick.plot import plot_timefunction
# from tick.plot import plot_point_process
# from tick.plot import plot_hawkes_kernels
# from tick.plot import plot_basis_kernels

# %%
import hawklat as hl

# %% [markdown]
# ## Simple example of exponential Kernel

# %%
α1 = 0.6
β1 = 0.8
λ01 = 1.2


# %%
def ek1(t):
    return α1 * np.exp(-β1 * t)


# %%
sup1 = 10

# %%
ekc1 = hl.time_funclat(ek1, sup1, 0)

# %%
ektf1 = HawkesKernelTimeFunc(ekc1)

# %%
ekprm1 = HawkesKernelExp(α1 / β1, β1)

# %%
# plot_timefunction(ekc1)

# %%
n_paths = 100

# %%
end_time_1 = 100
end_time_2 = 1000
end_time_3 = 10000

# %%
hawkestf1 = SimuHawkes(baseline=[λ01], kernels=[[ektf1]], end_time=end_time_1, verbose=False, seed=13)
hawkestf2 = SimuHawkes(baseline=[λ01], kernels=[[ektf1]], end_time=end_time_2, verbose=False, seed=13)
hawkestf3 = SimuHawkes(baseline=[λ01], kernels=[[ektf1]], end_time=end_time_3, verbose=False, seed=13)

# %%
multitf1 = SimuHawkesMulti(hawkestf1, n_simulations=n_paths)
multitf2 = SimuHawkesMulti(hawkestf2, n_simulations=n_paths)
multitf3 = SimuHawkesMulti(hawkestf3, n_simulations=n_paths)

# %%
multitf1.simulate()
multitf2.simulate()
multitf3.simulate()

# %%
hawkesprm1 = SimuHawkes(baseline=[λ01], kernels=[[ekprm1]], end_time=end_time_1, verbose=False, seed=13)
hawkesprm2 = SimuHawkes(baseline=[λ01], kernels=[[ekprm1]], end_time=end_time_2, verbose=False, seed=13)
hawkesprm3 = SimuHawkes(baseline=[λ01], kernels=[[ekprm1]], end_time=end_time_3, verbose=False, seed=13)

# %%
multiprm1 = SimuHawkesMulti(hawkesprm1, n_simulations=n_paths)
multiprm2 = SimuHawkesMulti(hawkesprm2, n_simulations=n_paths)
multiprm3 = SimuHawkesMulti(hawkesprm3, n_simulations=n_paths)

# %%
multiprm1.simulate()
multiprm2.simulate()
multiprm3.simulate()

# %%
def_bounds = [(0.01, 2.0), (0.01, 100.), (0.01, 200.)]

# %%
def_constraint = {'type': 'ineq', 'fun': lambda x:  x[2] - x[1]}

# %%
def_x0 = np.array([1., 0.3, 0.5])

# %%
# %%time

np.mean(hl.findθ(multiprm1.timestamps,
                 bounds=def_bounds, constraints=def_constraint,
                 x0=def_x0), axis=0)

# %%
# %%time

np.mean(hl.findθlat(multiprm1.timestamps,
                 bounds=def_bounds, constraints=def_constraint,
                 x0=def_x0, τ=0), axis=0)

# %%
# %%time

np.mean(hl.findθ(multitf1.timestamps,
                 bounds=def_bounds, constraints=def_constraint,
                 x0=def_x0), axis=0)

# %%
# %%time

np.mean(hl.findθlat(multitf1.timestamps,
                 bounds=def_bounds, constraints=def_constraint,
                 x0=def_x0, τ=0), axis=0)

# %%
# %%time

np.mean(hl.findθ(multiprm2.timestamps,
                 bounds=def_bounds, constraints=def_constraint,
                 x0=def_x0), axis=0)

# %%
# %%time

np.mean(hl.findθlat(multiprm2.timestamps,
                 bounds=def_bounds, constraints=def_constraint,
                 x0=def_x0, τ=0), axis=0)

# %%
# %%time

np.mean(hl.findθ(multitf2.timestamps,
                 bounds=def_bounds, constraints=def_constraint,
                 x0=def_x0), axis=0)

# %%
# %%time

np.mean(hl.findθlat(multitf2.timestamps,
                 bounds=def_bounds, constraints=def_constraint,
                 x0=def_x0, τ=0), axis=0)

# %%
# %%time

np.mean(hl.findθ(multiprm3.timestamps,
                 bounds=def_bounds, constraints=def_constraint,
                 x0=def_x0), axis=0)

# %%
# %%time

np.mean(hl.findθlat(multiprm3.timestamps,
                 bounds=def_bounds, constraints=def_constraint,
                 x0=def_x0, τ=0), axis=0)

# %%
# %%time

np.mean(hl.findθ(multitf3.timestamps,
                 bounds=def_bounds, constraints=def_constraint,
                 x0=def_x0), axis=0)

# %%
# %%time

np.mean(hl.findθlat(multitf3.timestamps,
                 bounds=def_bounds, constraints=def_constraint,
                 x0=def_x0, τ=0), axis=0)

# %% [markdown]
# ## Simple example of exponential Kernel with latency

# %% [markdown]
# ### Latency > 0

# %%
sup1 = 10
lat1 = 2

# %%
ekc3 = hl.time_funclat(ek1, sup1, lat1)

# %%
ektf3 = HawkesKernelTimeFunc(ekc3)

# %%
# plot_timefunction(ekc3)

# %%
hawkestf3_1 = SimuHawkes(baseline=[λ01], kernels=[[ektf3]], end_time=end_time_1, verbose=False, seed=13)
hawkestf3_2 = SimuHawkes(baseline=[λ01], kernels=[[ektf3]], end_time=end_time_2, verbose=False, seed=13)
hawkestf3_3 = SimuHawkes(baseline=[λ01], kernels=[[ektf3]], end_time=end_time_3, verbose=False, seed=13)

# %%
multitf3_1 = SimuHawkesMulti(hawkestf3_1, n_simulations=n_paths)
multitf3_2 = SimuHawkesMulti(hawkestf3_2, n_simulations=n_paths)
multitf3_3 = SimuHawkesMulti(hawkestf3_3, n_simulations=n_paths)

# %%
multitf3_1.simulate()
multitf3_2.simulate()
multitf3_3.simulate()

# %%
# %%time

np.mean(hl.findθlat(multitf3_1.timestamps,
                 bounds=def_bounds, constraints=def_constraint,
                 x0=def_x0, τ=lat1), axis=0)

# %%
# %%time

np.mean(hl.findθlat(multitf3_2.timestamps,
                 bounds=def_bounds, constraints=def_constraint,
                 x0=def_x0, τ=lat1), axis=0)

# %%
# %%time

np.mean(hl.findθlat(multitf3_3.timestamps,
                 bounds=def_bounds, constraints=def_constraint,
                 x0=def_x0, τ=lat1), axis=0)

# %%
# %%time

a = hl.τtsi(multitf3_1.timestamps[0][0], 2)

# %%
# %%time

a = hl.τtsi(multitf3_2.timestamps[0][0], 2)

# %%
# %%time

a = hl.τtsi(multitf3_3.timestamps[0][0], 2)

# %% [markdown]
# ## Multidimensional Hawkes with latency

# %%
α11 = 0.6
β11 = 1.8
α12 = 0.4
β12 = 1.8
α21 = 0.6
β21 = 2.0
α22 = 0.4
β22 = 2.0
λ01 = 0.6
λ02 = 0.2


# %%
def ek11(t):
    return α11 * np.exp(-β11 * t)
def ek12(t):
    return α12 * np.exp(-β12 * t)
def ek21(t):
    return α21 * np.exp(-β21 * t)
def ek22(t):
    return α22 * np.exp(-β22 * t)


# %%
sup11 = 10
sup12 = 10
sup21 = 10
sup22 = 10

# %%
ekc11 = time_funclat(ek11, sup11, lat2)
ekc12 = time_funclat(ek12, sup12, lat2)
ekc21 = time_funclat(ek21, sup21, lat2)
ekc22 = time_funclat(ek22, sup22, lat2)

# %%
ektf11 = HawkesKernelTimeFunc(ekc11)
ektf12 = HawkesKernelTimeFunc(ekc12)
ektf21 = HawkesKernelTimeFunc(ekc21)
ektf22 = HawkesKernelTimeFunc(ekc22)

# %%
# plot_timefunction(ekc11)

# %%
hawkestf2x2_1 = SimuHawkes(baseline=[λ01, λ02], kernels=[[ektf11, ektf12], [ektf21, ektf22]],
                          end_time=end_time_1, verbose=False, seed=13)
hawkestf2x2_2 = SimuHawkes(baseline=[λ01, λ02], kernels=[[ektf11, ektf12], [ektf21, ektf22]],
                          end_time=end_time_2, verbose=False, seed=13)
hawkestf2x2_3 = SimuHawkes(baseline=[λ01, λ02], kernels=[[ektf11, ektf12], [ektf21, ektf22]],
                          end_time=end_time_3, verbose=False, seed=13)

# %%
n_paths_2 = 1

# %%
multitf2x2_1 = SimuHawkesMulti(hawkestf2x2_1, n_simulations=n_paths_2)
multitf2x2_2 = SimuHawkesMulti(hawkestf2x2_2, n_simulations=n_paths_2)
multitf2x2_3 = SimuHawkesMulti(hawkestf2x2_3, n_simulations=n_paths_2)

# %%
multitf2x2_1.simulate()
multitf2x2_2.simulate()
multitf2x2_3.simulate()


# %%
def latwm(ts, m, lat):
    tsm = np.insert(ts[m], 0, 0, axis=0)
    return [[np.compress(np.logical_and(tsn < tsm[i] - lat, tsn >= tsm[i - 1] - lat), tsn)
                for i in range(1, len(tsm))] for tsn in ts]


# %%
# multitf2x2_1.timestamps[0]

# %%
[len(multitf2x2_1.timestamps[0][0]), len(multitf2x2_1.timestamps[0][1])]

# %%
# latwm(multitf2x2_1.timestamps[0], 0, lat2)

# %%
# latwm(multitf2x2_1.timestamps[0], 1, lat2)

# %%
[[len(latwm(multitf2x2_1.timestamps[0], 0, lat2)[0]), len(latwm(multitf2x2_1.timestamps[0], 0, lat2)[1])],
 [len(latwm(multitf2x2_1.timestamps[0], 1, lat2)[0]), len(latwm(multitf2x2_1.timestamps[0], 1, lat2)[1])]]

# %%
[[np.array([len(a) for a in latwm(multitf2x2_1.timestamps[0], 0, lat2)[0]]),
  np.array([len(a) for a in latwm(multitf2x2_1.timestamps[0], 0, lat2)[1]])],
 [np.array([len(a) for a in latwm(multitf2x2_1.timestamps[0], 1, lat2)[0]]),
  np.array([len(a) for a in latwm(multitf2x2_1.timestamps[0], 1, lat2)[1]])]]


# %%
def lllatm(θ, ts, tsisn, M, m, lat=0):  # ts = {{}, ..., {}} length M; 1 <= m <= M
    λ0 = θ[0]  # scalar
    αm = θ[1]  # (M) array
    βm = θ[2]  # (M) array
    tsm = ts[m]
    tn = tsm[-1]
    nm = len(tsm)
    rin = np.zeros((M, nm + 1))
    for n in range(M):
        β = βm[n]
        tsis = tstsn[n]
        ebdts = np.exp(-β * np.diff(tsm))
        ri = np.zeros(nm + 1)
        for i in range(nm):
            tsi = tsis[i]
            if len(tsi) > 0:
                ri[i + 1] = ebdts[i] * ri[i] + np.sum(np.exp(-β * (tsm[i] - lat - tsi)))
            else:
                ri[i + 1] = ebdts[i] * ri[i]
        rin[n] = ri
    s1 = np.sum(1 - np.exp(-β * (tn - lat - ts[:-1]) * np.heaviside(tn - lat - ts[:-1], 0)))
    s2 = np.sum(np.log(λ0 + α * ri[1:]))
    return -(tn * (1 - λ0) - (α / β) * s1 + s2)


# %% [markdown]
# ## Estimation of slowly decreasing Hawkes kernels: application to high-frequency order book dynamics

# %%
from tick.dataset import fetch_hawkes_bund_data

# %%
timestamps_list = fetch_hawkes_bund_data()
