# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tick library - guide and testing

# %% [markdown]
# ## Python Imports

# %% Python imports
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from scipy import optimize
# from scipy.optimize import minimize
# from scipy.optimize import Bounds
from functools import partial
import timeit

# %% [markdown]
# ## Tick Imports

# %% Tick Imports
from tick.hawkes import SimuHawkes, SimuHawkesMulti
from tick.base import TimeFunction
from tick.hawkes import HawkesKernelTimeFunc, HawkesKernel0
from tick.hawkes import HawkesKernelSumExp, HawkesKernelExp
from tick.hawkes import HawkesEM, HawkesSumExpKern
# from tick.hawkes import SimuPoissonProcess, SimuInhomogeneousPoisson
# from tick.hawkes import HawkesBasisKernels, HawkesSumGaussians
from tick.plot import plot_timefunction
# from tick.plot import plot_point_process
from tick.plot import plot_hawkes_kernels
# from tick.plot import plot_basis_kernels


# %% [markdown]
# ## Theoretical expected number of events for Poisson

# %% Poisson constant intensity


# def expnpoi(lambda0, time):
#     return lambda0 * time

# %% [markdown]
# ## Theoretical expected number of events for Poisson with Exp Kernel

# %% Poisson constant intensity


# def expnpoiexpk(alpha, beta, time):
#     return (alpha / beta) * (1 - np.exp(- beta * time))

# %% [markdown]
# ## Theoretical expected number of events for Hawkes with exponential kernel

# %% Kernel defined as mu(t) = alpha*np.exp(-beta*t)


# def expnexp(lambda0, alpha, beta, time):
#     if alpha - beta == 0:
#         print('alpha=beta !')
#         return np.inf
#     else:
#         return lambda0 * time + lambda0 * alpha / ((alpha - beta)**2)\
#             * (np.exp((alpha - beta) * time) - 1 - (alpha - beta) * time)

# %% [markdown]
# ## Defining Time Functions

# %% Define Support
support = 10
t0 = 5
# n_steps = int(support/t0)*1000

# %% [markdown]
# #### Using a function

# %% Function g1
def g1(t):
    return 0.3 * np.exp(-0.8 * t)

# %% [markdown]
# Introducing a latency with heaviside

# %% Function g2 (latency)
# def g2(t):
#     return 0.7 * 5.0 * np.exp(-5.0 * (t - t0))\
#         * np.heaviside(t - t0, 1) # To ensure zero before t0

# %% Function g3
def g3(t):
    return 0.1 * np.exp(-0.5 * t)

# %% Function g4
def g4(t):
    return 20 * np.exp(-100 * t)

# %% [markdown]
# ### Creating a complete function with latency and linear interpolation

# %% Define the time_function
def time_func(f, support, t0=0, steps=100,
              inter_mode=TimeFunction.InterConstRight):
    t_values = np.linspace(0, support, steps + 1)
    y_values = f(t_values - t0) * np.heaviside(t_values - t0, 1)
    return TimeFunction(values=(t_values, y_values),
                        border_type=TimeFunction.Border0,
                        # inter_mode=TimeFunction.InterLinear,
                        inter_mode=inter_mode)

# %% Define tf_1
tf_1 = time_func(g1, support-t0, 0, 100)

# %% Define tf_2
tf_2 = time_func(g1, support, t0, 200)

# %% Define tf_3
tf_3 = time_func(g3, support-t0, 0, 100)

# %% Define tf_4
tf_4 = time_func(g4, 0.05, 0, 100)

# %% Check equivalent values
# print([[tf_1.value(0), tf_1.value(1)], [tf_2.value(t0), tf_2.value(1+t0)]])

# %% Plot tf_1
plot_timefunction(tf_1)

# %% [markdown]
# ![Tick_Figure_1_0_TF.png](attachment:7ec4d7d7-df41-4fc7-9238-6aa7396b1c67.png)

# %% Plot tf_2
plot_timefunction(tf_2)

# %% Plot tf_3
plot_timefunction(tf_3)

# %% Plot tf_4
plot_timefunction(tf_4)

# %% [markdown]
# ![Tick_Figure_2_t0_TF.png](attachment:0d61b956-0013-41ad-877d-698cc574ff6d.png)

# %% [markdown]
# ## Simulating Poisson Processes

# %% Poisson with constant intensity
# run_time = 10
# intensity = 5

# poi = SimuPoissonProcess(intensity, end_time=run_time)
# poi.simulate()
# print([poi.n_total_jumps, expnpoi(intensity, run_time)])
# poi_timestamps = poi.timestamps


# %% Poisson with exponential intensity

# poiexp = SimuInhomogeneousPoisson([tf_1], end_time=20)
# poiexp.simulate()
# print([poiexp.n_total_jumps, expnpoiexpk(0.6, 0.8, 20)])
# poiexp_timestamps = poiexp.timestamps

# %% [markdown]
# ## Defining Non-parametric Kernels

# %% Define kernel_1
kernel_1 = HawkesKernelTimeFunc(tf_1)
print(kernel_1.get_norm())

# %% Define kernel_2
kernel_2 = HawkesKernelTimeFunc(tf_2)
print(kernel_2.get_norm())

# %% Define kernel_3
kernel_3 = HawkesKernelTimeFunc(tf_3)
print(kernel_3.get_norm())

# %% Define kernel_4
kernel_4 = HawkesKernelTimeFunc(tf_4)
print(kernel_4.get_norm())

# %% [markdown]
# ## Defining Parametric Kernels

# %% Define kernel_sumexp
# kernel_sumexp = HawkesKernelSumExp(
#     intensities=np.array([0.1, 0.2, 0.1]),
#     decays=np.array([1.0, 3.0, 7.0]))

# %% [markdown]
# ## Simulations

# %% [markdown]
# ### SimuHawkes

# %% Define simulation for one realization, sumexp kernel
# hawkes = SimuHawkes(n_nodes=1, end_time=40, seed=1398)
# hawkes.set_kernel(0, 0, kernel_sumexp)
# hawkes.set_baseline(0, 1.)

# %% Track intensity and simulation
# dt = 0.01
# hawkes.track_intensity(dt)
# hawkes.simulate()

# %% Get attributes of hawkes
# timestamps = hawkes.timestamps
# intensity = hawkes.tracked_intensity
# intensity_times = hawkes.intensity_tracked_times
# mean_intensity = hawkes.mean_intensity()

# %% Plot jumps
# pd.Series(np.arange(1, len(timestamps[0])+1),
#           index=timestamps[0]).plot(drawstyle='steps-post')

# %% Plot point express
# plot_point_process(hawkes)

# %% [markdown]
# ### SimuHawkesMulti

# %% Simulate with Multi
hawkes_m1 = SimuHawkes(n_nodes=1, end_time=20000)
hawkes_m1.set_baseline(0, 1.2)
hawkes_m1.set_kernel(0, 0, kernel_1)

hawkes_m2 = SimuHawkes(n_nodes=1, end_time=20000)
hawkes_m2.set_baseline(0, 1.2)
hawkes_m2.set_kernel(0, 0, kernel_2)

hawkes_m3 = SimuHawkes(n_nodes=1, end_time=20000)
hawkes_m3.set_baseline(0, 1.2)
hawkes_m3.set_kernel(0, 0, kernel_3)

hawkes_m4 = SimuHawkes(n_nodes=1, end_time=20000)
hawkes_m4.set_baseline(0, 1.2)
hawkes_m4.set_kernel(0, 0, kernel_4)

# %% Simulate Multivariate with Multi
hawkes_mv1 = SimuHawkes(n_nodes=2, end_time=20)
hawkes_mv1.set_baseline(0, 0.1)
hawkes_mv1.set_baseline(1, 0.2)
hawkes_mv1.set_kernel(0, 0, kernel_1)
hawkes_mv1.set_kernel(0, 1, kernel_2)
hawkes_mv1.set_kernel(1, 0, kernel_2)
hawkes_mv1.set_kernel(1, 1, kernel_3)


# %% Run Multi
multi_1 = SimuHawkesMulti(hawkes_m1, n_simulations=1)
multi_1.simulate()

multi_2 = SimuHawkesMulti(hawkes_m2, n_simulations=1)
multi_2.simulate()

multi_3 = SimuHawkesMulti(hawkes_m3, n_simulations=1)
multi_3.simulate()

multi_4 = SimuHawkesMulti(hawkes_m4, n_simulations=1)
multi_4.simulate()

# %% Run Multi for Multivariate
multi_v1 = SimuHawkesMulti(hawkes_mv1, n_simulations=1)
multi_v1.simulate()

# %% Get attributes from Multi
multi_1_timestamps = multi_1.timestamps
multi_1_mean_intensity = np.mean(np.array(multi_1.mean_intensity))
n_points_1 = np.array([len(t[0]) for t in multi_1_timestamps])

multi_2_timestamps = multi_2.timestamps
multi_2_mean_intensity = np.mean(np.array(multi_2.mean_intensity))
n_points_2 = np.array([len(t[0]) for t in multi_2_timestamps])

multi_3_timestamps = multi_3.timestamps
multi_3_mean_intensity = np.mean(np.array(multi_3.mean_intensity))
n_points_3 = np.array([len(t[0]) for t in multi_3_timestamps])

multi_4_timestamps = multi_4.timestamps
multi_4_mean_intensity = np.mean(np.array(multi_4.mean_intensity))
n_points_4 = np.array([len(t[0]) for t in multi_4_timestamps])

print([np.mean(n_points_1), np.mean(n_points_2), np.mean(n_points_3)])

# print([np.mean(first_point_1), np.mean(first_point_2)])

# %% Get attributes from Multi - Multivariate
multi_v1_timestamps = multi_v1.timestamps
multi_v1_mean_intensity = np.mean(np.array(multi_v1.mean_intensity))
n_points_v1 = np.array([len(t) for t in multi_v1_timestamps[0]])


# %% [markdown]
# ## Learning

# %% [markdown]
# #### Non-parametric

# %% [markdown]
# ##### HawkesEM

# %% HawkesEM - Discretization

kern_d =\
    np.concatenate(
        (np.array([0., 0.2*t0,  0.5*t0, 0.75*t0, 0.9*t0, t0, 1.25*t0, 1.5*t0]),
         np.array([0.02, 0.05, 0.075, 0.1, 0.2, 0.5, 0.75, 1., 2.,
                   support])))

# %% HawkesEM - Learner Kernel 1

em_1 = HawkesEM(kernel_discretization=kern_d, max_iter=10000, tol=1e-5,
                verbose=True, n_threads=-1)
em_1.fit(multi_1_timestamps)
em_1_baseline = em_1.baseline
em_1_kernel = em_1.kernel
em_1_score = em_1.score()

# %% HawkesEM - Learner Kernel 2

em_2 = HawkesEM(kernel_discretization=kern_d, max_iter=10000, tol=1e-5,
                verbose=True, n_threads=-1)
em_2.fit(multi_2_timestamps)
em_2_baseline = em_2.baseline
em_2_kernel = em_2.kernel
em_2_score = em_2.score()

# %% Show head of kernels - HawkesEM

print(pd.DataFrame(
    {'0': em_1_kernel[0, 0],
     't0': em_2_kernel[0, 0]}).head(10))

# %% Plot HawkesEM 1
plot_hawkes_kernels(em_1, hawkes=hawkes_m1)

# %% [markdown]
# ![Tick_Figure_3_0_EM_fit.png](attachment:e79e4b3e-595b-42bd-b622-748c42b42725.png)

# %% Plot HawkesEM 1 - Log
plot_hawkes_kernels(em_1, hawkes=hawkes_m1, log_scale=True)

# %% [markdown]
# ![Tick_Figure_4_0_EM_fit_log.png](attachment:ea3e1e50-2b8d-4856-819e-ed95fdcc4425.png)

# %% Plot HawkesEM 2
plot_hawkes_kernels(em_2, hawkes=hawkes_m2)

# %% [markdown]
# ![Tick_Figure_5_t0_EM_fit.png](attachment:634ad445-9c32-4bc2-aaa6-bb76c5913b0c.png)

# %% Plot HawkesEM 2 - Log
plot_hawkes_kernels(em_2, hawkes=hawkes_m2, log_scale=True)

# %% [markdown]
# ![Tick_Figure_6_t0_EM_fit_log.png](attachment:b85e8868-40a8-41c0-a99d-7506f0da7805.png)

# %% [markdown]
# ##### HawkesBasisKernels

# %% HawkesBasisKernels

# bk_1 = HawkesBasisKernels(kernel_support=4, n_basis=1, kernel_size=100,
#                           max_iter=10000, tol=1e-5, C=0.1, verbose=True,
#                           n_threads=-1)
# bk_1.fit(multi_1_timestamps)
# bk_1_baseline = bk_1.baseline
# bk_1_amplitudes = bk_1.amplitudes
# bk_1_kernel = bk_1.basis_kernels

# bk_2 = HawkesBasisKernels(kernel_support=4, n_basis=1, kernel_size=100,
#                           max_iter=10000, tol=1e-5, C=0.1, verbose=True,
#                           n_threads=-1)
# bk_2.fit(multi_2_timestamps)
# bk_2_baseline = bk_2.baseline
# bk_2_amplitudes = bk_2.amplitudes
# bk_2_kernel = bk_2.basis_kernels

# %% Plot HawkesBasisKernels 1
# plot_hawkes_kernels(bk_1, hawkes=hawkes_m1)

# %% Plot HawkesBasisKernels 2
# plot_hawkes_kernels(bk_2, hawkes=hawkes_m2)

# %% Plot HawkesBasisKernels 2
# plot_basis_kernels(bk, basis_kernels=[g1, g2])

# %% [markdown]
# #### Parametric

# %% [markdown]
# ##### HawkesSumExpKern

# %% HawkesSumExpKern - Learner Kernel 1

# sek_1 = HawkesSumExpKern(decays=[5.],
#                          n_baselines=1, penalty='l2', solver='agd',
#                          elastic_net_ratio=0.8,
#                          max_iter=10000, tol=1e-5, C=1000., verbose=True)
# sek_1.fit(multi_1_timestamps)
# sek_1_baseline = sek_1.baseline
# sek_1_adjacency = sek_1.adjacency
# sek_1_score = sek_1.score()

# %% HawkesSumExpKern - Learner Kernel 2

# sek_2 = HawkesSumExpKern(decays=[5.],
#                          n_baselines=1, penalty='l2', solver='agd',
#                          elastic_net_ratio=0.8,
#                          max_iter=10000, tol=1e-5, C=1000., verbose=True)
# sek_2.fit(multi_2_timestamps)
# sek_2_baseline = sek_2.baseline
# sek_2_adjacency = sek_2.adjacency
# sek_2_score = sek_2.score()

# %% Plot HawkesSumExpKern 1
# plot_hawkes_kernels(sek_1, hawkes=hawkes_m1)

# %% [markdown]
# ![Tick_Figure_7_0_SE_fit.png](attachment:46264624-15de-49e1-9d0d-632e44d60fb6.png)

# %% Plot HawkesSumExpKern 1 - Log
# plot_hawkes_kernels(sek_1, hawkes=hawkes_m1, log_scale=True)

# %% [markdown]
# ![Tick_Figure_8_0_SE_fit_log.png](attachment:25b0a2f6-0c60-4b19-a8b5-f5bc6fe04f08.png)

# %% Plot HawkesSumExpKern 2
# plot_hawkes_kernels(sek_2, hawkes=hawkes_m2)

# %% [markdown]
# ![Tick_Figure_9_0t_SE_fit.png](attachment:92cb48a1-1b40-400e-a1ba-0957d7d3726d.png)

# %% Plot HawkesSumExpKern 2 - Log
# plot_hawkes_kernels(sek_2, hawkes=hawkes_m2, log_scale=True)

# %% [markdown]
# ![Tick_Figure_10_t0_SE_fit_log.png](attachment:c15dc720-3a2e-4419-9879-a7e37cf4f00c.png)

# %% [markdown]
# ##### HawkesSumGaussians

# %% HawkesSumGaussians
# sg_1 = HawkesSumGaussians(max_mean_gaussian=4, n_gaussians=7,
#                         lasso_grouplasso_ratio=0.5,
#                         max_iter=10000, tol=1e-5, C=1000., verbose=True)
# sg_1.fit(multi_1_timestamps)
# sg_1_baseline = sg_1.baseline
# sg_1_amplitudes = sg_1.amplitudes
# sg_1_means = sg_1.means_gaussians
# sg_1_std = sg_1.std_gaussian

# sg_2 = HawkesSumGaussians(max_mean_gaussian=4, n_gaussians=7,
#                         lasso_grouplasso_ratio=0.5,
#                         max_iter=10000, tol=1e-5, C=1000., verbose=True)
# sg_2.fit(multi_2_timestamps)
# sg_2_baseline = sg_2.baseline
# sg_2_amplitudes = sg_2.amplitudes
# sg_2_means = sg_2.means_gaussians
# sg_2_std = sg_2.std_gaussian

# %% Plot HawkesSumGaussians
# plot_hawkes_kernels(sg, hawkes=hawkes_m)

# %% Log-Likelihood Exp Kernel

def arec(lvav, lambda0, alpha, beta, dti):
    newav = np.exp(- beta * dti) * (1 + lvav[1])
    return [lvav[0] + np.log(lambda0 + alpha * newav), newav]

def loglk1(lambda0, alpha, beta, tmstmp):
    tdiff = np.diff(tmstmp)
    ans = [np.log(lambda0), 0]
    for t in tdiff:
        ans = arec(ans, lambda0, alpha, beta, t)
    return ans[0]

def loglk2(lambda0, alpha, beta, tmstmp):
    tt = tmstmp[-1]
    lt = lambda0 * tt
    return lt - (alpha / beta) * np.sum(np.exp(- beta * (tt - tmstmp[:-1]))-1)

def loglk(lambda0, alpha, beta, tmstmp):
    return loglk1(lambda0, alpha, beta, tmstmp) -\
        loglk2(lambda0, alpha, beta, tmstmp)

def loglkopt(theta, tmstmp):
    lambda0 = theta[0]
    alpha = theta[1]
    beta = theta[2]
    return -(loglk1(lambda0, alpha, beta, tmstmp) -
        loglk2(lambda0, alpha, beta, tmstmp))

# %% Log-Likelihood Exp Kernel with Latency no recursion

def fkern(lambda0, alpha, beta, ti, tmstmp, latency=0.):
    dtk = ti - latency - np.compress(tmstmp < ti - latency, tmstmp)
    return np.log(lambda0 + alpha * np.sum(np.exp(- beta * dtk)))


def loglk1lat(lambda0, alpha, beta, tmstmp, latency=0.):
    return np.sum(np.array([fkern(lambda0, alpha, beta, ti, tmstmp, latency)
                for ti in tmstmp]))

def loglk2lat(lambda0, alpha, beta, tmstmp, latency=0.):
    tt = tmstmp[-1]
    lt = lambda0 * tt
    abt = np.sum(np.exp(- beta * np.maximum(0., tt - latency - tmstmp[:-1]))-1)
    return lt - (alpha / beta) * abt

def loglklatopt(theta, tmstmp, latency=0.):
    lambda0 = theta[0]
    alpha = theta[1]
    beta = theta[2]
    return -(loglk1lat(lambda0, alpha, beta, tmstmp, latency) -
        loglk2lat(lambda0, alpha, beta, tmstmp, latency))

# %% Log-Likelihood Exp Kernel with Latency and Cutoff, no recursion

def fkerncut(lambda0, alpha, beta, ti, tmstmp, cutoff, latency=0.):
    tmstamplat = np.compress(tmstmp < ti - latency, tmstmp)
    dtk = ti - latency - np.where(tmstamplat >= ti - latency - cutoff,
                                  tmstamplat, -np.inf)
    return np.log(lambda0 + alpha * np.sum(np.exp(- beta * dtk)))


def loglk1latcut(lambda0, alpha, beta, tmstmp, cutoff, latency=0.):
    return np.sum(np.array([fkerncut(
        lambda0, alpha, beta, ti, tmstmp, cutoff, latency) for ti in tmstmp]))

def loglk2latcut(lambda0, alpha, beta, tmstmp, cutoff, latency=0.):
    tt = tmstmp[-1]
    lt = lambda0 * tt
    tmstamplat = np.compress(tmstmp[:-1] < tt - latency, tmstmp[:-1])
    abt = np.sum(np.exp(
        - beta * np.maximum(0., tt - latency -
                            np.where(tmstamplat >= tt - latency - cutoff,
                                     tmstamplat, -np.inf)))-1)
    return lt - (alpha / beta) * abt

def loglklatoptcut(theta, tmstmp, cutoff, latency=0.):
    lambda0 = theta[0]
    alpha = theta[1]
    beta = theta[2]
    return -(loglk1latcut(lambda0, alpha, beta, tmstmp, cutoff, latency) -
             loglk2latcut(lambda0, alpha, beta, tmstmp, cutoff, latency))

# %% Log-Likelihood Multivariate Exp Kernels with Latency no recursion

def fmkern(lambda0, alpha, beta, m, ti, tmstmp, latency=0.):
    dtk = ti - latency - np.compress(tmstmp < ti - latency, tmstmp)
    return np.log(lambda0 + alpha * np.sum(np.exp(- beta * dtk)))


def loglk1latm(lambda0, alpha, beta, m, tmstmp, latency=0.):
    return np.sum(np.array([fmkern(lambda0, alpha, beta, ti, tmstmp, latency)
                for ti in tmstmp]))

def loglk2latm(lambda0, alpha, beta, m, tmstmp, latency=0.):
    tt = tmstmp[-1]
    lt = lambda0 * tt
    abt = np.sum(np.exp(- beta * np.maximum(0., tt - latency - tmstmp[:-1]))-1)
    return lt - (alpha / beta) * abt

def loglklatoptm(theta, tmstmp, latency=0.):
    lambda0 = theta[0]
    alpha = theta[1]
    beta = theta[2]
    return -(loglk1lat(lambda0, alpha, beta, tmstmp, latency) -
        loglk2lat(lambda0, alpha, beta, tmstmp, latency))

# %% Test dtk

lat = 1
tmst = multi_v1_timestamps[0].copy()

dtkmij = [
    [
      [tm[i] - lat - np.compress(tj < tm[i] - lat, tj) for tj in tmst]
      for i in np.arange(len(tm))]
    for tm in tmst]

dtkmji = [
    [
     [tm[i] - lat - np.compress(tj < tm[i] - lat, tj)
      for i in np.arange(len(tm))]
     for tj in tmst]
    for tm in tmst]


lastt = np.array([tm[-1] for tm in tmst])

def llm(theta, n, dtk, ltm):
    nrg = np.arange(n)
    lambda0 = theta[:n].copy()
    alpha = theta[n:n*n+n].copy().reshape((n, n))
    beta = theta[n*n+n:].copy().reshape((n, n))
    for m in nrg:
        for i in 


# %% Log-Likelihood Exp Kernel with Latency and recursion

# def areclat(lvav, lambda0, alpha, beta, t1, t2, tmstmp, latency=0.):
#     prevt1 = np.compress(tmstmp < t1, tmstmp)
#     tw1 = np.compress(prevt1 >= t1 - latency, prevt1)
#     prevt2 = np.compress(tmstmp < t2, tmstmp)
#     tw2 = np.compress(prevt2 >= t2 - latency, prevt2)
#     dti = t1 - t2
#     nsum1 = np.sum(np.exp(- beta * (t1 - tw1)))
#     nsum2 = np.sum(np.exp(- beta * (t2 - tw2)))
#     newav = np.exp(- beta * dti) * (np.exp(beta * latency) + lvav[1]) -\
#         np.exp(beta * latency) * (lsum1 - np.exp(- beta * dti) * lsum2)
#     return [lvav[0] + np.log(lambda0 + alpha * newav), newav]


# %% Test LL

# teste0 = np.array([0.599672, 0.619958, 0.787331, 1.01128, 1.04814, 1.08816,\
#                    1.09742, 2.1432, 2.46201, 2.88413, 2.95975, 3.57213,\
#                    4.94965, 5.2049, 6.95427, 7.13965, 8.33601, 8.55164, 8.781,\
#                    8.92576, 9.41525, 10.1974, 10.2142, 10.5937, 10.6832,\
#                    11.1033, 11.2313, 11.5169, 11.5633, 11.805, 12.1914,\
#                    12.3337, 12.4833, 12.5905, 12.6663, 14.1659, 14.1986,\
#                    14.3381, 14.4392, 14.6677, 15.4212, 15.5708, 15.6997,\
#                    16.2507, 16.3233, 17.6775, 17.857, 19.1116])

# tt0 = teste0[-1]

# loglk2(1.2, 0.6, 0.8, teste0)

# %% Simulate ExpKernel

# hawkes_m1 = SimuHawkes(n_nodes=1, end_time=2000)
# hawkes_m1.set_baseline(0, 1.2)
# hawkes_m1.set_kernel(0, 0, HawkesKernelExp(0.6 / 0.8, 0.8))
# hawkes_m1.simulate()

# %% Check ExpKernel

# hawkes_m1_n_jumps = hawkes_m1.n_total_jumps
# hawkes_m1_timestamps = hawkes_m1.timestamps[0]

# %% Calculate LL

# print(loglk(1.2, 0.6, 0.8, hawkes_m1_timestamps))

# %% Calculate LL2

# print(loglklat(1.2, 0.6, 0.8, hawkes_m1_timestamps, 1))

# %% Minimize -LL: define function on timestamps

# def loglkf(x):
#     return -loglk(x[0], x[1], x[2], hawkes_m1_timestamps)

# %% Minimize -LL g1 rec

start = timeit.default_timer()
optim0 = partial(loglkopt, tmstmp=multi_4_timestamps[0][0])
print(optim0([1.2, 20., 100.]))
bounds = [(0.01, 2.0), (0.01, 100.), (0.01, 200.)]
# x0 = np.array([1., 0.3, 0.5])
# res = minimize(loglkf, x0, method='SLSQP', bounds=bounds)
res0 = optimize.shgo(optim0, bounds)
stop = timeit.default_timer()
print('Time Spent: ', round(stop - start), ' seconds')
print(res0)

# %% Minimize -LL g1

start = timeit.default_timer()
optim1 = partial(loglklatopt, tmstmp=multi_4_timestamps[0][0])
print(optim1([1.2, 20., 100.]))
bounds = [(0.01, 2.0), (0.01, 100.), (0.01, 200.)]
# x0 = np.array([1., 0.3, 0.5])
# res = minimize(loglkf, x0, method='SLSQP', bounds=bounds)
res1 = optimize.shgo(optim1, bounds)
stop = timeit.default_timer()
print('Time Spent: ', round(stop - start), ' seconds')
print(res1)

# %% Minimize -LL g1 cutoff

start = timeit.default_timer()
optim1b = partial(loglklatoptcut, tmstmp=multi_4_timestamps[0][0],
                  cutoff=1e-1)
print(optim1b([1.2, 20., 100.]))
bounds = [(0.01, 2.0), (0.01, 100.), (0.01, 200.)]
# x0 = np.array([1., 0.3, 0.5])
# res = minimize(loglkf, x0, method='SLSQP', bounds=bounds)
res1b = optimize.shgo(optim1b, bounds)
stop = timeit.default_timer()
print('Time Spent: ', round(stop - start), ' seconds')
print(res1b)

# %% Minimize -LL g2 latency=0

start = timeit.default_timer()
optim2 = partial(loglklatopt, tmstmp=multi_2_timestamps[0][0])
print(optim2([1.2, 0.6, 0.8]))
bounds = [(0.5, 2.0), (0.1, 1.5), (0.2, 1.6)]
# x0 = np.array([1., 0.3, 0.5])
# res = minimize(loglkf, x0, method='SLSQP', bounds=bounds)
res2 = optimize.shgo(optim2, bounds)
stop = timeit.default_timer()
print('Time Spent: ', round(stop - start), ' seconds')
print(res2)

# %% Minimize -LL g2 latency>0

start = timeit.default_timer()
optim3 = partial(loglklatopt, tmstmp=multi_2_timestamps[0][0], latency=5)
print(optim3([1.2, 0.6, 0.8]))
bounds = [(0.01, 2.0), (0.01, 1.5), (0.01, 1.6)]
# x0 = np.array([1., 0.3, 0.5])
# res = minimize(loglkf, x0, method='SLSQP', bounds=bounds)
res3 = optimize.shgo(optim3, bounds)
stop = timeit.default_timer()
print('Time Spent: ', round(stop - start), ' seconds')
print(res3)
