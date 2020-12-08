# %% Imports


import numpy as np
from scipy.optimize import minimize
from functools import partial
from numba import njit
from numba import types
from numba.typed import Dict, List

from tick.base import TimeFunction
from tick.hawkes import HawkesKernelTimeFunc

# %% Define the time_function


def time_funclat(f, support, lat=0, steps=100,
              inter_mode=TimeFunction.InterConstRight):
    t_values = np.linspace(0, support, steps + 1)
    y_values = f(t_values)
    if lat > 0:
        t_values_lat = np.linspace(0, lat, steps)
        y_values_lat = np.zeros(steps)
        t_values_shifted = t_values + lat
        t_all = np.concatenate((t_values_lat, t_values_shifted))
        y_all = np.concatenate((y_values_lat, y_values))
    else:
        t_all = t_values.view()
        y_all = y_values.view()
    return TimeFunction(values=(t_all, y_all),
                        border_type=TimeFunction.Border0,
                        inter_mode=inter_mode)

# %% Log-likelihood without latency


@njit
def δts0(ts):
    return ts[-1] - ts[:-1]


@njit
def ll(θ, ts, Δts, δts):
    λ0 = θ[0]
    α = θ[1]
    β = θ[2]
    tn = ts[-1]
    ebdts = np.exp(-β * Δts)
    ri = np.zeros(len(ts))
    for i in range(1, len(ts)):
        ri[i] = ebdts[i - 1] * (1 + ri[i - 1])
    s1 = np.sum(1 - np.exp(-β * δts))
    s2 = np.sum(np.log(λ0 + α * ri))
    return -(tn * (1 - λ0) - (α / β) * s1 + s2)


def findθ(tss, bounds, constraints, x0):
    results = []
    for path in tss:
        ts = path[0]
        Δts = np.diff(ts)
        δts = δts0(ts)
        optim0 = partial(ll, ts=ts, Δts=Δts, δts=δts)
        res0 = minimize(optim0, x0, method='SLSQP',
                        bounds=bounds, constraints=constraints)
        results = results + [res0.x]
    return np.array(results)


@njit
def lls(θ, tsM, M, m, Δts, δts):
    # tsM = {{}, ..., {}} length M; 1 <= m <= M
    λ0 = θ[0]  # scalar
    αm = θ[1]  # (M) array
    βm = θ[2]  # (M) array
    ts = tsM[m]
    tn = ts[-1]
    nm = len(ts)
    rin = np.zeros((M, nm + 1))
    for n in range(M):
        β = βm[n]
        ebdts = np.exp(-β * Δts)
        for i in range(1, len(ts)):
            rin[i] = ebdts[i - 1] * (1 + rin[i - 1])
    s1 = np.sum(1 - np.exp(-β * δts))
    s2 = np.sum(np.log(λ0 + α * ri))
    return -(tn * (1 - λ0) - (α / β) * s1 + s2)


# %% Log-likelihood with latency


# # No recursion
# def lllatnr(θ, ts, dtsns, lat=0):
#     λ0 = θ[0]
#     α = θ[1]
#     β = θ[2]
#     tn = ts[-1]
#     ri = np.zeros(len(ts))
#     for i in range(1, len(ts)):
#         ri[i] = np.sum(np.exp(-β * (ts[i] - lat - ts[:i])) * np.heaviside(tn - lat - ts[:i], 0))
#     s1 = np.sum(1 - np.exp(-β * dtsns))
#     s2 = np.sum(np.log(λ0 + α * ri))
#     return -(tn * (1 - λ0) - (α / β) * s1 + s2)


# # No recursion
# def findθlatnr(tss, lat=0, bounds=def_bounds, x0=def_x0):
#     results = []
#     for path in tss:
#         ts = path[0]
#         dtsns = dtsn(ts, lat)
#         optim0 = partial(lllatnr, ts=ts, dtsns=dtsns, lat=lat)
#         res0 = minimize(optim0, x0, method='SLSQP', bounds=def_bounds)
#         results = results + [res0.x]
#     return np.array(results)


@njit
def findtlat(ts, τ, i):
    j = np.searchsorted(ts[i + 1:] - τ - ts[i], 0) + i + 1
    if j <= len(ts) - 1:
        return [j, ts[j] - τ - ts[i]]
    else:
        return [j, 0.]


# @njit
def τtsi(ts, τ):
    # dtsis = dict()
    dtsis = Dict.empty(
        key_type=types.int64,
        value_type=types.float64[:],
    )
    for i in range(0, len(ts) - 1):
        k, v = findtlat(ts, τ, i)
        if v > 0:
            if int(k) not in dtsis:
                # dtsis[int(k)] = [v]
                dtsis[int(k)] = np.array([v])
            else:
                # dtsis[k].extend([v])
                dtsis[int(k)] = np.append(dtsis[int(k)], v)
    for i in range(0, len(ts)):
        if i not in dtsis:
            dtsis[i] = np.array([np.inf])
    return dtsis
    # return [np.array(dtsis.get(i, np.inf)) for i in range(1, len(ts))]


# @njit
# def τtsj(ts, τ):
#     dtsis = Dict.empty(
#         key_type=types.int64,
#         value_type=types.float64[:],
#     )
#     for j in range(0, len(ts) - 1):
#         k = np.searchsorted(ts - τ - ts[j], 0)
#         if k <= len(ts) - 1:
#             v = ts[k] - τ - ts[j]
#             if int(k) not in dtsis:
#                 dtsis[int(k)] = np.array([v])
#             else:
#                 dtsis[int(k)] = np.append(dtsis[int(k)], v)
#     for i in range(0, len(ts)):
#         if i not in dtsis:
#             dtsis[i] = np.array([np.inf])
#     return dtsis


def heav(ts, τ):
    return np.heaviside(ts[-1] - τ - ts[:-1], 0)


def δtsτ(ts, τ):
    return (ts[-1] - τ - ts[:-1]) * heav(ts, τ)


@njit
def lllat(θ, ts, Δts, δts, τts):
    λ0 = θ[0]
    α = θ[1]
    β = θ[2]
    tn = ts[-1]
    ebdts = np.exp(-β * Δts)
    ri = np.zeros(len(ts))
    for i in range(1, len(ts)):
        # ri[i] = ebdts[i - 1] * ri[i - 1] + np.sum(np.exp(-β * τts[i - 1]))
        ri[i] = ebdts[i - 1] * ri[i - 1] + np.sum(np.exp(-β * τts[i]))
    s1 = np.sum(1 - np.exp(-β * δts))
    s2 = np.sum(np.log(λ0 + α * ri))
    return -(tn * (1 - λ0) - (α / β) * s1 + s2)


def findθlat(tss, bounds, constraints, x0, τ=0):
    results = []
    for path in tss:
        ts = path[0]
        Δts = np.diff(ts)
        δts = δtsτ(ts, τ)
        τts = τtsi(ts, τ)
        optim0 = partial(lllat, ts=ts, Δts=Δts, δts=δts, τts=τts)
        res0 = minimize(optim0, x0, method='SLSQP',
                        bounds=bounds, constraints=constraints)
        results = results + [res0.x]
    return np.array(results)