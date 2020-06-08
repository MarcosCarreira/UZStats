#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:25:46 2020

@author: marcoscscarreira
"""

# %% Python Imports


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xlwings as xw
import itertools

# %% Import tick


from tick.base import TimeFunction
from tick.hawkes import SimuHawkes, SimuHawkesMulti
# from tick.hawkes import HawkesConditionalLaw, HawkesKernelExp
from tick.hawkes import HawkesKernelTimeFunc, HawkesEM
from tick.plot import plot_timefunction
from tick.plot import plot_hawkes_kernel_norms, plot_hawkes_kernels
from tick.plot import plot_point_process, plot_basis_kernels

# %% Marcos' PATHPROJ

PATHPROJ = os.path.join(os.path.expanduser("~"), "My Papers",
                        "UZModelUncertainty")

# %% Input and Output Paths

PATHIN = PATHPROJ+'/CLOB_data/'
PATHOUT = PATHPROJ+'/UZClass/'

# %% Connect

wb = xw.Book('TimeFunction.xlsx')

sht = wb.sheets('Sheet2')


# %% Simulate - functions


# def g1(t):
#     return 0.7 * 0.5 * np.exp(-0.5 * t)


# def g2(t):
#     return 0.5 * 0.7 * np.exp(-0.7 * t)


# def g1(t):
#     return 0.7 / 10 * 0.5 * 10 * np.exp(-0.5 * 10 * t)


# def g2(t):
#     return 0.5 / 10 * 0.7 * 10 * np.exp(-0.7 * 10 * t)


def g1(t):
    return 0.7 * 0.5 * 10 * np.exp(-0.5 * 10 * t)


def g2(t):
    return 0.5 * 0.7 * 10 * np.exp(-0.7 * 10 * t)

# %% Simulate - inputs and definitions


end_time = sht.range('C4').value
t_values = np.linspace(sht.range('C9').value, sht.range('C10').value,
                       int(sht.range('C11').value))
u_values = sht.range('B13:C16').value
baseline = sht.range('B18:C18').value
seed = int(sht.range('C19').value)

# %% Plot Time Function (Basis)

step = (sht.range('C10').value - sht.range('C9').value) /\
         (sht.range('C11').value)
tf_1 = TimeFunction((t_values, g1(t_values)), dt=step)
tf_2 = TimeFunction((t_values, g2(t_values)), dt=step)
time_functions = [tf_1, tf_2]
_, ax_list = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
for tf, ax in zip(time_functions, ax_list):
    plot_timefunction(tf, ax=ax)
    ax.set_xlim([sht.range('C9').value - 0.5, sht.range('C10').value + 0.5])
plt.show()

# %% Simulate

n_f = 2
n_realizations = int(sht.range('C23').value)
verbose = False

hawkes = SimuHawkes(baseline=baseline, seed=seed, verbose=verbose)
for i, j in itertools.product(range(n_f), repeat=n_f):
    u1, u2 = u_values[2 * i + j]
    print([u1, u2])
    y_values = g1(t_values) * u1 + g2(t_values) * u2
    kernel = HawkesKernelTimeFunc(t_values=t_values, y_values=y_values)
    hawkes.set_kernel(i, j, kernel)

hawkes.end_time = end_time
hawkes.threshold_negative_intensity(allow=True)
multi = SimuHawkesMulti(hawkes, n_simulations=n_realizations)
multi.simulate()
ticks = multi.timestamps

df_jumps = pd.DataFrame([[len(ts) for ts in sim] for sim in ticks],
                        columns=['1', '2'])
df_jumps.hist()

# %% Estimation definitions


kernel_size = int(sht.range('C6').value)
max_iter = int(sht.range('C7').value)
kernel_support = int(sht.range('C20').value)

tol = sht.range('C22').value

# %% Estimation

n_threads = 4

em = HawkesEM(kernel_support=kernel_support, kernel_size=kernel_size,
              n_threads=n_threads, max_iter=max_iter, verbose=True,
              tol=tol)
em.fit(ticks)

# %% Plot

fig = plot_hawkes_kernels(em, hawkes=hawkes, support=kernel_support)

# %% Attributes


print(em.n_nodes)
print(em.baseline)
print(em.n_realizations)
print(em.kernel)
