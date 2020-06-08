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
from tick.hawkes import HawkesConditionalLaw, HawkesKernelExp
from tick.hawkes import HawkesKernelTimeFunc, HawkesBasisKernels
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

sht = wb.sheets('Sheet1')

# %% Definitions

end_time = sht.range('C4').value
C = sht.range('C5').value
kernel_size = int(sht.range('C6').value)
max_iter = int(sht.range('C7').value)

# %% Simulate - functions


def g1(t):
    return (np.cos(np.pi * t / 10) + 1.1)


def g2(t):
    return (np.cos(np.pi * (t / 10 + 1)) + 1.1)


# def g1(t):
#     return 0.7 * 0.5 * np.exp(-0.5 * t)


# def g2(t):
#     return 0.5 * 0.7 * np.exp(-0.7 * t)

# %% Simulate - input


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
verbose = True

hawkes = SimuHawkes(baseline=baseline, seed=seed, verbose=verbose)
for i, j in itertools.product(range(n_f), repeat=n_f):
    u1, u2 = u_values[2 * i + j]
    print([u1, u2])
    y_values = g1(t_values) * u1 + g2(t_values) * u2
    kernel = HawkesKernelTimeFunc(t_values=t_values, y_values=y_values)
    hawkes.set_kernel(i, j, kernel)

hawkes.end_time = end_time
hawkes.threshold_negative_intensity(allow=True)
hawkes.simulate()
ticks = hawkes.timestamps

# %% Estimation definitions


kernel_support =int(sht.range('C20').value)
n_basis = int(sht.range('C21').value)
ode_tol = sht.range('C22').value

# %% Estimation


em = HawkesBasisKernels(kernel_support, n_basis=n_basis,
                        kernel_size=kernel_size, C=C, n_threads=-1,
                        max_iter=max_iter, verbose=verbose, ode_tol=ode_tol)
em.fit(ticks)

# %% Plot

fig = plot_hawkes_kernels(em, hawkes=hawkes, support=19.9, show=False)
for ax in fig.axes:
    ax.set_ylim([0, 0.025])

fig = plot_basis_kernels(em, basis_kernels=[g2, g1], show=False)
for ax in fig.axes:
    ax.set_ylim([0, 0.5])

plt.show()

# %% Attributes


print(em.n_nodes)
print(em.baseline)
print(em.amplitudes)
print(em.basis_kernels)
print(em.kernel_dt)
print(em.kernel_discretization)

