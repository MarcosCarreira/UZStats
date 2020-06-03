#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:30:54 2020

@author: marcoscscarreira
"""

# %% Python Imports

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% Import tick


from tick.hawkes import HawkesSumExpKern
from tick.hawkes import SimuHawkesMulti, SimuHawkesSumExpKernels

# %% Marcos' PATHPROJ
PATHPROJ = os.path.join(os.path.expanduser("~"), "My Papers",
                        "UZModelUncertainty")

# %% Input and Output Paths
PATHIN = PATHPROJ+'/CLOB_data/'
PATHOUT = PATHPROJ+'/UZClass/'

# %% Inputs


EV_14_LBLS = ['L_B', 'C_A', 'M_A', 'I_B', 'DmI_A', 'Dm_A', 'Dc_A',
              'L_A', 'C_B', 'M_B', 'I_A', 'DmI_B', 'Dm_B', 'Dc_B']

events = EV_14_LBLS
n_ev = len(events)

adjacency_DOL = pd.read_csv(PATHOUT+'adjacency_DOL.csv', index_col=0)
adjacency_WDO = pd.read_csv(PATHOUT+'adjacency_WDO.csv', index_col=0)
baseline_DOL = pd.read_csv(PATHOUT+'baseline_DOL.csv', index_col=0)
baseline_WDO = pd.read_csv(PATHOUT+'baseline_WDO.csv', index_col=0)

adjacency_CME1 = pd.read_csv(PATHOUT+'adjacency_CME1.csv', index_col=0)
adjacency_CME2 = pd.read_csv(PATHOUT+'adjacency_CME2.csv', index_col=0)
baseline_CME1 = pd.read_csv(PATHOUT+'baseline_CME1.csv', index_col=0)
baseline_CME2 = pd.read_csv(PATHOUT+'baseline_CME2.csv', index_col=0)

decay_DOL = np.array([43900.])
decay_WDO = np.array([38500.])

decay_CME = np.array([195000.])

run_time = int(9.25*3600)

run_time_CME = int(16*3600)

adj_DOL = adjacency_DOL.values.view().reshape((n_ev, n_ev, 1))
bas_DOL = baseline_DOL.values.view().reshape((n_ev))
dec_DOL = decay_DOL.view()

adj_WDO = adjacency_WDO.values.view().reshape((n_ev, n_ev, 1))
bas_WDO = baseline_WDO.values.view().reshape((n_ev))
dec_WDO = decay_WDO.view()

adj_CME1 = adjacency_CME1.values.view().reshape((n_ev, n_ev, 1))
bas_CME1 = baseline_CME1.values.view().reshape((n_ev))
dec_CME = decay_CME.view()

adj_CME2 = adjacency_CME2.values.view().reshape((n_ev, n_ev, 1))
bas_CME2 = baseline_CME2.values.view().reshape((n_ev))

# %% Simulation function


def hawkes_sumexp_simul(adjacency, decays, baseline, end_time=run_time,
                        seed=0):
    hawkes = SimuHawkesSumExpKernels(
        adjacency=adjacency, decays=decays, baseline=baseline,
        end_time=end_time, seed=seed)
    hawkes.threshold_negative_intensity(allow=True)
    hawkes.simulate()
    return hawkes

# %% Simulate


hawkes_simu_DOL = hawkes_sumexp_simul(adj_DOL, dec_DOL, bas_DOL,
                                      end_time=run_time, seed=1234)
hawkes_simu_WDO = hawkes_sumexp_simul(adj_WDO, dec_WDO, bas_WDO,
                                      end_time=run_time, seed=1234)

hawkes_simu_CME1 = hawkes_sumexp_simul(adj_CME1, dec_CME, bas_CME1,
                                       end_time=run_time_CME, seed=1234)
hawkes_simu_CME2 = hawkes_sumexp_simul(adj_CME2, dec_CME, bas_CME2,
                                       end_time=run_time_CME, seed=1234)


hawkes_simu_DOL_mean_int = pd.Series(hawkes_simu_DOL.mean_intensity(),
                                     index=events)
hawkes_simu_WDO_mean_int = pd.Series(hawkes_simu_WDO.mean_intensity(),
                                     index=events)

hawkes_simu_CME1_mean_int = pd.Series(hawkes_simu_CME1.mean_intensity(),
                                      index=events)
hawkes_simu_CME2_mean_int = pd.Series(hawkes_simu_CME2.mean_intensity(),
                                      index=events)

hawkes_simu_DOL_ts = hawkes_simu_DOL.timestamps
hawkes_simu_WDO_ts = hawkes_simu_WDO.timestamps

hawkes_simu_CME1_ts = hawkes_simu_CME1.timestamps
hawkes_simu_CME2_ts = hawkes_simu_CME2.timestamps

hawkes_simu_DOL_counts = pd.Series(
    [len(ts) for ts in hawkes_simu_DOL_ts], index=events)
hawkes_simu_WDO_counts = pd.Series(
    [len(ts) for ts in hawkes_simu_WDO_ts], index=events)

hawkes_simu_CME1_counts = pd.Series(
    [len(ts) for ts in hawkes_simu_CME1_ts], index=events)
hawkes_simu_CME2_counts = pd.Series(
    [len(ts) for ts in hawkes_simu_CME2_ts], index=events)

print([hawkes_simu_DOL_counts.sum(), hawkes_simu_WDO_counts.sum()])

print([hawkes_simu_CME1_counts.sum(), hawkes_simu_CME2_counts.sum()])

# %% Join timestamps functions


def np_method(dummies):
    return pd.Series(
        dummies.columns[np.where(dummies.fillna(0) != 0)[1]].values,
        index=dummies.index.values, name='Event').rename_axis('t')

def events_time_series(timestamps, labels):
    return np_method(pd.concat([
        pd.DataFrame(np.full(len(timestamps[j]), labels[j]),
                     index=timestamps[j], columns=[labels[j]])
        for j in range(len(labels))], axis=1))

# %% Join timestamps


ts_ev_DOL = events_time_series(hawkes_simu_DOL_ts, events)
ts_ev_WDO = events_time_series(hawkes_simu_WDO_ts, events)

ts_ev_CME1 = events_time_series(hawkes_simu_CME1_ts, events)
ts_ev_CME2 = events_time_series(hawkes_simu_CME2_ts, events)

# %% Duration functions

def pivot_durations(ts_ev, labels, aggfunc=np.mean):
    df_ev = ts_ev.reset_index()
    df_ev['dt'] = df_ev['t'].diff()
    df_ev['Previous_Event'] = df_ev['Event'].shift(+1).values
    df_piv = pd.pivot_table(df_ev, values='dt', index=['Previous_Event'],
                          columns=['Event'], aggfunc=aggfunc, margins=True)
    return df_piv[labels].reindex(labels)

# %% Duration and counts

dur_ev_DOL = pivot_durations(ts_ev_DOL, events)
dur_ev_WDO = pivot_durations(ts_ev_WDO, events)

count_ev_DOL = pivot_durations(ts_ev_DOL, events, 'count')
count_ev_WDO = pivot_durations(ts_ev_WDO, events, 'count')

def q10(array):
    return np.quantile(array, 0.1)
def q30(array):
    return np.quantile(array, 0.3)
def q70(array):
    return np.quantile(array, 0.7)
def q90(array):
    return np.quantile(array, 0.9)


dur_ev_DOL_min = pivot_durations(ts_ev_DOL, events, np.min)
dur_ev_DOL_q10 = pivot_durations(ts_ev_DOL, events, q10)
dur_ev_DOL_q30 = pivot_durations(ts_ev_DOL, events, q30)
dur_ev_DOL_q50 = pivot_durations(ts_ev_DOL, events, np.median)
dur_ev_DOL_q70 = pivot_durations(ts_ev_DOL, events, q70)
dur_ev_DOL_q90 = pivot_durations(ts_ev_DOL, events, q90)
