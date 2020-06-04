#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:44:54 2020

@author: marcoscscarreira
"""

# %% Python Imports


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xlwings as xw

# %% Import tick


from tick.hawkes import HawkesSumExpKern
from tick.hawkes import SimuHawkesMulti, SimuHawkesSumExpKernels

# %% Marcos' PATHPROJ

PATHPROJ = os.path.join(os.path.expanduser("~"), "My Papers",
                        "UZModelUncertainty")

# %% Input and Output Paths

PATHIN = PATHPROJ+'/CLOB_data/'
PATHOUT = PATHPROJ+'/UZClass/'

# %% Connect

wb = xw.Book('Hawkes.xlsx')

sht = wb.sheets('Simul')

# %% Edit content - examples

# sht.range('A1').value = [['Foo 1', 'Foo 2', 'Foo 3'], [10.0, 20.0, 30.0]]
# sht.range('A1').expand().value
# df = pd.DataFrame([[1,2], [3,4]], columns=['a', 'b'])
# sht.range('A1').value = df
# sht.range('A1').options(pd.DataFrame, expand='table').value
# df2 = sht.range('A1').options(pd.DataFrame, expand='table').value

# %% Get inputs for tick

EV_14_LBLS = ['L_B', 'C_A', 'M_A', 'I_B', 'DmI_A', 'Dm_A', 'Dc_A',
              'L_A', 'C_B', 'M_B', 'I_A', 'DmI_B', 'Dm_B', 'Dc_B']

events = EV_14_LBLS
n_ev = len(events)

decay = np.array([sht.range('C3').value])
baseline = sht.range('B6').options(pd.DataFrame, expand='table').value
adjacency = sht.range('E6').options(pd.DataFrame, expand='table').value

adj = adjacency.values.view().reshape((n_ev, n_ev, 1))
bas = baseline.values.view().reshape((n_ev))
dec = decay.view()

run_time = sht.range('V3').value

# %% Simulation functions


# Main simulation


def hawkes_sumexp_simul(adjacency, decays, baseline, end_time=run_time,
                        seed=0):
    hawkes = SimuHawkesSumExpKernels(
        adjacency=adjacency, decays=decays, baseline=baseline,
        end_time=end_time, seed=seed)
    hawkes.threshold_negative_intensity(allow=True)
    hawkes.simulate()
    return hawkes

# Join timestamps


def np_method(dummies):
    return pd.Series(
        dummies.columns[np.where(dummies.fillna(0) != 0)[1]].values,
        index=dummies.index.values, name='Event').rename_axis('t')


def events_time_series(timestamps, labels):
    return np_method(pd.concat([
        pd.DataFrame(np.full(len(timestamps[j]), labels[j]),
                     index=timestamps[j], columns=[labels[j]])
        for j in range(len(labels))], axis=1))

# Duration


def pivot_durations(ts_ev, labels, aggfunc=np.mean):
    df_ev = ts_ev.reset_index()
    df_ev['dt'] = df_ev['t'].diff()
    df_ev['Previous_Event'] = df_ev['Event'].shift(+1).values
    df_piv = pd.pivot_table(df_ev, values='dt', columns=['Previous_Event'],
                          index=['Event'], aggfunc=aggfunc, margins=True)
    return df_piv[labels].reindex(labels)

# Trades


def next_row_add(event, bid, ask):
    if (event == 'M_B'):
        return [event, bid, ask, bid, False]
    elif (event == 'M_A'):
        return [event, bid, ask, ask, False]
    elif (event == 'Dm_B'):
        return [event, bid - 1, ask, bid, False]
    elif (event == 'Dm_A'):
        return [event, bid, ask + 1, ask, False]
    elif (event == 'DmI_B'):
        return [event, bid - 1, bid, bid, False]
    elif (event == 'DmI_A'):
        return [event, ask, ask + 1, ask, False]
    elif (event == 'Dc_B'):
        return [event, bid - 1, ask, pd.NA, False]
    elif (event == 'Dc_A'):
        return [event, bid, ask + 1, pd.NA, False]
    elif (event == 'I_B'):
        if (ask - bid == 1):
            return [event, bid, ask, pd.NA, True]
        else:
            return [event, bid + 1, ask, pd.NA, False]
    elif (event == 'I_A'):
        if (ask - bid == 1):
            return [event, bid, ask, pd.NA, True]
        else:
            return [event, bid, ask - 1, pd.NA, False]
    else:
        return [event, bid, ask, pd.NA, True]


# %% Simulate


# Main simulation

seed = np.random.randint(0, 10000000)
sht.range('Y3').value = seed

hawkes_simu = hawkes_sumexp_simul(adj, dec, bas, end_time=run_time, seed=seed)

hawkes_simu_mean_int = pd.Series(hawkes_simu.mean_intensity(), index=events)
sht.range('U7').value = hawkes_simu_mean_int
sht.range('U7').options(pd.Series, expand='table').value

hawkes_simu_ts = hawkes_simu.timestamps

hawkes_simu_counts = pd.Series([len(ts) for ts in hawkes_simu_ts],
                               index=events)
sht.range('X7').value = hawkes_simu_counts
sht.range('X7').options(pd.Series, expand='table').value

print('Intensities sent to file')

# Join timestamps

ts_ev = events_time_series(hawkes_simu_ts, events)

print('Timestamps series defined')

# Duration and counts

dur_ev = pivot_durations(ts_ev, events)
sht.range('E79').value = dur_ev
sht.range('E79').options(pd.DataFrame, expand='table').value

count_ev = pivot_durations(ts_ev, events, aggfunc='count')
sht.range('E43').value = count_ev
sht.range('E43').options(pd.DataFrame, expand='table').value

print('Durations and counts sent to file')

# Trades

EV_10_LBLS = ['M_A', 'I_B', 'DmI_A', 'Dm_A', 'Dc_A',
              'M_B', 'I_A', 'DmI_B', 'Dm_B', 'Dc_B']

ts_trd = ts_ev.to_frame().query('Event != "C_A"').query('Event != "C_B"')\
    .query('Event != "L_A"').query('Event != "L_B"').squeeze()

df_trd = ts_trd.to_frame().copy()
df_trd.loc[0.] = 'Start'
df_trd = df_trd.sort_index()
df_trd['bid'] = 0
df_trd['ask'] = 1
df_trd['trade_price'] = pd.NA
df_trd['fail'] = False

for j in range(1, len(df_trd)):
    df_trd.iloc[j] = next_row_add(
        df_trd.iloc[j][0], df_trd.iloc[j-1][1], df_trd.iloc[j-1][2])

df_trd['Spread'] = df_trd['ask'] - df_trd['bid']

df_trd2 = df_trd[pd.notna(df_trd['trade_price'])]
df_trd2.loc[:, 'px_chg'] = df_trd2.loc[:, 'trade_price'].copy().diff()\
    .fillna(0)
df_trd2 = df_trd2[df_trd2['px_chg'] != 0]
df_trd2.loc[:, 'px_chg_sign'] = np.sign(df_trd2.loc[:, 'px_chg'])
df_trd2.loc[:, 'run'] = df_trd2['px_chg_sign'] ==\
    df_trd2['px_chg_sign'].shift()

sht.range('Z46').value = len(df_trd2)
sht.range('Z47').value = df_trd2['px_chg'].abs().sum()
sht.range('Z49').value = df_trd2['px_chg'].mean()
sht.range('Z51').value = (df_trd2['run'].sum()) / (2 * (len(df_trd2) -
                                                        df_trd2['run'].sum()))
sht.range('Z52').value = df_trd['Spread'].mean()
sht.range('Z53').value = df_trd['fail'].sum()

print('Spread and fails sent to file')
