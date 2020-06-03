# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Testing Florian's Excellent Code

# %% Python imports
import os
# import timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

# print(os.getcwd())

# %% Armada Class imports
# from armadaClassMarcos import ArmadaData_UZModel as uz
from armadaClassMarcos import Armada_Data as ad
# from armadaClassMarcos import Armada_TOB as atob

# %% Othmane imports

# import Plotting as pltg

# %% Marcos imports

# import armadauzdf as mcsc

# %% Tick Imports

from tick.hawkes import HawkesConditionalLaw, HawkesSumExpKern
from tick.plot import plot_hawkes_kernel_norms, plot_hawkes_kernels

# %% Pandas Options
# pd.set_option('mode.chained_assignment', None)
pd.options.display.max_columns = 30
pd.options.display.max_rows = 50

# %% Florian's PATHPROJ
# PATHPROJ = os.path.join(os.path.expanduser("~"), "Documents", "GitHub",\
#                        "UZStats")

# %% Marcos' PATHPROJ
PATHPROJ = os.path.join(os.path.expanduser("~"), "My Papers",
                        "UZModelUncertainty")

# %% Input and Output Paths
PATHIN = PATHPROJ+'/CLOB_data/'
PATHOUT = PATHPROJ+'/UZClass/'

# %% CME Constants
TS = 0.5
MOSCME = 1
MINDTCME = 0.000000001
START_TIME = pd.to_timedelta('00:00:00')
END_TIME = pd.to_timedelta('23:59:59')
EVENT_WINDOW = 1000

# %% BMF Constants
TS1 = 0.5
MOSDOL = 5
MOSWDO = 1
MINDT1 = 0.001
START_TIME1 = pd.to_timedelta('09:00:00')
END_TIME1 = pd.to_timedelta('18:15:00')
EVENT_WINDOW1 = 1000

# %% BMF file names
FILE_BMF1 = 'DOLG1720170119.csv'
FILE_BMF2 = 'WDOG1720170119.csv'

# %% CME file names
FILE1 = '20180105_6EH8.zip'
FILE2 = '20180104_6EH8.zip'

# %% Save files when running the examples
SAVECME = False
SAVEBMF = False

# %% [markdown]
# Start function here

# %% New init for Armada TOB - Part 1


# Outputs a clean df with trades collapsed by price and Level1 changes only

def init1(pathin, pathout, file_name, tick_value, min_order_size, start_time,
          end_time, file_type='CME', min_dt=MINDTCME, save_files=False):
    data = ad(pathin, file_name, file_type)  # ad=ArmadaData
    # Select times
    datadf = data.select_times(pd.to_timedelta(start_time),
                               pd.to_timedelta(end_time)).df
    # Add order flags and count
    datadf['OrderQ'] = datadf['trade_price'].isnull().copy()
    datadf['OrderN'] = datadf['OrderQ'].copy().cumsum()
    datadf['last_trade'] = datadf['trade_price'].copy().fillna(method='ffill')
    # Group trades by time and price
    datadfg = datadf.groupby(['OrderN', 'DateTime', 'OrderQ', 'last_trade'],
                             sort=False).sum(min_count=1)
    # Reset columns from groupby
    datadfg = datadfg.reset_index()
    datadfg['trade_price'] = np.where(datadfg['OrderQ'], np.nan,
                                      datadfg['last_trade'].copy())
    datadfg = datadfg.drop(columns=['OrderN', 'last_trade'])
    # Clear trades without book update or sweep not instantaneous - function
    def clear_invalid_trades(df, dt=min_dt):
        dfc = df.copy()
        dfc['Prev_Trade'] = (~dfc['OrderQ']).shift()
        dfc['Signif_dt'] = dfc['DateTime'].diff().dt.total_seconds() > dt
        dfc['Check'] = (dfc['Prev_Trade'] & (dfc['Signif_dt'])).shift(
            periods=-1, fill_value=False)
        return dfc[~dfc['Check']].copy()\
            .drop(['Prev_Trade', 'Signif_dt', 'Check'], axis=1)
    # Clear trades without book update or sweep not instantaneous
    # Ideally a fixed point iteration, but let's run it 3 times for now
    datadfg = clear_invalid_trades(datadfg)
    datadfg = clear_invalid_trades(datadfg)
    datadfg = clear_invalid_trades(datadfg)
    # Levels 1 and 2 diff (flag changes in each of the first two levels)
    def lvldiff(df):
        dfc = df.copy()
        dfdiff1 = dfc[['bid_1_qty', 'bid_1_price', 'ask_1_price',
                       'ask_1_qty']].copy().diff().abs()
        dfc['lvl1'] = dfdiff1.sum(axis=1) != 0
        dfprevtrade = ((~(dfc['OrderQ'].copy())).shift(fill_value=True))
        dfc['lvl2'] = (~dfc['OrderQ']) |\
            (dfc['OrderQ'] & dfprevtrade) |\
            (dfc['OrderQ'] & dfc['lvl1'])
        return dfc
    # Excluding Level 2 events
    datadfg = lvldiff(datadfg)
    datadfg = datadfg[datadfg['lvl2']]
    datadfg = datadfg.drop(['bid_2_qty', 'bid_2_ord', 'bid_2_price',
                            'bid_1_ord', 'ask_1_ord', 'ask_2_price',
                            'ask_2_ord', 'ask_2_qty', 'lvl2'], axis=1)
    datadfg['bid_traded'] = datadfg['bid_1_price'].copy().fillna(
        method='ffill') >= datadfg['trade_price']
    datadfg['ask_traded'] = datadfg['ask_1_price'].copy().fillna(
        method='ffill') <= datadfg['trade_price']
    datadfg['dt'] = datadfg['DateTime'].diff().dt.total_seconds()
    if save_files:
        datadfg.to_csv(pathout+file_name[:-4]+'_df.csv')
    return datadfg

# %% New init for Armada TOB - Part 2


# Outputs df with compressed trade information on the subsequent OB status

def init2(data_frame, pathout, file_name, tick_value, min_order_size,
          start_time, end_time, file_type='CME', min_dt=MINDTCME,
          save_files=False):
    # Define key for groupby
    datadf = data_frame.copy()
    datadf['OrderN'] = datadf['OrderQ'].copy().cumsum()
    datadf = datadf[datadf['OrderN'] > 0].copy()
    datadf['OrderN'] = datadf['OrderN']*(1-2*datadf['OrderQ'])
    # Group trades (sum qty, count of price levels traded)
    datadfg = datadf.groupby(['DateTime', 'OrderN'], sort=False)
    dfagg = datadfg.agg({'OrderQ': all, 'bid_1_qty': sum, 'bid_1_price': sum,
                         'trade_price': 'count', 'trade_qty': sum,
                         'ask_1_price': sum, 'ask_1_qty': sum, 'lvl1': any,
                         'bid_traded': any, 'ask_traded': any, 'dt': sum})
    dfagg = dfagg.reset_index()
    dfagg = dfagg.rename(columns={'trade_price': 'levels_traded',
                                  'lvl1': 'Level1Q'})
    # Some recheck for invalid trades might be needed here
    # The complement of the function below should be run fixed-point style
    # def find_invalid_trades_again(df, dt):
    # --------------------------------------------------------------------
    #     dfc = df.copy()
    #     dfc['Prev_Trade'] = (dfc['OrderN'].shift()) < 0
    #     dfc['Signif_dt'] = dfc['dt'] > dt
    #     dfc['Check'] = (dfc['Prev_Trade'] & (dfc['Signif_dt']))
    #     return dfc[dfc['Check']].copy()\
    #         .drop(['Prev_Trade', 'Signif_dt', 'Check'], axis=1)
    # --------------------------------------------------------------------
    # Push trades on next order book state
    dfagg['OrderId'] = np.abs(dfagg['OrderN']) +\
        (1 + np.sign(dfagg['OrderN']))/2
    # dfstates = dfagg.groupby(['DateTime', 'OrderId']).sum()
    dfagg2 = dfagg.groupby(['OrderId'])
    dfstates = dfagg2.agg({'DateTime': 'first', 'bid_1_qty': sum,
                           'bid_1_price': sum, 'levels_traded': sum,
                           'trade_qty': sum, 'ask_1_price': sum,
                           'ask_1_qty': sum, 'Level1Q': any, 'OrderQ': any,
                           'bid_traded': any, 'ask_traded': any, 'dt': sum})
    dfstates = dfstates.reset_index()
    # dfstates = dfstates.drop(['OrderN', 'OrderQ'], axis=1)
    # Normalize amount by the minimum order size (MOS)
    dfstates['bid_1_qty'] = dfstates['bid_1_qty']/min_order_size
    dfstates['ask_1_qty'] = dfstates['ask_1_qty']/min_order_size
    dfstates['trade_qty'] = dfstates['trade_qty']/min_order_size
    # Spread, Midprice , Microprice and Imbalance
    dfstates['Spread_Ticks'] = (dfstates['ask_1_price'] -
                                dfstates['bid_1_price']) / tick_value
    dfstates['Midprice'] = (dfstates['ask_1_price'] + dfstates['bid_1_price']
                            )/2
    dfstates['Microprice'] = (dfstates['ask_1_price'] * dfstates['bid_1_qty']
                              + dfstates['bid_1_price'] *
                              dfstates['ask_1_qty']) / \
        (dfstates['bid_1_qty'] + dfstates['ask_1_qty'])
    dfstates['Imbalance'] = dfstates['bid_1_qty'] / (dfstates['bid_1_qty'] +
                                                     dfstates['ask_1_qty']) -\
        1/2
    dfstates['Imbal_Sign'] = pd.cut(dfstates['Imbalance'],
                                    [-0.5, -0.2, +0.2, +0.5],
                                    labels=[-1, 0, 1])
    # Changes in top of book (diff)
    dfstates['bid_1_qty_diff'] = dfstates['bid_1_qty'].diff()
    dfstates['bid_1_price_diff'] = dfstates['bid_1_price'].diff()
    dfstates['ask_1_price_diff'] = dfstates['ask_1_price'].diff()
    dfstates['ask_1_qty_diff'] = dfstates['ask_1_qty'].diff()
    # PriceQ column (was there a price change?)
    dfstates['PriceQ'] = (dfstates['bid_1_price_diff'] != 0) |\
        (dfstates['ask_1_price_diff'] != 0)
    # ConsQ column (was there a comsumption of liquidity?)
    # Trades that take out levels but leave an unfilled balance: False
    dfstates['ConsQ'] = np.where(
        dfstates['PriceQ'], ~((dfstates['bid_1_price_diff'] > 0) |
                              (dfstates['ask_1_price_diff'] < 0)),
        (dfstates['bid_1_qty_diff'] < 0) | (dfstates['ask_1_qty_diff'] < 0) |
        (~dfstates['Level1Q']))
    # AskQ column (was the event on the Ask side?)
    # Trades that take out levels but leave an unfilled balance: Cons sign
    dfstates['AskQ'] = np.where(
        dfstates['Level1Q'], ((dfstates['ask_1_price_diff'] != 0) |
                              (dfstates['ask_1_qty_diff'] != 0)),
        dfstates['ask_traded'])
    dfstates.at[0, 'PriceQ'] = False
    dfstates.at[0, 'ConsQ'] = False
    dfstates.at[0, 'AskQ'] = False
    # Calculate event size
    dfstates['Event_Size_order'] = np.where(
        dfstates['AskQ'],
        np.where(dfstates['ask_1_price_diff'] != 0,
                  dfstates['ask_1_qty'],
                  np.abs(dfstates['ask_1_qty_diff'])),
        np.where(dfstates['bid_1_price_diff'] != 0,
                  dfstates['bid_1_qty'],
                  np.abs(dfstates['bid_1_qty_diff'])))
    dfstates['Event_Size'] = np.where(
        dfstates['trade_qty'] > 0, dfstates['trade_qty'],
        dfstates['Event_Size_order'])
    dfstates['Event_Size'] = dfstates['Event_Size'].fillna(1)
    # Classify state
    dfstates['event_code'] =\
        dfstates['AskQ'] * 8 + dfstates['ConsQ'] * 4 +\
        dfstates['Level1Q'] * 2 + dfstates['PriceQ'] * 1
    event_dict = {
        0: 'Start', 1: 'PLb', 2: 'Lb', 3: 'Pb+', 4: 'Mb', 5: 'PbM-', 6: 'Cb',
        7: 'PbC-', 8: 'Start', 9: 'PLa', 10: 'La', 11: 'Pa-', 12: 'Ma',
        13: 'PaM+', 14: 'Ca', 15: 'PaC+'}
    dfstates['Event_detail'] = dfstates['event_code'].map(event_dict)
    dfstates['Event_detail_Prev'] = dfstates['Event_detail'].copy().shift()\
        .fillna('La')
    event_dict_14 = {
        0: 'L_B', 1: 'DmI_B', 2: 'L_B', 3: 'I_B', 4: 'M_B', 5: 'Dm_B',
        6: 'C_B', 7: 'Dc_B', 8: 'L_A', 9: 'DmI_A', 10: 'L_A', 11: 'I_A',
        12: 'M_A', 13: 'Dm_A', 14: 'C_A', 15: 'Dc_A'}
    dfstates['Event_14'] = dfstates['event_code'].map(event_dict_14)
    # event_dict_CLM = {
    #     0: 'La', 1: 'Lb', 2: 'Lb', 3: 'Lb', 4: 'Mb', 5: 'Mb', 6: 'Cb',
    #     7: 'Cb', 8: 'Lb', 9: 'La', 10: 'La', 11: 'La', 12: 'Ma',
    #     13: 'Ma', 14: 'Ca', 15: 'Ca'}
    # dfstates['Event_CLM'] = dfstates['event_code'].map(event_dict_CLM)
    # dfstates['Event_CLM_Prev'] = dfstates['Event_CLM'].copy().shift()\
    #     .fillna('La')
    event_dict_consec = {
        'Ca': False, 'Cb': True, 'La': True, 'Lb': False, 'Ma': False,
        'Mb': True, 'PLa': False, 'PLb': True, 'Pa-': True, 'PaC+': False,
        'PaM+': False, 'Pb+': False, 'PbC-': True, 'PbM-': True}
    dfstates['Reversion'] = dfstates['Event_detail'].map(event_dict_consec)\
        ^ dfstates['Event_detail_Prev'].map(event_dict_consec)
    # dfstates['event_code_short'] =\
    #     dfstates['AskQ'] * 2 + dfstates['ConsQ'] * 1
    # event_dict_short = {0: 'Ib', 1: 'Cb', 2: 'Ia', 3: 'Ca'}
    # dfstates['Event'] = dfstates['event_code_short'].map(event_dict_short)
    if save_files:
        dfstates.to_csv(pathout+file_name[:-4]+'_df_states.csv')
    cols_output1 =\
        ['DateTime', 'OrderId', 'bid_1_qty', 'bid_1_price', 'ask_1_price',
         'ask_1_qty', 'trade_qty', 'levels_traded', 'AskQ', 'ConsQ',
         'Level1Q', 'PriceQ', 'Event_detail', 'Event_detail_Prev', 'Event_14',
         'Reversion', 'dt', 'Spread_Ticks', 'Midprice', 'Microprice',
         'Imbalance', 'Imbal_Sign', 'Event_Size']
    dfstates = dfstates[cols_output1]
    return dfstates


# %% Run all inits


def initall(pathin, pathout, file_name, tick_value, min_order_size,
            start_time, end_time, file_type='CME', min_dt=MINDTCME,
            save_files=False):
    dfinit1 = init1(pathin, pathout, file_name, tick_value, min_order_size,
                    start_time, end_time, file_type, min_dt, save_files)
    dfinit2 = init2(dfinit1, pathout, file_name, tick_value, min_order_size,
                    start_time, end_time, file_type, min_dt, save_files)
    return dfinit2

# %% [markdown]
# End function here

# %% Test init functions


dfDOL = initall(PATHIN, PATHOUT, FILE_BMF1, TS1, MOSDOL, START_TIME1,
                END_TIME1, 'BMF', MINDT1, SAVEBMF)
dfWDO = initall(PATHIN, PATHOUT, FILE_BMF2, TS1, MOSWDO, START_TIME1,
                END_TIME1, 'BMF', MINDT1, SAVEBMF)

dfCME1 = initall(PATHIN, PATHOUT, FILE1, TS, MOSCME, START_TIME, END_TIME,
                  'CME', MINDTCME, SAVECME)
dfCME2 = initall(PATHIN, PATHOUT, FILE2, TS, MOSCME, START_TIME, END_TIME,
                  'CME', MINDTCME, SAVECME)

# %% Find Starts


# dfDOL[dfDOL['Event'] == 'Start']

# %% Eevent Sizes

dfDOL['Event_Size'].describe()
dfWDO['Event_Size'].describe()
dfCME1['Event_Size'].describe()
dfCME2['Event_Size'].describe()

def q10(array):
    return np.quantile(array, 0.1)
def q30(array):
    return np.quantile(array, 0.3)
def q70(array):
    return np.quantile(array, 0.7)
def q90(array):
    return np.quantile(array, 0.9)

dfDOL_ES = pd.pivot_table(dfDOL, 'Event_Size', index='Event_14',
                          aggfunc=[np.mean, q10, q30, np.median, q70, q90])
dfWDO_ES = pd.pivot_table(dfWDO, 'Event_Size', index='Event_14',
                          aggfunc=[np.mean, q10, q30, np.median, q70, q90])

dfDOL[dfDOL['Event_14'] == 'DmI_A'][['bid_1_qty', 'ask_1_qty', 'trade_qty']]\
    .describe()
dfDOL[dfDOL['Event_14'] == 'DmI_B'][['bid_1_qty', 'ask_1_qty', 'trade_qty']]\
    .describe()
    
dfCME1_ES = pd.pivot_table(dfCME1, 'Event_Size', index='Event_14',
                          aggfunc=[np.mean, q10, q30, np.median, q70, q90])
dfCME2_ES = pd.pivot_table(dfCME2, 'Event_Size', index='Event_14',
                          aggfunc=[np.mean, q10, q30, np.median, q70, q90])

# %% Intensities - pivots function


def pivots_intensities(data_frame, max_q=20, plot_q=True, title=''):
    data_framec = data_frame.copy()
    data_frame_reinf = data_framec[~data_framec['Reversion']].copy()
    data_frame_rever = data_framec[data_framec['Reversion']].copy()

    def cols_intens(sub_data_frame):
        data_framec = sub_data_frame.copy()
        data_framec['ProvA'] = data_framec['La'] + data_framec['Pa-']
        data_framec['ProvB'] = data_framec['Lb'] + data_framec['Pb+']
        data_framec['DeplA'] = data_framec['PaC+'] + data_framec['PaM+']
        data_framec['DeplB'] = data_framec['PbC-'] + data_framec['PbM-']
        data_framec['ConsA'] = data_framec['Ma'] + data_framec['Ca'] +\
            data_framec['DeplA']
        data_framec['ConsB'] = data_framec['Mb'] + data_framec['Cb'] +\
            data_framec['DeplB']
        data_framec['CancA'] = data_framec['PaC+'] + data_framec['Ca']
        data_framec['CancB'] = data_framec['PbC-'] + data_framec['Cb']
        data_framec['TradA'] = data_framec['Ma'] + data_framec['PaM+']
        data_framec['TradB'] = data_framec['Mb'] + data_framec['PbM-']
        data_framec['Provision'] = data_framec['ProvA'] +\
            data_framec['ProvB']
        data_framec['Consumption'] = data_framec['ConsA'] +\
            data_framec['ConsB']
        data_framec['Cancel'] = data_framec['CancA'] +\
            data_framec['CancB']
        data_framec['Trade'] = data_framec['TradA'] +\
            data_framec['TradB']
        return data_framec

    def create_pivots(sub_data_frame):
        sub_df_bid_count = cols_intens(pd.pivot_table(
            sub_data_frame, values='Reversion', index=['bid_1_qty'],
            columns='Event_detail', aggfunc='count', margins=False).fillna(0))
        sub_df_ask_count = cols_intens(pd.pivot_table(
            sub_data_frame, values='Reversion', index=['ask_1_qty'],
            columns='Event_detail', aggfunc='count', margins=False).fillna(0))
        sub_df_bid_dt = cols_intens(pd.pivot_table(
            sub_data_frame, values='dt', index=['bid_1_qty'],
            columns='Event_detail', aggfunc=np.sum, margins=False).fillna(0))
        sub_df_ask_dt = cols_intens(pd.pivot_table(
            sub_data_frame, values='dt', index=['ask_1_qty'],
            columns='Event_detail', aggfunc=np.sum, margins=False).fillna(0))
        return [sub_df_bid_count, sub_df_ask_count, sub_df_bid_dt,
                sub_df_ask_dt]

    df_all_bid_count, df_all_ask_count, df_all_bid_dt, df_all_ask_dt =\
        create_pivots(data_framec)
    df_rein_bid_count, df_rein_ask_count, df_rein_bid_dt, df_rein_ask_dt =\
        create_pivots(data_frame_reinf)
    df_reve_bid_count, df_reve_ask_count, df_reve_bid_dt, df_reve_ask_dt =\
        create_pivots(data_frame_rever)

    def average_bid_ask(data_frame_bid, data_frame_ask):
        cols_df = ['ProvA', 'ProvB', 'ConsA', 'ConsB', 'DeplA', 'DeplB',
                   'CancA', 'CancB', 'TradA', 'TradB', 'Provision',
                   'Consumption', 'Cancel', 'Trade']
        cols_bid = {'ProvA': 'Prov-', 'ProvB': 'Prov+', 'ConsA': 'Cons-',
                    'ConsB': 'Cons+', 'DeplA': 'Depl-', 'DeplB': 'Depl+',
                    'CancA': 'Canc-', 'CancB': 'Canc+', 'TradA': 'Trad-',
                    'TradB': 'Trad+'}
        cols_ask = {'ProvA': 'Prov+', 'ProvB': 'Prov-', 'ConsA': 'Cons+',
                    'ConsB': 'Cons-', 'DeplA': 'Depl+', 'DeplB': 'Depl-',
                    'CancA': 'Canc+', 'CancB': 'Canc-', 'TradA': 'Trad+',
                    'TradB': 'Trad-'}
        data_frame_bidc = data_frame_bid[cols_df].copy().rename(
            columns=cols_bid)
        data_frame_askc = data_frame_ask[cols_df].copy().rename(
            columns=cols_ask)
        data_frame_avg = (data_frame_bidc.add(data_frame_askc))/2
        return data_frame_avg

    df_all_count = average_bid_ask(df_all_bid_count, df_all_ask_count)
    df_all_dt = average_bid_ask(df_all_bid_dt, df_all_ask_dt)
    df_all_intens = df_all_count / df_all_dt
    df_all_dur = df_all_dt / df_all_count

    df_rein_count = average_bid_ask(df_rein_bid_count, df_rein_ask_count)
    df_rein_dt = average_bid_ask(df_rein_bid_dt, df_rein_ask_dt)
    df_rein_intens = df_rein_count / df_rein_dt
    df_rein_dur = df_rein_dt / df_rein_count

    df_reve_count = average_bid_ask(df_reve_bid_count, df_reve_ask_count)
    df_reve_dt = average_bid_ask(df_reve_bid_dt, df_reve_ask_dt)
    df_reve_intens = df_reve_count / df_reve_dt
    df_reve_dur = df_reve_dt / df_reve_count

    def plot_intensity(data_frame, cols, title_plot):
        sub_df_plot = data_frame[cols].copy().iloc[:max_q]
        sub_df_plot.plot(title=title + title_plot, figsize=(15, 10))

    if plot_q:

        # cols_cp = ['Consumption', 'Provision']
        cols_agg = ['Cancel', 'Provision', 'Trade']
        cols_all = ['Canc+', 'Prov+', 'Trad+', 'Canc-', 'Prov-', 'Trad-']

        title_count = ' - Events by queue size - count'
        # plot_intensity(df_all_count, cols=cols_cp, title_plot=title_count)
        plot_intensity(df_all_count, cols=cols_agg, title_plot=title_count)
        # plot_intensity(df_all_count, cols=cols_all, title_plot=title_count)
        title_intens = ' - Events by queue size - intensity'
        # plot_intensity(df_all_intens, cols=cols_cp, title_plot=title_intens)
        plot_intensity(df_all_intens, cols=cols_agg, title_plot=title_intens)
        # plot_intensity(df_all_intens, cols=cols_all, title_plot=title_intens)
        title_dur = ' - Events by queue size - durations'
        # plot_intensity(df_all_dur, cols=cols_cp, title_plot=title_dur)
        plot_intensity(df_all_dur, cols=cols_agg, title_plot=title_dur)
        # plot_intensity(df_all_dur, cols=cols_all, title_plot=title_dur)

        title_count_rein = ' - Events by queue size - count - Reinforce'
        # plot_intensity(df_rein_count, cols=cols_cp,
        #                title_plot=title_count_rein)
        plot_intensity(df_rein_count, cols=cols_agg,
                       title_plot=title_count_rein)
        plot_intensity(df_rein_count, cols=cols_all,
                       title_plot=title_count_rein)
        title_intens_rein = ' - Events by queue size - intensity - Reinforce'
        # plot_intensity(df_rein_intens, cols=cols_cp,
        #                title_plot=title_intens_rein)
        plot_intensity(df_rein_intens, cols=cols_agg,
                       title_plot=title_intens_rein)
        plot_intensity(df_rein_intens, cols=cols_all,
                       title_plot=title_intens_rein)
        title_dur_rein = ' - Events by queue size - durations - Reinforce'
        # plot_intensity(df_rein_dur, cols=cols_cp,
        #                title_plot=title_dur_rein)
        plot_intensity(df_rein_dur, cols=cols_agg,
                       title_plot=title_dur_rein)
        plot_intensity(df_rein_dur, cols=cols_all,
                       title_plot=title_dur_rein)

        title_count_reve = ' - Events by queue size - count - Revert'
        # plot_intensity(df_rein_count, cols=cols_cp,
        #                title_plot=title_count_reve)
        plot_intensity(df_reve_count, cols=cols_agg,
                       title_plot=title_count_reve)
        plot_intensity(df_reve_count, cols=cols_all,
                       title_plot=title_count_reve)
        title_intens_reve = ' - Events by queue size - intensity - Revert'
        # plot_intensity(df_reve_intens, cols=cols_cp,
        #                title_plot=title_intens_reve)
        plot_intensity(df_reve_intens, cols=cols_agg,
                       title_plot=title_intens_reve)
        plot_intensity(df_reve_intens, cols=cols_all,
                       title_plot=title_intens_reve)
        title_dur_reve = ' - Events by queue size - durations - Revert'
        # plot_intensity(df_reve_dur, cols=cols_cp,
        #                title_plot=title_dur_reve)
        plot_intensity(df_reve_dur, cols=cols_agg,
                       title_plot=title_dur_reve)
        plot_intensity(df_reve_dur, cols=cols_all,
                       title_plot=title_dur_reve)

# %% Intensity examples


pivots_intensities(dfDOL, 25, True, 'DOL 2017-01-19')

pivots_intensities(dfWDO, 60, True, 'WDO 2017-01-19')

pivots_intensities(dfCME1, 25, 'CME 2018-01-05')

pivots_intensities(dfCME2, 25, 'CME 2018-01-04')

# %% Functions for tick application


EV_14_LBLS = ['L_B', 'C_A', 'M_A', 'I_B', 'DmI_A', 'Dm_A', 'Dc_A',
              'L_A', 'C_B', 'M_B', 'I_A', 'DmI_B', 'Dm_B', 'Dc_B']


def get_seconds(data_frame):
    times = data_frame['DateTime'].copy()
    start = times.iloc[0]
    return (times - start).dt.total_seconds().values


def get_timestamps_from_dummies(data_frame, col):
    data_framec = data_frame.copy()
    data_framec = data_framec[data_framec[col] == 1].copy()
    return data_framec.index.values


def get_event_timestamps(data_frame, cols):
    data_framec = data_frame.copy()
    data_framec['Timestamp'] = get_seconds(data_framec)
    df_dummies = pd.get_dummies(data_framec.set_index('Timestamp')[cols])
    labels = df_dummies.columns.values
    list_values = [get_timestamps_from_dummies(df_dummies, col)
                   for col in labels]
    return [list_values, labels]


def get_event_14_timestamps(data_frame):
    data_framec = data_frame.copy()
    data_framec['Timestamp'] = get_seconds(data_framec)
    df_dummies = pd.get_dummies(data_framec.set_index('Timestamp')['Event_14'])
    df_dummies = df_dummies[EV_14_LBLS]
    labels = df_dummies.columns.values
    list_values = [get_timestamps_from_dummies(df_dummies, col)
                   for col in labels]
    return [list_values, labels]


def get_hawkes(data_frame, cols, plot=False):
    timestamps, labels = get_event_timestamps(data_frame, cols)
    hawkes_learner = HawkesConditionalLaw(
        claw_method="log", delta_lag=0.1, min_lag=5e-4, max_lag=500,
        quad_method="log", n_quad=10, min_support=1e-4, max_support=1,
        n_threads=-1)
    hawkes_learner.fit(timestamps)
    if plot:
        plot_hawkes_kernel_norms(hawkes_learner, node_names=labels)
    hbase = hawkes_learner.baseline
    hmean = hawkes_learner.mean_intensity
    hnorms = hawkes_learner.kernels_norms
    return [hbase, hmean, hbase/hmean, hnorms]


def get_hawkes_events(data_frame, symmetries1d=[], symmetries2d=[],
                      plot=False):
    timestamps, labels = get_event_14_timestamps(data_frame)
    hawkes_learner = HawkesConditionalLaw(
        claw_method="log", delta_lag=0.1, min_lag=5e-4, max_lag=500,
        quad_method="log", n_quad=10, min_support=1e-4, max_support=1,
        n_threads=-1)
    hawkes_learner.set_model(symmetries1d=symmetries1d,
                             symmetries2d=symmetries2d)
    hawkes_learner.fit(timestamps)
    if plot:
        plot_hawkes_kernel_norms(hawkes_learner, node_names=labels)
    hbase = hawkes_learner.baseline
    hmean = hawkes_learner.mean_intensity
    hnorms = hawkes_learner.kernels_norms
    return [hbase, hmean, hbase/hmean, hnorms]


def hawkes_sum_exp_14(timestamps, labels, reord_labels, decay):
    hawkes_learner = HawkesSumExpKern(decay, solver='bfgs')
    hawkes_learner.fit(timestamps)
    baseline_learner = pd.DataFrame(hawkes_learner.baseline,
                                    columns=['Baseline'],index=labels)
    baseline_learner = baseline_learner.copy().reindex(reord_labels)
    adjacency_learner = hawkes_learner.adjacency
    nodes = len(labels)
    adjacency_learner = np.reshape(adjacency_learner.view(), (nodes, nodes))
    adjacency_learner = pd.DataFrame(adjacency_learner.view(), index=labels,
                                     columns=labels)
    adjacency_learner = adjacency_learner[reord_labels].copy()\
        .reindex(reord_labels)
    return [baseline_learner, adjacency_learner]


def min_hawkes_exp(timestamps, learner, decays_init):
    return -learner(decays_init, solver='bfgs').fit(timestamps).score()


def plot_hawkes_sum_exp(timestamps, decays_range):
    scores = pd.Series([HawkesSumExpKern(np.array([decay]), solver='bfgs')\
        .fit(timestamps).score() for decay in decays_range],
                       index= decays_range)
    scores.plot(legend=False)


def plot_hawkes_sum_exp_two(timestamps1, label1, timestamps2, label2,
                            decays_range):
    scores1 = [HawkesSumExpKern(np.array([decay]), solver='bfgs')\
        .fit(timestamps1).score() for decay in decays_range]
    scores2 = [HawkesSumExpKern(np.array([decay]), solver='bfgs')\
        .fit(timestamps2).score() for decay in decays_range]
    scores = pd.DataFrame({label1: scores1, label2: scores2},
                          index=decays_range)
    scores.plot()


def find_decay(timestamps, decays_init):
    def min_hawkes_sum_exp(decays_init):
        return -HawkesSumExpKern(decays_init, solver='bfgs')\
            .fit(timestamps).score()
    return minimize(min_hawkes_sum_exp, decays_init,
                method='Nelder-Mead', options={'disp': True})

# %% Apply tick

# dfDOL_ts, dfDOL_lbls = get_event_timestamps(dfDOL.iloc[1:], 'Reversion')
# df_decays = np.arange(1, 1001, 10)
# df_scores = pd.DataFrame(
#     np.array([[decay, HawkesSumExpKern(np.array([decay, decay])).fit(dfDOL_ts)
#                 .score()] for decay in df_decays]),
#     columns=['decay', 'score']).set_index('decay')
# df_decay_max = [df_scores.idxmax()[0], df_scores.max()[0]]
# df_hawkes_learner = HawkesSumExpKern(np.array(
#     [df_decay_max[0], df_decay_max[0]]))
# df_hawkes_learner.fit(dfDOL_ts)
# baseline_learner = list(hawkes_learner.baseline)
# adjacency_learner = list(hawkes_learner.adjacency)


dfDOL_ts, dfDOL_lbls = get_event_timestamps(dfDOL.iloc[1:], 'Event_14')
dfWDO_ts, dfWDO_lbls = get_event_timestamps(dfWDO.iloc[1:], 'Event_14')

dfCME1_ts, dfCME1_lbls = get_event_timestamps(dfCME1.iloc[1:], 'Event_14')
dfCME2_ts, dfCME2_lbls = get_event_timestamps(dfCME2.iloc[1:], 'Event_14')

# plot_hawkes_sum_exp(dfDOL_ts, np.arange(1, 1001, 10))
# plot_hawkes_sum_exp(dfWDO_ts, np.arange(1, 1001, 10))

plot_hawkes_sum_exp_two(dfDOL_ts, 'DOL', dfWDO_ts, 'WDO',
                        np.arange(1, 1001, 10))

plot_hawkes_sum_exp_two(dfCME1_ts, 'CME 20180105', dfCME2_ts, 'CME 20180104',
                        np.arange(1, 100001, 1000))

plot_hawkes_sum_exp_two(dfCME1_ts, 'CME 20180105', dfCME2_ts, 'CME 20180104',
                        np.arange(1, 10001, 100))

decay_DOL = find_decay(dfDOL_ts, np.array([[100.]]))
decay_WDO = find_decay(dfWDO_ts, np.array([[100.]]))

decay_DOL.x
decay_WDO.x

decay_CME = np.array([3000])

baseline_DOL, adjacency_DOL = hawkes_sum_exp_14(dfDOL_ts, dfDOL_lbls,
                                                EV_14_LBLS, decay_DOL.x)
baseline_WDO, adjacency_WDO = hawkes_sum_exp_14(dfWDO_ts, dfWDO_lbls,
                                                EV_14_LBLS, decay_WDO.x)

baseline_CME1, adjacency_CME1 = hawkes_sum_exp_14(dfCME1_ts, dfCME1_lbls,
                                                EV_14_LBLS, decay_CME)
baseline_CME2, adjacency_CME2 = hawkes_sum_exp_14(dfCME2_ts, dfCME2_lbls,
                                                EV_14_LBLS, decay_CME)

sns.heatmap(adjacency_DOL, center=0, cmap='RdBu',
            annot=True, fmt=".2f")
sns.heatmap(adjacency_WDO, center=0, cmap='RdBu',
            annot=True, fmt=".2f")

sns.heatmap(adjacency_CME1, center=0, cmap='RdBu',
            annot=True, fmt=".2f")
sns.heatmap(adjacency_CME2, center=0, cmap='RdBu',
            annot=True, fmt=".2f")

df_baseline = pd.DataFrame({'DOL': baseline_DOL['Baseline'],
                            'WDO': baseline_WDO['Baseline']})
sns.heatmap(df_baseline, center=0, cmap='RdBu',
            annot=True, fmt=".3f")

df_baseline_CME = pd.DataFrame({'CME 20180105': baseline_CME1['Baseline'],
                            'CME 20180104': baseline_CME2['Baseline']})
sns.heatmap(df_baseline_CME, center=0, cmap='RdBu',
            annot=True, fmt=".3f")


df_baseline.to_csv(PATHOUT+'df_baseline.csv')
adjacency_DOL.to_csv(PATHOUT+'adjacency_DOL.csv')
adjacency_WDO.to_csv(PATHOUT+'adjacency_WDO.csv')
baseline_DOL.to_csv(PATHOUT+'baseline_DOL.csv')
baseline_WDO.to_csv(PATHOUT+'baseline_WDO.csv')

df_baseline_CME.to_csv(PATHOUT+'df_baseline_CME.csv')
adjacency_CME1.to_csv(PATHOUT+'adjacency_CME1.csv')
adjacency_CME2.to_csv(PATHOUT+'adjacency_CME2.csv')
baseline_CME1.to_csv(PATHOUT+'baseline_CME1.csv')
baseline_CME2.to_csv(PATHOUT+'baseline_CME2.csv')

hawkes_reversion = get_hawkes(dfDOL.iloc[1:], 'Reversion', True)
hawkes_ask = get_hawkes(dfDOL.iloc[1:], 'AskQ', True)
hawkes_cons = get_hawkes(dfDOL.iloc[1:], 'ConsQ', True)
hawkes_imbal = get_hawkes(dfDOL.iloc[1:], 'Imbal_Sign', True)

ts_14, lbls_14 = get_event_timestamps(dfDOL.iloc[1:], 'Event_12')

hawkes_event_DOL = get_hawkes_events(dfDOL.iloc[1:], plot=True)

hawkes_event_WDO = get_hawkes_events(dfWDO.iloc[1:], plot=True)

df_intens = pd.DataFrame(
    {'DOL baseline': hawkes_event_DOL[0],
     'WDO baseline': hawkes_event_WDO[0],
     'DOL mean int': hawkes_event_DOL[1],
     'WDO mean int': hawkes_event_WDO[1],
     'DOL ratios': hawkes_event_DOL[2],
     'WDO ratios': hawkes_event_WDO[2]},
    index=EV_14_LBLS)

sns.heatmap(df_intens.transpose(), cmap='YlOrRd', annot=True, fmt=".2f")

sns.heatmap(hawkes_event_DOL[3], center=0, cmap='RdBu',
            annot=True, fmt=".2f",
            xticklabels=EV_14_LBLS, yticklabels=EV_14_LBLS)

sns.heatmap(hawkes_event_WDO[3], center=0, cmap='RdBu',
            annot=True, fmt=".2f",
            xticklabels=EV_14_LBLS, yticklabels=EV_14_LBLS)


# %% Several days


FILES_DOL = [
    'DOLG1720170103.csv', 'DOLG1720170104.csv',
    'DOLG1720170105.csv', 'DOLG1720170106.csv', 'DOLG1720170109.csv',
    'DOLG1720170110.csv', 'DOLG1720170111.csv', 'DOLG1720170112.csv',
    'DOLG1720170113.csv', 'DOLG1720170116.csv', 'DOLG1720170117.csv',
    'DOLG1720170118.csv', 'DOLG1720170119.csv', 'DOLG1720170120.csv',
    'DOLG1720170123.csv', 'DOLG1720170124.csv', 'DOLG1720170126.csv',
    'DOLG1720170127.csv', 'DOLG1720170130.csv']

FILES_WDO = [
    'WDOG1720170103.csv', 'WDOG1720170104.csv',
    'WDOG1720170105.csv', 'WDOG1720170106.csv', 'WDOG1720170109.csv',
    'WDOG1720170110.csv', 'WDOG1720170111.csv', 'WDOG1720170112.csv',
    'WDOG1720170113.csv', 'WDOG1720170116.csv', 'WDOG1720170117.csv',
    'WDOG1720170118.csv', 'WDOG1720170119.csv', 'WDOG1720170120.csv',
    'WDOG1720170123.csv', 'WDOG1720170124.csv', 'WDOG1720170126.csv',
    'WDOG1720170127.csv', 'WDOG1720170130.csv']

# %% Function for lists


def get_hawkes_events_list(
        pathin, pathout, file_list, tick_value, min_order_size, start_time,
        end_time, file_type='CME', min_dt=MINDTCME, save_files=False,
        symmetries1d=[], symmetries2d=[], plot=False):
    df_list = [initall(
        pathin, pathout, file, tick_value, min_order_size, start_time,
        end_time, file_type, min_dt, save_files) for file in file_list]
    timestamps = [get_event_14_timestamps(df)[0] for df in df_list]
    hawkes_learner = HawkesConditionalLaw(
        claw_method="log", delta_lag=0.1, min_lag=5e-4, max_lag=500,
        quad_method="log", n_quad=10, min_support=1e-4, max_support=1,
        n_threads=-1)
    hawkes_learner.set_model(symmetries1d=symmetries1d,
                             symmetries2d=symmetries2d)
    hawkes_learner.fit(timestamps)
    if plot:
        plot_hawkes_kernel_norms(hawkes_learner, node_names=EV_14_LBLS)
    hbase = hawkes_learner.baseline
    hmean = hawkes_learner.mean_intensity
    hnorms = hawkes_learner.kernels_norms
    return [hbase, hmean, hbase/hmean, hnorms]


# %% Run

hawkes_event_DOLs = get_hawkes_events_list(
    PATHIN, PATHOUT, FILES_DOL, TS1, MOSDOL, START_TIME1, END_TIME1, 'BMF',
    MINDT1, False, [], [], False)

sns.heatmap(hawkes_event_DOLs[3], center=0, cmap='RdBu',
            annot=True, fmt=".2f",
            xticklabels=EV_14_LBLS, yticklabels=EV_14_LBLS)

hawkes_event_WDOs = get_hawkes_events_list(
    PATHIN, PATHOUT, FILES_WDO, TS1, MOSWDO, START_TIME1, END_TIME1, 'BMF',
    MINDT1, False, [], [], False)

sns.heatmap(hawkes_event_WDOs[3], center=0, cmap='RdBu',
            annot=True, fmt=".2f",
            xticklabels=EV_14_LBLS, yticklabels=EV_14_LBLS)

# %% Imbalance prediction

# dfDOL2 = dfDOL.copy()
# dfDOL2['nextdMP'] = dfDOL2['Microprice'].diff().shift(-1).copy()
# dfDOL2['nextEvent'] = dfDOL2['Event_detail'].shift(-1).copy()

# dfpredDOL = pd.pivot_table(dfDOL2, values='nextdMP', index=['bid_1_qty'],
#                            columns='Event_detail', aggfunc=np.mean,
#                            margins=True)

# sns.relplot(x='bid_1_qty', y='nextdMP', col='Imbal_Sign', row='Event_detail',
#             data=dfDOL2)

# %% Statistics of Qty - function


def ob_stats(data_frame, quant_max=0.98):
    data_framec = data_frame.copy()
    qmax = np.max(data_framec[['bid_1_qty', 'ask_1_qty']].quantile(quant_max))
    print(data_framec[['bid_1_qty', 'ask_1_qty', 'Imbalance']].describe())
    data_framec_tr = data_framec[data_framec['trade_qty'] > 0].copy()
    print('Trades - mean')
    print(data_framec_tr[['trade_qty', 'levels_traded']].mean())
    print('Trades - median')
    print(data_framec_tr[['trade_qty', 'levels_traded']].median())
    sns.jointplot('bid_1_qty', 'ask_1_qty', data=data_framec, kind='hex',
                  xlim=(0, qmax), ylim=(0, qmax))
    plt.show()
    sns.distplot(data_framec['Imbalance'])
    plt.show()

# %% Statistics of Qty - examples


ob_stats(dfDOL)
ob_stats(dfWDO)

ob_stats(dfCME1)
ob_stats(dfCME2)

# %% Plot events' frequency - function


def plot_events_perc(data_frame, title='', window=EVENT_WINDOW,
                     perc_format=True, save_fig=False):
    if perc_format:
        const = 100
    else:
        const = 1
    df_dummies = pd.get_dummies(data_frame.set_index('DateTime')['Event_CLM'])
    plot_title = title + ' | Frequency (%) of events - Window of ' +\
        str(window) + ' events'
    plot_file = title + '_Freq_' + str(window)
    (df_dummies.rolling(window).mean()*const)\
        .plot(title=plot_title, figsize=(15, 10))
    plt.legend(loc='center right', bbox_to_anchor=(1.1, 0.5))
    if save_fig:
        plt.savefig(PATHOUT+plot_file+'.png', dpi=200, format='png')

# %% Plot events' frequency - examples


plot_events_perc(dfDOL, title='DOL 2017-01-19', window=10000,
                 save_fig=SAVEBMF)
plot_events_perc(dfWDO, title='WDO 2017-01-19', window=10000,
                 save_fig=SAVEBMF)

plot_events_perc(dfCME1, title='CME 2018-01-05', window=10000,
                 save_fig=SAVECME)
plot_events_perc(dfCME2, title='CME 2018-01-04', window=10000,
                 save_fig=SAVECME)

# %% Plot reversion frequency - function


def plot_reversion_perc(data_frame, title='', window=EVENT_WINDOW,
                        perc_format=True, save_fig=False):
    subdf = data_frame[['DateTime', 'Reversion']].copy()\
        .set_index('DateTime')
    plot_title = title + ' | Reversion % - Window of ' + str(window)\
        + ' events'
    plot_file = title + '_Reversion_' + str(window)
    if perc_format:
        const = 100
    else:
        const = 1
    (subdf.rolling(window).mean()*const).plot(
        title=plot_title, figsize=(15, 10), legend=False)
    if save_fig:
        plt.savefig(PATHOUT+plot_file+'.png', dpi=200, format='png')

# %% Plot reversion frequency - examples


plot_reversion_perc(dfDOL, title='DOL 2017-01-19', window=1000,
                    save_fig=SAVEBMF)
plot_reversion_perc(dfDOL, title='DOL 2017-01-19', window=10000,
                    save_fig=SAVEBMF)
plot_reversion_perc(dfWDO, title='WDO 2017-01-19', window=1000,
                    save_fig=SAVEBMF)
plot_reversion_perc(dfWDO, title='WDO 2017-01-19', window=10000,
                    save_fig=SAVEBMF)

plot_reversion_perc(dfCME1, title='CME 2018-01-05', window=1000,
                    save_fig=SAVECME)
plot_reversion_perc(dfCME1, title='CME 2018-01-05', window=10000,
                    save_fig=SAVECME)
plot_reversion_perc(dfCME2, title='CME 2018-01-04', window=1000,
                    save_fig=SAVECME)
plot_reversion_perc(dfCME2, title='CME 2018-01-04', window=10000,
                    save_fig=SAVECME)

# %% Plot reversion frequency - examples - zoom in event


subdfCME1 = dfCME1[(dfCME1['DateTime'] >=
                    pd.to_datetime('2018-01-05 07:15:00'))
                   & (dfCME1['DateTime'] <=
                      pd.to_datetime('2018-01-05 07:45:00'))].copy()

plot_reversion_perc(subdfCME1, title='CME 2018-01-05 event', window=100,
                    save_fig=True)
plot_reversion_perc(subdfCME1, title='CME 2018-01-05 event', window=1000,
                    save_fig=True)

plot_events_perc(subdfCME1, title='CME 2018-01-05 event', window=100,
                 save_fig=True)
plot_events_perc(subdfCME1, title='CME 2018-01-05 event', window=1000,
                 save_fig=True)


# %% Plot durations - function


def plot_duration(data_frame, title='', window=EVENT_WINDOW, invert=False,
                  save_fig=False):
    subdf = data_frame[['DateTime', 'dt']].copy()\
        .set_index('DateTime')
    roll_series = subdf.rolling(window).mean()
    if invert:
        plot_title = title + ' | -Log(Duration) - Window of ' + str(window)\
            + ' events'
        plot_file = title + '_Log(Duration)_' + str(window)
        (-(roll_series.apply(np.log))).plot(title=plot_title,
                                            figsize=(15, 10), legend=False)
    else:
        plot_title = title + ' | Duration - Window of ' + str(window)\
            + ' events'
        plot_file = title + '_Duration_' + str(window)
        (roll_series).plot(title=plot_title, figsize=(15, 10), legend=False)
    if save_fig:
        plt.savefig(PATHOUT+plot_file+'.png', dpi=200, format='png')

# %% Plot durations - examples


plot_duration(dfDOL, title='DOL 1000', window=1000, save_fig=SAVEBMF)
plot_duration(dfDOL, title='DOL 10000', window=10000, save_fig=SAVEBMF)
plot_duration(dfWDO, title='WDO 1000', window=1000, save_fig=SAVEBMF)
plot_duration(dfWDO, title='WDO 10000', window=10000, save_fig=SAVEBMF)


plot_duration(dfCME1, title='CME 2018-01-05', window=1000,
              save_fig=SAVECME)
plot_duration(dfCME1, title='CME 2018-01-05', window=10000,
              save_fig=SAVECME)
plot_duration(dfCME1, title='CME 2018-01-05', window=1000, invert=True,
              save_fig=SAVECME)

plot_duration(subdfCME1, title='CME 2018-01-05', window=100,
              save_fig=SAVECME)
plot_duration(subdfCME1, title='CME 2018-01-05', window=1000,
              save_fig=SAVECME)

# %% Transition matrix


def transition_events(data_frame, event='Event_CLM', normalize=False):
    # Options for event: 'Event_CLM' (default) and 'Event_detail'
    return pd.crosstab(index=data_frame[event].values,
                       columns=data_frame[event].shift(-1).values,
                       margins=True, normalize=normalize)

# %% Test transition matrix


trans_CLM_count_DOL = transition_events(dfDOL, event='Event_CLM')
trans_CLM_count_WDO = transition_events(dfWDO, event='Event_CLM')
if SAVEBMF:
    trans_CLM_count_DOL.to_csv(PATHOUT+'trans_CLM_count_ev_DOL.csv')
    trans_CLM_count_WDO.to_csv(PATHOUT+'trans_CLM_count_ev_WDO.csv')

trans_detail_count_DOL = transition_events(dfDOL, event='Event_detail')
trans_detail_count_WDO = transition_events(dfWDO, event='Event_detail')
if SAVECME:
    trans_detail_count_DOL.to_csv(PATHOUT+'trans_detail_count_ev_DOL.csv')
    trans_detail_count_WDO.to_csv(PATHOUT+'trans_detail_count_ev_WDO.csv')

# %% Transition matrix - Pivot


def pivot_events(data_frame, event='Event_CLM',
                 piv_values='dt',
                 aggfunc=np.mean):
    # Options for event: 'Event', 'Event_CLM', 'Event_detail'
    # Options for values: 'dt'
    dfc = data_frame.copy()
    dfc['Previous_Event'] = dfc[event].shift(+1).values
    return pd.pivot_table(dfc, values=piv_values, index=['Previous_Event'],
                          columns=event, aggfunc=aggfunc, margins=True)


def pivot_prev_events(data_frame, event='Event_CLM', piv_values='Imbalance',
                      aggfunc=np.mean):
    # Options for event: 'Event', 'Event_CLM', 'Event_detail'
    # Options for values: 'Imbalance', 'trade_qty', 'Spread_Ticks'
    dfc = data_frame.copy()
    dfc['Previous_Event'] = dfc[event].shift(+1).values
    dfc['Previuos_Values'] = dfc[piv_values].shift(+1).values
    return pd.pivot_table(dfc, values='Previuos_Values',
                          index=['Previous_Event'], columns=event,
                          aggfunc=aggfunc, margins=True)

# %% Test Pivot


pivot_dt_detail_DOL = pivot_events(dfDOL, event='Event_detail')
pivot_imb_detail_DOL = pivot_prev_events(dfDOL, event='Event_detail')
pivot_dt_detail_DOL.to_csv(PATHOUT+'pivot_dt_detail_DOL.csv')
pivot_imb_detail_DOL.to_csv(PATHOUT+'pivot_imb_detail_DOL.csv')

pivot_dt_CLM_DOL = pivot_events(dfDOL, event='Event_CLM')
pivot_imb_CLM_DOL = pivot_prev_events(dfDOL, event='Event_CLM')
pivot_dt_CLM_DOL.to_csv(PATHOUT+'pivot_dt_CLM_DOL.csv')
pivot_imb_CLM_DOL.to_csv(PATHOUT+'pivot_imb_CLM_DOL.csv')

pivot_dt_detail_WDO = pivot_events(dfWDO, event='Event_detail')
pivot_imb_detail_WDO = pivot_prev_events(dfWDO, event='Event_detail')
pivot_dt_detail_WDO.to_csv(PATHOUT+'pivot_dt_detail_WDO.csv')
pivot_imb_detail_WDO.to_csv(PATHOUT+'pivot_imb_detail_WDO.csv')

pivot_dt_CLM_WDO = pivot_events(dfWDO, event='Event_CLM')
pivot_imb_CLM_WDO = pivot_prev_events(dfWDO, event='Event_CLM')
pivot_dt_CLM_WDO.to_csv(PATHOUT+'pivot_dt_CLM_WDO.csv')
pivot_imb_CLM_WDO.to_csv(PATHOUT+'pivot_imb_CLM_WDO.csv')

# %% From now on previous code do not uncomment

# DO NOT RUN DO NOT RUN DO NOT RUN DO NOT RUN DO NOT RUN DO NOT RUN DO NOT RUN

# %% run_event_data


# def run_event_data(pathin, pathout, file_name, tick_value, start_time,
#                    end_time, file_type='CME', save_files=False):
#     """Generate intensities.

#     run_event_data(pathin, pathout, file_name, tick_value, start_time,
#                    end_time)
#     returns the intensties (input for Othmane's code)
#     """
#     data = ad(pathin, file_name, file_type)
#     print(data.get_processing_date())
#     tob_obj = atob(data, tick_value)
#     tob_obj.print2file_df_intensity(pathout)

# %% run_intensity_one_days


# def run_intensity_one_days(pathin, pathout, tick_value, file_name=[],
#                            file_type='CME'):
#     """Generate intensities for one day.

#     run_intensity_one_days(pathin, pathout, tick_value, file_name,
#                            start_time, end_time)
#     returns the intensties for one day (input for Othmane's code)
#     """
#     data = ad(pathin, file_name, file_type)
#     print(data.get_processing_date())
#     tob_obj = atob(data, tick_value)
#     output = tob_obj.get_tob_intensity_output()
#     bid_inten = output.get_bid_intensity()
#     ask_inten = output.get_ask_intensity()
#     both_inten = output.get_aggregated_bid_ask()
#     bid_inten.plot_intensities(pathout, 'bid')
#     ask_inten.plot_intensities(pathout, 'ask')
#     both_inten.plot_intensities(pathout, 'bid_plus_ask')

# %% run_intensity_multi_days


# def run_intensity_multi_days(pathin, pathout, tick_value,
#                              file_names=[], file_type='CME'):
#     """Run the intensity function for multiple days.

#     run_intensity_multi_days(pathin, pathout, tick_value,
#                              file_names=[], file_type='CME')
#     returns the intensties (input for Othmane's code) for multiple days
#     """
#     tick_value = TS
#     filepaths = [pathout]
#     # create directories if do not exist
#     for path in filepaths:
#         if not os.path.exists(path):
#             os.makedirs(path)
#     # either explicitly set file_names or get file_names from data path
#     if len(file_names) == 0:
#         for file in os.listdir(pathin):
#             if file.endswith("csv") or file.endswith(".zip"):
#                 file_names.append(file)
#     for f_name in file_names:
#         start = timeit.default_timer()
#         print('--START------')
#         data = ad(pathin, f_name, file_type)
#         tob_obj = atob(data, tick_value)
#         if file_names[0] == f_name:
#             output = tob_obj.get_tob_intensity_output()
#         else:
#             output.append(tob_obj.get_tob_intensity_output())
#         stop = timeit.default_timer()
#         print('Time Spent: ', round(stop - start), ' seconds')
#         print('--END-------')
#     intensity = output.get_aggregated_bid_ask()
#     intensity.plot_intensities(pathout, True)
#     intensity.plot_proba_stat(pathout, True)

# %% runc_multi_days


# def runc_multi_days(pathin, pathout, tick_value, start_time, end_time,
#                     file_names=[], file_type='CME'):
#     """Run uz_stats for multiple days.

#     runc_multi_days(pathin, pathout, tick_value, start_time, end_time,
#                     file_names=[], file_type='CME')
#     returns the UZ stats for multiple days
#     """
#     tick_value = TS
#     filepaths = [pathout]
#     # create directories if do not exist
#     for path in filepaths:
#         if not os.path.exists(path):
#             os.makedirs(path)
#     # either explicitly set file_names or get file_names from data path
#     if len(file_names) == 0:
#         for file in os.listdir(pathin):
#             if file.endswith("csv") or file.endswith(".zip"):
#                 file_names.append(file)
#     for f_name in file_names:
#         start = timeit.default_timer()
#         print('--START------')
#         data = ad(pathin, f_name, file_type)
#         uz_obj = uz(data, tick_value, start_time, end_time)
#         if file_names[0] == f_name:
#             output = uz_obj.get_Armada_UZModel_output()
#         else:
#             output.append(uz_obj.get_Armada_UZModel_output())
#         stop = timeit.default_timer()
#         print('Time Spent: ', round(stop - start), ' seconds')
#         print('--END-------')
#     output.print2file_df_cont_alt_by_ticks(pathout)
#     output.print2file_df_uz_stats(pathout)
#     output.plot_html_uz_stats(pathout)


# %% run_unc_zones_read


# def run_unc_zones_read(pathin, pathout, file_name, tick_value, start_time,
#                        end_time, file_type='CME', save_files=False):
#     """Plot uz data.

#     run_unc_zones(pathin, pathout, file_name, tick_value, end_of_time)
#     returns the uncertainty zones data frame
#     """
#     data = ad(pathin, file_name, file_type)
#     data.plot_html_ohlc(pathout, '5min', pd.to_timedelta('00:00:00'),
#                         pd.to_timedelta('23:59:00'))
#     # data.plot_html_ohlc(pathout,'1min', pd.to_timedelta('07:00:00'),
#     #                     pd.to_timedelta('10:00:00'))
#     # data.plot_html_ohlc(pathout, '1S', pd.to_timedelta('07:27:00'),
#     #                     pd.to_timedelta('07:33:00'))
#     uz_obj = uz(data, tick_value, start_time, end_time)
#     # ohlc = uz_obj.ohlc(pathout)
#     uz_obj.print2file_df_cont_alt_by_ticks(pathout)
#     uz_obj.print2file_df_uz_stats(pathout)
#     # this should be start_time - 10s:
#     data.plot_html_1mintick(pathout, pd.to_timedelta('07:29:50'))


# %% run_tob


# def run_tob(pathin, pathout, file_name, tick_value, start_time,
#             end_time, file_type='CME', save_files=False):
#     """Create the tob object (enhanced Top of Order Book).

#     run_tob(pathin, pathout, file_name, tick_value, start_time,
#             end_time, file_type='CME', save_files=False)
#     returns the enhanced Top of The Order Book
#     """
#     data = ad(pathin, file_name, file_type)
#     print(data.get_processing_date())
#     data.plot_html_ohlc(pathout, '1min', pd.to_timedelta('00:00:00'),
#                         pd.to_timedelta('23:59:00'))
#     tob_obj = atob(data, tick_value)
#     tob_obj.print2file_df_tob(pathout, start_time, end_time)


# %% Run benchmark


# def run_benchmark(pathin, pathout, file_name, tick_value, start_time,\
#                   end_time):
#     import armadauzdf
#     armadauzdf.run_unc_zones(pathin, pathout, file_name, tick_value,
#                              start_time,\
#                   end_time, 9.25, False)


# %% Run tests CME

# run_intensity_multi_days(PATHIN, PATHOUT, TS, [], 'CME')

# run_event_data(PATHIN, PATHOUT, FILE1, TS, START_TIME, END_TIME)

# %% Run tests BMF - DOL

# run_event_data(PATHIN, PATHOUT, FILE_BMF1, TS, START_TIME1, END_TIME1, 'BMF')

# %% Intensity columns

# INT_COLUMNS = ['order_type', 'size_before', 'var_DateTime', 'Number',
#                'Intensity']
# PIVOT_COLUMNS = ['var_DateTime', 'Number', 'Intensity']

# %% Use files and plot - DOL

# DF_INT_BID_DOL = pd.read_csv(PATHOUT+'df_intensity_bid.csv')
# DF_INT_ASK_DOL = pd.read_csv(PATHOUT+'df_intensity_ask.csv')
# DF_INT_BID_DOL = DF_INT_BID_DOL[INT_COLUMNS].copy()
# DF_INT_ASK_DOL = DF_INT_ASK_DOL[INT_COLUMNS].copy()
# DF_INT_BID_DOL['size_before'] = DF_INT_BID_DOL['size_before']/5
# DF_INT_ASK_DOL['size_before'] = DF_INT_ASK_DOL['size_before']/5
# DF_INT_DOL = DF_INT_BID_DOL.pivot(index='size_before',
#                                   columns='order_type',
#                                   values='Intensity')
# DF_INT_DOL.loc[:40].plot()

# %% Run tests BMF - WDO

# run_event_data(PATHIN, PATHOUT, FILE_BMF2, TS, START_TIME1, END_TIME1, 'BMF')

# %% Use files and plot - WDO

# DF_INT_BID_WDO = pd.read_csv(PATHOUT+'df_intensity_bid.csv')
# DF_INT_ASK_WDO = pd.read_csv(PATHOUT+'df_intensity_ask.csv')
# DF_INT_BID_WDO = DF_INT_BID_WDO[INT_COLUMNS].copy()
# DF_INT_ASK_WDO = DF_INT_ASK_WDO[INT_COLUMNS].copy()
# DF_INT_BID_WDO['size_before'] = DF_INT_BID_WDO['size_before']
# DF_INT_ASK_WDO['size_before'] = DF_INT_ASK_WDO['size_before']
# DF_INT_WDO = DF_INT_BID_WDO.pivot(index='size_before',
#                                   columns='order_type',
#                                   values='Intensity')
# DF_INT_WDO.loc[:80].plot()

# %% Debug Class UZ

# DOL = ad(PATHIN, FILE_BMF1, 'BMF')
# DOLdf = DOL.df
# UZDOL = uz(DOL, TS1, START_TIME1, END_TIME1)
# UZDOLdf = UZDOL.df
# UZDOLdft0 = UZDOL.df[~UZDOL.df['OT']].copy()
# UZDOLdft = UZDOL.df_trades
# UZDOLdftt = UZDOL.df_trades_by_time
# UZDOLdfttp = UZDOL.df_trades_by_price
# UZDOLdfuz = UZDOL.df_trades_adduz

# %% Debug script

# DOL2 = pd.read_csv(PATHIN+FILE_BMF1)
# DOL2 = mcsc.column_datetime(mcsc.column_ot(DOL2))
# DOL2 = mcsc.select_times(DOL2, START_TIME1, END_TIME1)
# DOL2T = DOL2[~DOL2['OT']].copy()
# DOL2Tt = mcsc.collapse_time(DOL2T)
# DOL2Ttp = mcsc.collapse_price(DOL2Tt)


# %% Test UZ BMF

# run_unc_zones_read(PATHIN, PATHOUT, FILE_BMF1, TS1, START_TIME1, END_TIME1,
#                     'BMF')

# %% Debug Class TOB


# DOL = ad(PATHIN, FILE_BMF1, 'BMF')
# DOLdf = DOL.select_times(pd.to_timedelta('09:00:00'),
#                          pd.to_timedelta('18:15:00')).df

# %% Copy df to play


# df2 = DOLdf.copy()

# %% Add Order flags and count


# df2['OrderQ'] = df2['trade_price'].isnull().copy()
# df2['OrderN'] = df2['OrderQ'].copy().cumsum()
# df2['last_trade'] = df2['trade_price'].copy().fillna(method='ffill')

# %% Group trades


# df2g = df2.groupby(['OrderN', 'DateTime', 'OrderQ', 'last_trade'],
#                    sort=False).sum(min_count=1)

# %% Reset columns from groupby


# df2g = df2g.reset_index()
# df2g['trade_price'] = np.where(df2g['OrderQ'], np.nan,
#                                df2g['last_trade'].copy())
# df2g = df2g.drop(columns=['OrderN', 'last_trade'])

# %% Print counts


# print(['Orders: ', df2g['OrderQ'].sum(), 'Trades: ',
#        len(df2g)-df2g['OrderQ'].sum()])

# %% Levels 1 and 2 diff


# def lvldiff(df):
#     dfc = df.copy()
#     dfdiff1 = dfc[['bid_1_qty', 'bid_1_price', 'ask_1_price',
#                    'ask_1_qty']].copy().diff().abs()
#     dfc['lvl1'] = dfdiff1.sum(axis=1) != 0
#     dfdiff2 = dfc[['bid_2_qty', 'bid_2_price', 'ask_2_price',
#                    'ask_2_qty']].copy().diff().abs()
#     dfc['lvl2'] = dfdiff2.sum(axis=1) != 0
#     return dfc

# %% Exclude Level 2 events


# df2gdiff = lvldiff(df2g)
# df2glvl1 = df2gdiff[~((~df2gdiff['lvl1']) & (df2gdiff['lvl2']))]
# df2glvl1 = df2glvl1.drop(['bid_2_qty', 'bid_2_ord', 'bid_2_price',
#                           'bid_1_ord', 'ask_1_ord',
#                           'ask_2_price', 'ask_2_ord', 'ask_2_qty',
#                           'lvl2'], axis=1)

# %% Check trades without book update or sweep not instantaneous


# def find_invalid_trades(df, dt=0.001):
#     dfc = df.copy()
#     dfc['Prev_Trade'] = (~dfc['OrderQ']).shift()
#     dfc['Signif_dt'] = dfc['DateTime'].diff().dt.total_seconds() > dt
#     dfc['Check'] = (dfc['Prev_Trade'] & (dfc['Signif_dt'])).shift(
#         periods=-1, fill_value=False)
#     return dfc[dfc['Check']].copy()\
#         .drop(['Prev_Trade', 'Signif_dt', 'Check'], axis=1)


# %% Clear trades without book update or sweep not instantaneous - function


# def clear_invalid_trades(df, dt=0.001):
#     dfc = df.copy()
#     dfc['Prev_Trade'] = (~dfc['OrderQ']).shift()
#     dfc['Signif_dt'] = dfc['DateTime'].diff().dt.total_seconds() > dt
#     dfc['Check'] = (dfc['Prev_Trade'] & (dfc['Signif_dt'])).shift(
#         periods=-1, fill_value=False)
#     return dfc[~dfc['Check']].copy()\
#         .drop(['Prev_Trade', 'Signif_dt', 'Check'], axis=1)

# %% Clear trades without book update or sweep not instantaneous


# df2glvl1clean = clear_invalid_trades(df2glvl1)
# df2glvl1clean = clear_invalid_trades(df2glvl1clean)


# %% subdf - recheck trades without book update or sweep not instantaneous

# dfsub = find_invalid_trades(df2glvl1clean)

# subdftrd = df2glvl1.loc[284416:284456].copy()

# %% Define side of trade - calculate columns

# df2glvl1clean['bid_traded'] = df2glvl1clean['bid_1_price'].copy().fillna(
#     method='ffill') >= df2glvl1clean['trade_price']
# df2glvl1clean['ask_traded'] = df2glvl1clean['ask_1_price'].copy().fillna(
#     method='ffill') <= df2glvl1clean['trade_price']
# df2glvl1clean['dt'] = df2glvl1clean['DateTime'].diff().dt.total_seconds()

# %% Define key for groupby


# df2glvl1clean['OrderN'] = df2glvl1clean['OrderQ'].copy().cumsum()
# df2glvl1clean = df2glvl1clean[df2glvl1clean['OrderN'] >0].copy()
# df2glvl1clean['OrderN'] = df2glvl1clean['OrderN']*\
#     (2*df2glvl1clean['OrderQ']-1)

# %% Group trades again


# df2glvl1cleang = df2glvl1clean.groupby(['DateTime', 'OrderN'], sort=False)
# df3 = df2glvl1cleang.agg({'OrderQ': all,
#                           'bid_1_qty': sum,
#                           'bid_1_price': sum,
#                           'trade_price': 'count',
#                           'trade_qty': sum,
#                           'ask_1_price': sum,
#                           'ask_1_qty': sum,
#                           'lvl1': any,
#                           'bid_traded': any,
#                           'ask_traded': any,
#                           'dt': sum})
# df3 = df3.reset_index()
# df3 = df3.rename(columns={'trade_price': 'levels_traded',
#     'lvl1': 'NoTradeQ'})

# %% Recheck trades

# def find_invalid_trades_again(df, dt=0.001):
#     dfc = df.copy()
#     dfc['Prev_Trade'] = (dfc['OrderN'].shift()) < 0
#     dfc['Signif_dt'] = dfc['dt'] > dt
#     dfc['Check'] = (dfc['Prev_Trade'] & (dfc['Signif_dt']))
#     return dfc[dfc['Check']].copy()\
#         .drop(['Prev_Trade', 'Signif_dt', 'Check'], axis=1)

# %% Show problems

# dfsub2 = find_invalid_trades_again(df3)

# %% Push trades on next order book state

# df3['OrderId'] = np.abs(df3['OrderN']) + (1-np.sign(df3['OrderN']))/2
# df4 = df3.groupby(['DateTime', 'OrderId']).sum()
# df4 = df4.reset_index()
# df4 = df4.drop(['OrderN', 'OrderQ'], axis=1)

# %% Short excerpt

# df4sub = df4.head(50)

# %% Initializa df
# df = DOLdf.copy()

# %% TradeQ column
# df['TradeQ'] = ~df.trade_price.isnull().copy()

# %% Initialize new df
# dfc = pd.DataFrame(columns=df.columns)

# %% Loop through df and append either row or qty (slow!!!)
# for index, row in df.iterrows():
#     if len(dfc) > 0:
#         timeQ = dfc.iloc[-1]['DateTime'] == row['DateTime']
#         priceQ = dfc.iloc[-1]['trade_price'] == row['trade_price']
#         if timeQ and row['TradeQ'] and priceQ:
#             dfc_tail = list(dfc.tail(1).iterrows())[0]
#             dfc.at[dfc_tail[0], 'trade_qty'] += row['trade_qty']
#         else:
#             dfc = dfc.append(row)
#     else:
#         dfc = dfc.append(row)
#     if len(dfc) % 1000 == 0:
#         print(len(dfc))

# %% Export collapsed df
# dfc.to_csv(PATHOUT+'dfc.csv')

# %% Read collapsed df
# dfc = pd.read_csv(PATHOUT+'dfc.csv')
