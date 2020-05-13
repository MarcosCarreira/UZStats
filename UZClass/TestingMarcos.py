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
import timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# print(os.getcwd())

# %% Armada Class imports
from armadaClassMarcos import ArmadaData_UZModel as uz
from armadaClassMarcos import Armada_Data as ad
from armadaClassMarcos import Armada_TOB as atob

# %% Othmane imports

import Plotting as pltg

# %% Marcos imports

import armadauzdf as mcsc

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
MINDTCME = 0.001
START_TIME = pd.to_timedelta('07:30:00')
END_TIME = pd.to_timedelta('12:45:00')

# %% BMF Constants
TS1 = 0.5
MOSDOL = 5
MOSWDO = 1
MINDT1 = 0.001
START_TIME1 = pd.to_timedelta('09:00:00')
END_TIME1 = pd.to_timedelta('18:15:00')

# %% BMF file names
FILE_BMF1 = 'DOLG1720170119.csv'
FILE_BMF2 = 'WDOG1720170119.csv'

# %% CME file names
FILE1 = '20180105_6EH8.zip'

# %% run_event_data


def run_event_data(pathin, pathout, file_name, tick_value, start_time,
                   end_time, file_type='CME', save_files=False):
    """Generate intensities.

    run_event_data(pathin, pathout, file_name, tick_value, start_time,
                   end_time)
    returns the intensties (input for Othmane's code)
    """
    data = ad(pathin, file_name, file_type)
    print(data.get_processing_date())
    tob_obj = atob(data, tick_value)
    tob_obj.print2file_df_intensity(pathout)

# %% run_intensity_one_days


def run_intensity_one_days(pathin, pathout, tick_value, file_name=[],
                           file_type='CME'):
    """Generate intensities for one day.

    run_intensity_one_days(pathin, pathout, tick_value, file_name,
                           start_time, end_time)
    returns the intensties for one day (input for Othmane's code)
    """
    data = ad(pathin, file_name, file_type)
    print(data.get_processing_date())
    tob_obj = atob(data, tick_value)
    output = tob_obj.get_tob_intensity_output()
    bid_inten = output.get_bid_intensity()
    ask_inten = output.get_ask_intensity()
    both_inten = output.get_aggregated_bid_ask()
    bid_inten.plot_intensities(pathout, 'bid')
    ask_inten.plot_intensities(pathout, 'ask')
    both_inten.plot_intensities(pathout, 'bid_plus_ask')

# %% run_intensity_multi_days


def run_intensity_multi_days(pathin, pathout, tick_value,
                             file_names=[], file_type='CME'):
    """Run the intensity function for multiple days.

    run_intensity_multi_days(pathin, pathout, tick_value,
                             file_names=[], file_type='CME')
    returns the intensties (input for Othmane's code) for multiple days
    """
    tick_value = TS
    filepaths = [pathout]
    # create directories if do not exist
    for path in filepaths:
        if not os.path.exists(path):
            os.makedirs(path)
    # either explicitly set file_names or get file_names from data path
    if len(file_names) == 0:
        for file in os.listdir(pathin):
            if file.endswith("csv") or file.endswith(".zip"):
                file_names.append(file)
    for f_name in file_names:
        start = timeit.default_timer()
        print('--START------')
        data = ad(pathin, f_name, file_type)
        tob_obj = atob(data, tick_value)
        if file_names[0] == f_name:
            output = tob_obj.get_tob_intensity_output()
        else:
            output.append(tob_obj.get_tob_intensity_output())
        stop = timeit.default_timer()
        print('Time Spent: ', round(stop - start), ' seconds')
        print('--END-------')
    intensity = output.get_aggregated_bid_ask()
    intensity.plot_intensities(pathout, True)
    intensity.plot_proba_stat(pathout, True)


# %% runc_multi_days


def runc_multi_days(pathin, pathout, tick_value, start_time, end_time,
                    file_names=[], file_type='CME'):
    """Run uz_stats for multiple days.

    runc_multi_days(pathin, pathout, tick_value, start_time, end_time,
                    file_names=[], file_type='CME')
    returns the UZ stats for multiple days
    """
    tick_value = TS
    filepaths = [pathout]
    # create directories if do not exist
    for path in filepaths:
        if not os.path.exists(path):
            os.makedirs(path)
    # either explicitly set file_names or get file_names from data path
    if len(file_names) == 0:
        for file in os.listdir(pathin):
            if file.endswith("csv") or file.endswith(".zip"):
                file_names.append(file)
    for f_name in file_names:
        start = timeit.default_timer()
        print('--START------')
        data = ad(pathin, f_name, file_type)
        uz_obj = uz(data, tick_value, start_time, end_time)
        if file_names[0] == f_name:
            output = uz_obj.get_Armada_UZModel_output()
        else:
            output.append(uz_obj.get_Armada_UZModel_output())
        stop = timeit.default_timer()
        print('Time Spent: ', round(stop - start), ' seconds')
        print('--END-------')
    output.print2file_df_cont_alt_by_ticks(pathout)
    output.print2file_df_uz_stats(pathout)
    output.plot_html_uz_stats(pathout)


# %% run_unc_zones_read


def run_unc_zones_read(pathin, pathout, file_name, tick_value, start_time,
                       end_time, file_type='CME', save_files=False):
    """Plot uz data.

    run_unc_zones(pathin, pathout, file_name, tick_value, end_of_time)
    returns the uncertainty zones data frame
    """
    data = ad(pathin, file_name, file_type)
    data.plot_html_ohlc(pathout, '5min', pd.to_timedelta('00:00:00'),
                        pd.to_timedelta('23:59:00'))
    # data.plot_html_ohlc(pathout,'1min', pd.to_timedelta('07:00:00'),
    #                     pd.to_timedelta('10:00:00'))
    # data.plot_html_ohlc(pathout, '1S', pd.to_timedelta('07:27:00'),
    #                     pd.to_timedelta('07:33:00'))
    uz_obj = uz(data, tick_value, start_time, end_time)
    # ohlc = uz_obj.ohlc(pathout)
    uz_obj.print2file_df_cont_alt_by_ticks(pathout)
    uz_obj.print2file_df_uz_stats(pathout)
    # this should be start_time - 10s:
    data.plot_html_1mintick(pathout, pd.to_timedelta('07:29:50'))


# %% run_tob


def run_tob(pathin, pathout, file_name, tick_value, start_time,
            end_time, file_type='CME', save_files=False):
    """Create the tob object (enhanced Top of Order Book).

    run_tob(pathin, pathout, file_name, tick_value, start_time,
            end_time, file_type='CME', save_files=False)
    returns the enhanced Top of The Order Book
    """
    data = ad(pathin, file_name, file_type)
    print(data.get_processing_date())
    data.plot_html_ohlc(pathout, '1min', pd.to_timedelta('00:00:00'),
                        pd.to_timedelta('23:59:00'))
    tob_obj = atob(data, tick_value)
    tob_obj.print2file_df_tob(pathout, start_time, end_time)


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

# %% [markdown]
# Start function here

# %% New init for Armada TOB - Part 1


# Outputs a clean df with trades collapsed by price and Level1 changes only

def init1(pathin, pathout, file_name, tick_value, min_order_size, start_time,
          end_time, file_type='CME', min_dt=MINDTCME, save_files=False):
    data = ad(pathin, file_name, file_type) # ad=ArmadaData
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
    # Levels 1 and 2 diff (flag changes in each of the first two levels)
    def lvldiff(df):
        dfc = df.copy()
        dfdiff1 = dfc[['bid_1_qty', 'bid_1_price', 'ask_1_price',
                       'ask_1_qty']].copy().diff().abs()
        dfc['lvl1'] = dfdiff1.sum(axis=1) != 0
        dfdiff2 = dfc[['bid_2_qty', 'bid_2_price', 'ask_2_price',
                       'ask_2_qty']].copy().diff().abs()
        dfc['lvl2'] = dfdiff2.sum(axis=1) != 0
        return dfc
    # Excluding Level 2 events
    datadfg = lvldiff(datadfg)
    datadfg = datadfg[~((~datadfg['lvl1']) & (datadfg['lvl2']))]
    datadfg = datadfg.drop(['bid_2_qty', 'bid_2_ord', 'bid_2_price',
                            'bid_1_ord', 'ask_1_ord', 'ask_2_price',
                            'ask_2_ord', 'ask_2_qty', 'lvl2'], axis=1)
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
    datadf['OrderN'] = datadf['OrderN']*(2*datadf['OrderQ']-1)
    # Group trades (sum qty, count of price levels traded)
    datadfg = datadf.groupby(['DateTime', 'OrderN'], sort=False)
    dfagg = datadfg.agg({'OrderQ': all, 'bid_1_qty': sum, 'bid_1_price': sum,
                         'trade_price': 'count', 'trade_qty': sum,
                         'ask_1_price': sum, 'ask_1_qty': sum, 'lvl1': any,
                         'bid_traded': any, 'ask_traded': any, 'dt': sum})
    dfagg = dfagg.reset_index()
    dfagg = dfagg.rename(columns={'trade_price': 'levels_traded',
                                  'lvl1': 'NoTradeQ'})
    # Some recheck for invalid trades might be needed here
    def find_invalid_trades_again(df, dt):
        dfc = df.copy()
        dfc['Prev_Trade'] = (dfc['OrderN'].shift()) < 0
        dfc['Signif_dt'] = dfc['dt'] > dt
        dfc['Check'] = (dfc['Prev_Trade'] & (dfc['Signif_dt']))
        return dfc[dfc['Check']].copy()\
            .drop(['Prev_Trade', 'Signif_dt', 'Check'], axis=1)
    # Push trades on next order book state
    dfagg['OrderId'] = np.abs(dfagg['OrderN'])+(1-np.sign(dfagg['OrderN']))/2
    dfstates = dfagg.groupby(['DateTime', 'OrderId']).sum()
    dfstates = dfstates.reset_index()
    dfstates = dfstates.drop(['OrderN', 'OrderQ'], axis=1)
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
    dfstates['ConsQ'] = np.where(dfstates['PriceQ'], ~(
        (dfstates['bid_1_price_diff'] > 0) |
        (dfstates['ask_1_price_diff'] < 0)),
        (dfstates['bid_1_qty_diff'] < 0) |
        (dfstates['ask_1_qty_diff'] < 0) |
        (~dfstates['NoTradeQ'])
        )
    # AskQ column (was the event on the Ask side?)
    # Trades that take out levels but leave an unfilled balance: Cons sign
    dfstates['AskQ'] = np.where(dfstates['NoTradeQ'], (
        (dfstates['ask_1_price_diff'] != 0) |
        (dfstates['ask_1_qty_diff'] != 0)),
        dfstates['ask_traded']
        )
    dfstates.at[0, 'PriceQ'] = False
    dfstates.at[0, 'ConsQ'] = False
    dfstates.at[0, 'AskQ'] = False
    # Classify state
    dfstates['event_code'] =\
        dfstates['AskQ'] * 8 + dfstates['ConsQ'] * 4 +\
        dfstates['NoTradeQ'] * 2 + dfstates['PriceQ'] * 1
    event_dict = {
        0: 'Start', 1: 'MLb', 2: 'Lb', 3: 'Pb+', 4: 'Mb', 5: 'PbM-', 6: 'Cb',
        7: 'PbC-', 8: 'Start', 9: 'MLa', 10: 'La', 11: 'Pa-', 12: 'Ma',
        13: 'PaM+', 14: 'Ca', 15: 'PaC+'}
    dfstates['Event'] = dfstates['event_code'].map(event_dict)
    if save_files:
        dfstates.to_csv(pathout+file_name[:-4]+'_df_states.csv')
    cols_output1 =\
        ['DateTime', 'OrderId', 'bid_1_qty', 'bid_1_price', 'ask_1_price',
         'ask_1_qty', 'trade_qty', 'levels_traded', 'AskQ', 'ConsQ',
         'NoTradeQ', 'PriceQ', 'Event', 'dt', 'Spread_Ticks', 'Midprice',
         'Microprice', 'Imbalance', 'Imbal_Sign']
    cols_output2 =\
        ['bid_traded', 'ask_traded', 'bid_1_qty_diff', 'bid_1_price_diff',
         'ask_1_price_diff', 'ask_1_qty_diff']
    dfstates = dfstates[cols_output1 + cols_output2]
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


# dforig = init1(PATHIN, PATHOUT, FILE_BMF1, TS1, MOSDOL, START_TIME1,
#                END_TIME1, 'BMF', 0.001, False)

dfDOL = initall(PATHIN, PATHOUT, FILE_BMF1, TS1, MOSDOL, START_TIME1,
                 END_TIME1, 'BMF', 0.001, False)
dfWDO = initall(PATHIN, PATHOUT, FILE_BMF2, TS1, MOSDOL, START_TIME1,
                 END_TIME1, 'BMF', 0.001, False)

# dftest30 = dftest.head(30)

# %% Find Starts

# dfDOL[dfDOL['Event'] == 'Start']

# %% Transition matrix


def transition_events(data_frame, values=None, aggfunc=None, normalize=False):
    return pd.crosstab(index=data_frame['Event'].values,
                       columns=data_frame['Event'].shift(-1).values,
                       values=values, aggfunc=aggfunc,
                       margins=True, normalize=normalize)

# %% Test transition matrix


trans_count_DOL = transition_events(dfDOL)
trans_freq_DOL = transition_events(dfDOL, normalize=True)
trans_freq_WDO = transition_events(dfWDO, normalize=True)
trans_freq_DOL.to_csv(PATHOUT+'trans_freq_DOL.csv')
trans_freq_WDO.to_csv(PATHOUT+'trans_freq_WDO.csv')



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
# df3 = df3.rename(columns={'trade_price': 'levels_traded', 'lvl1': 'NoTradeQ'})

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

