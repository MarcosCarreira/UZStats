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
START_TIME = pd.to_timedelta('07:30:00')
END_TIME = pd.to_timedelta('12:45:00')

# %% BMF Constants
TS1 = 0.5
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

# %% Debug Class TOB


DOL = ad(PATHIN, FILE_BMF1, 'BMF')
DOLdf = DOL.select_times(pd.to_timedelta('09:00:00'),
                         pd.to_timedelta('18:15:00')).df

# %% [markdown]
# Start function here

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

# %% Copy df to play


df2 = DOLdf.copy()

# %% Add Order flags and count


df2['OrderQ'] = df2['trade_price'].isnull().copy()
df2['OrderN'] = df2['OrderQ'].copy().cumsum()
df2['last_trade'] = df2['trade_price'].copy().fillna(method='ffill')

# %% Group trades


df2g = df2.groupby(['OrderN', 'DateTime', 'OrderQ', 'last_trade'],
                   sort=False).sum(min_count=1)

# %% Reset columns from groupby


df2g = df2g.reset_index()
df2g['trade_price'] = np.where(df2g['OrderQ'], np.nan,
                               df2g['last_trade'].copy())
df2g = df2g.drop(columns=['OrderN', 'last_trade'])

# %% Print counts


print(['Orders: ', df2g['OrderQ'].sum(), 'Trades: ',
       len(df2g)-df2g['OrderQ'].sum()])

# %% Levels 1 and 2 diff


def lvldiff(df):
    dfc = df.copy()
    dfdiff1 = dfc[['bid_1_qty', 'bid_1_price', 'ask_1_price',
                   'ask_1_qty']].copy().diff().abs()
    dfc['lvl1'] = dfdiff1.sum(axis=1) != 0
    dfdiff2 = dfc[['bid_2_qty', 'bid_2_price', 'ask_2_price',
                   'ask_2_qty']].copy().diff().abs()
    dfc['lvl2'] = dfdiff2.sum(axis=1) != 0
    return dfc

# %% Exclude Level 2 events


df2gdiff = lvldiff(df2g)
df2glvl1 = df2gdiff[~((~df2gdiff['lvl1']) & (df2gdiff['lvl2']))]
df2glvl1 = df2glvl1.drop(['bid_2_qty', 'bid_2_ord', 'bid_2_price',
                          'bid_1_ord', 'ask_1_ord',
                          'ask_2_price', 'ask_2_ord', 'ask_2_qty',
                          'lvl2'], axis=1)

# %% Check trades without book update or sweep not instantaneous


def find_invalid_trades(df, dt=0.001):
    dfc = df.copy()
    dfc['Prev_Trade'] = (~dfc['OrderQ']).shift()
    dfc['Signif_dt'] = dfc['DateTime'].diff().dt.total_seconds() > dt
    dfc['Check'] = (dfc['Prev_Trade'] & (dfc['Signif_dt'])).shift(
        periods=-1, fill_value=False)
    return dfc[dfc['Check']].copy()\
        .drop(['Prev_Trade', 'Signif_dt', 'Check'], axis=1)


# %% Clear trades without book update or sweep not instantaneous - function


def clear_invalid_trades(df, dt=0.001):
    dfc = df.copy()
    dfc['Prev_Trade'] = (~dfc['OrderQ']).shift()
    dfc['Signif_dt'] = dfc['DateTime'].diff().dt.total_seconds() > dt
    dfc['Check'] = (dfc['Prev_Trade'] & (dfc['Signif_dt'])).shift(
        periods=-1, fill_value=False)
    return dfc[~dfc['Check']].copy()\
        .drop(['Prev_Trade', 'Signif_dt', 'Check'], axis=1)

# %% Clear trades without book update or sweep not instantaneous


df2glvl1clean = clear_invalid_trades(df2glvl1)
df2glvl1clean = clear_invalid_trades(df2glvl1clean)


# %% subdf - recheck trades without book update or sweep not instantaneous

dfsub = find_invalid_trades(df2glvl1clean)

# subdftrd = df2glvl1.loc[284416:284456].copy()

# %% Define side of trade - calculate columns

df2glvl1clean['bid_traded'] = df2glvl1clean['bid_1_price'].copy().fillna(
    method='ffill') >= df2glvl1clean['trade_price']
df2glvl1clean['ask_traded'] = df2glvl1clean['ask_1_price'].copy().fillna(
    method='ffill') <= df2glvl1clean['trade_price']
df2glvl1clean['dt'] = df2glvl1clean['DateTime'].diff().dt.total_seconds()

# %% Define key for groupby


df2glvl1clean['OrderN'] = df2glvl1clean['OrderQ'].copy().cumsum()
df2glvl1clean = df2glvl1clean[df2glvl1clean['OrderN'] >0].copy()
df2glvl1clean['OrderN'] = df2glvl1clean['OrderN']*\
    (2*df2glvl1clean['OrderQ']-1)

# %% Group trades again


df2glvl1cleang = df2glvl1clean.groupby(['DateTime', 'OrderN'], sort=False)
df3 = df2glvl1cleang.agg({'OrderQ': all,
                          'bid_1_qty': sum,
                          'bid_1_price': sum,
                          'trade_price': 'count',
                          'trade_qty': sum,
                          'ask_1_price': sum,
                          'ask_1_qty': sum,
                          'lvl1': any,
                          'bid_traded': any,
                          'ask_traded': any,
                          'dt': sum})
df3 = df3.reset_index()
df3 = df3.rename(columns={'trade_price': 'levels_traded', 'lvl1': 'NoTradeQ'})

# %% Recheck trades

def find_invalid_trades_again(df, dt=0.001):
    dfc = df.copy()
    dfc['Prev_Trade'] = (dfc['OrderN'].shift()) < 0
    dfc['Signif_dt'] = dfc['dt'] > dt
    dfc['Check'] = (dfc['Prev_Trade'] & (dfc['Signif_dt']))
    return dfc[dfc['Check']].copy()\
        .drop(['Prev_Trade', 'Signif_dt', 'Check'], axis=1)

# %% Show problems

dfsub2 = find_invalid_trades_again(df3)

# %% Push trades on next order book state

df3['OrderId'] = np.abs(df3['OrderN']) + (1-np.sign(df3['OrderN']))//2
df4 = df3.groupby(['DateTime', 'OrderId']).sum()
df4 = df4.reset_index()
df4 = df4.drop(['OrderN', 'OrderQ'], axis=1)

# %% Short excerpt

df4sub = df4.head(50)

# %% Define side of event - function

# df2glvl1['bid_1_qty_diff'] = df2glvl1['bid_1_qty'].copy().diff()
# df2glvl1['bid_1_price_diff'] = df2glvl1['bid_1_price'].copy().diff()
# df2glvl1['ask_1_price_diff'] = df2glvl1['ask_1_price'].copy().diff()
# df2glvl1['ask_1_qty_diff'] = df2glvl1['ask_1_qty'].copy().diff()



# %% Print counts comparison
# print(['Orders df2g: ', df2g['OrderQ'].sum(),
#        'Orders dfc: ',len(dfc)-dfc['TradeQ'].sum()])
# print(['Trades df2g: ', len(df2g)-df2g['OrderQ'].sum(),
#        'Trades dfc: ',dfc['TradeQ'].sum()])
# print(['bid_1_qty df2g: ', df2g['bid_1_qty'].sum(),
#        'bid_1_qty dfc: ',dfc['bid_1_qty'].sum()])
# print(['trade_qty df2g: ', df2g['trade_qty'].sum(),
#        'trade_qty dfc: ',dfc['trade_qty'].sum()])

# %% Filter trades

# dfct = dfc[dfc['TradeQ']]
# df2gt = df2g[~df2g['OrderQ']]
# dfsub = df2[(df2['DateTime'] > pd.to_datetime('2017-01-19 09:00:45'))].copy()
# dfsub = dfsub[(dfsub['DateTime'] < pd.to_datetime('2017-01-19 09:00:46'))]\
#                .copy()

# %% [markdown]
# End function here

# %% Debug Class TOB
# DOLTOB = atob(DOL.select_times(pd.to_timedelta('09:00:00'),
#                                pd.to_timedelta('18:15:00')), TS1)
#print(DOLTOB.tob.tail(10))

# %%
#DOLTOB.tob.tail(10)

# %%
