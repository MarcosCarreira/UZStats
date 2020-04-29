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
# import numpy as np
# import matplotlib.pyplot as plt

# %% Armada Class imports
from armadaClassMarcos import ArmadaData_UZModel as uz
from armadaClassMarcos import Armada_Data as ad
from armadaClassMarcos import Armada_TOB as atob

# %% Othmane imports

# import Plotting as pltg

# %% Pandas Options
# pd.set_option('mode.chained_assignment', None)
pd.options.display.max_columns = 20
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


# %%
# def run_benchmark(pathin, pathout, file_name, tick_value, start_time,\
#                   end_time):
#     import armadauzdf
#     armadauzdf.run_unc_zones(pathin, pathout, file_name, tick_value,
#                              start_time,\
#                   end_time, 9.25, False)

# %%
# def run_BFM_tob(pathin, pathout, file_name, tick_value, start_time,\
#                   end_time, save_files=False):
#     data = ad(pathin,file_name, 'BMF')
#     print(data.file_name)
#     print(data.processing_date) # test if bug if fixed
#     data.plot_html_ohlc(pathout,'1min', pd.to_timedelta('00:00:00'),
#                         pd.to_timedelta('23:59:00'))
#     tob_obj = atob(data, tick_value)
#     tob_obj.print2file_df_tob(pathout, start_time, end_time)
#     data.plot_html_1mintick(pathout,pd.to_timedelta('08:59:55'))
#     uz_obj = uz(data,tick_value,start_time,end_time)
#     uz_obj.print2file_df_cont_alt_by_ticks(pathout)
#     uz_obj.print2file_df_uz_stats(pathout)
#     print('done')

# %%
# def run_compare_tob(pathin, pathout, file_names = []):
#     file_before = 'tob_before.zip'
#     file_after = 'tob.zip'
#     tob_before = pd.read_csv(pathout+file_before)
#     tob_after = pd.read_csv(pathout+file_after)
#     tob_after2 = tob_after[3:699774]
#     tob_before.rename(columns = {'Bid 1 Qty':'bid_1_qty',\
#                                        'Bid 2 Qty':'bid_2_qty',\
#                                        'Bid 1 Price':'bid_1_price',\
#                                        'Bid 2 Price':'bid_2_price',\
#                                        'Bid 1 Ord':'bid_1_ord',\
#                                        'Ask 1 Ord':'ask_1_ord',\
#                                        'Bid 2 Ord':'bid_2_ord',\
#                                        'Ask 2 Ord':'ask_2_ord',\
#                                        'Ask 1 Qty':'ask_1_qty',\
#                                        'Ask 2 Qty':'ask_2_qty',\
#                                        'Ask 1 Price':'ask_1_price',\
#                                        'Ask 2 Price':'ask_2_price',\
#                                        'Trade Price':'trade_price',\
#                                        'Trade Qty':'trade_qty',\
#                                     'Aggression':'aggression',\
#                                            }, inplace = True)
#     #tob_before.drop(['bid_2_qty', 'bid_2_price','bid_2_ord','ask_2_ord',\
#     #            'ask_2_qty', 'ask_2_price', 'OT'],axis=1, inplace=True)
#     #tob_after2.drop(['bid_1_price_last', 'ask_1_price_last',
#                       'bid_price_traded'\
#     #                 ,'ask_price_traded'],axis=1, inplace=True)
#     tob_before = \
#             tob_before.set_index(tob_before['DateTime'])
#     tob_after2 = \
#             tob_after2.set_index(tob_after2['DateTime'])
#     bid_1_pr_match = pd.DataFrame( np.where(tob_after2['bid_1_price'] == \
#           tob_before['Bid_Price'], True, False), index= tob_before.index)
#     #df_bid_1_price['before']=tob_before['Bid_Price'].copy()
#     #df_bid_1_price['after']=tob_after2['bid_1_price'].copy()
#     df = pd.concat([tob_after2, tob_before, bid_1_pr_match], axis=1)
#     #print(df_bid_1_price.sum())

# %% Run tests CME

# run_intensity_multi_days(PATHIN, PATHOUT, TS, [], 'CME')

# run_event_data(PATHIN, PATHOUT, FILE1, TS, START_TIME, END_TIME)

# %% Run tests BMF - DOL

run_event_data(PATHIN, PATHOUT, FILE_BMF1, TS, START_TIME1, END_TIME1, 'BMF')

#%% Intensity columns

INT_COLUMNS = ['order_type', 'size_before', 'var_DateTime', 'Number',
               'Intensity']
PIVOT_COLUMNS = ['var_DateTime', 'Number', 'Intensity']

# %% Use files and plot - DOL

DF_INT_BID_DOL = pd.read_csv(PATHOUT+'df_intensity_bid.csv')
DF_INT_ASK_DOL = pd.read_csv(PATHOUT+'df_intensity_ask.csv')
DF_INT_BID_DOL = DF_INT_BID_DOL[INT_COLUMNS].copy()
DF_INT_ASK_DOL = DF_INT_ASK_DOL[INT_COLUMNS].copy()
DF_INT_BID_DOL['size_before'] = DF_INT_BID_DOL['size_before']/5
DF_INT_ASK_DOL['size_before'] = DF_INT_ASK_DOL['size_before']/5
DF_INT_DOL = DF_INT_BID_DOL.pivot(index='size_before',
                                  columns='order_type',
                                  values='Intensity')
DF_INT_DOL.loc[:40].plot()

# %% Run tests BMF - WDO

run_event_data(PATHIN, PATHOUT, FILE_BMF2, TS, START_TIME1, END_TIME1, 'BMF')

# %% Use files and plot - WDO

DF_INT_BID_WDO = pd.read_csv(PATHOUT+'df_intensity_bid.csv')
DF_INT_ASK_WDO = pd.read_csv(PATHOUT+'df_intensity_ask.csv')
DF_INT_BID_WDO = DF_INT_BID_WDO[INT_COLUMNS].copy()
DF_INT_ASK_WDO = DF_INT_ASK_WDO[INT_COLUMNS].copy()
DF_INT_BID_WDO['size_before'] = DF_INT_BID_WDO['size_before']
DF_INT_ASK_WDO['size_before'] = DF_INT_ASK_WDO['size_before']
DF_INT_WDO = DF_INT_BID_WDO.pivot(index='size_before',
                                  columns='order_type',
                                  values='Intensity')
DF_INT_WDO.loc[:80].plot()
