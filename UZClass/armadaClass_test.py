#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 08:40:11 2020

@author: huchede
"""

# %% Input examples

import os
import timeit
import pandas as pd
from armadaClass import ArmadaData_UZModel as uz
from armadaClass import Armada_Data as ad
from armadaClass import Armada_TOB as atob
import numpy as np

#from armadaClass import Armada_UZModel_output as auo


PATHPROJ = os.path.join(os.path.expanduser("~"), "Documents", "GitHub",\
                       "UZStats")
PATHIN = PATHPROJ+'/CLOB_data/'
PATHOUT = PATHPROJ+'/UZClass/'

TS = 0.5
START_TIME = pd.to_timedelta('07:30:00')
END_TIME = pd.to_timedelta('12:45:00')

FILE1 = '20180105_6EH8.zip'
FILE_BMF = 'DOLG1720170119.csv'

def run_event_data(pathin, pathout, file_name, tick_value, start_time,\
                  end_time, save_files=False):
    '''run_unc_zones(pathin, pathout, file_name, tick_value, end_of_time)
    returns the uncertainty zones data frame'''
    data = ad(pathin,file_name)
    print(data.get_processing_date())
    tob_obj = atob(data, tick_value)
    tob_obj.print2file_df_intensity(pathout)



def run_intensity_one_days(pathin, pathout, tick_value, file_name = []):
    data = ad(pathin,file_name)
    print(data.get_processing_date())
    tob_obj = atob(data, tick_value)
    output = tob_obj.get_tob_intensity_output()
    bid_inten = output.get_bid_intensity()
    ask_inten = output.get_ask_intensity()
    both_inten = output.get_aggregated_bid_ask()
    bid_inten.plot_intensities(pathout, 'bid')
    ask_inten.plot_intensities(pathout, 'ask')
    both_inten.plot_intensities(pathout, 'bid_plus_ask')
    bid_inten.plot_proba_stat(pathout, 'bid')
    #ask_inten.plot_proba_stat(pathout, 'ask')
    #both_inten.plot_proba_stat(pathout, 'bid_plus_ask')

def run_intensity_multi_days(pathin, pathout, tick_value, file_names = []):
    
    tick_value = TS
    filepaths = [pathout]
    #create directories if do not exist
    for path in filepaths:
        if not os.path.exists(path):
            os.makedirs(path)
            
    #either explicitly set file_names or get file_names from data path
    if len(file_names)==0:
        for file in os.listdir(pathin):
            if file.endswith("csv") or file.endswith(".zip"):
                file_names.append(file)
    
    for f in file_names:
        start = timeit.default_timer()
        print('--START------')
        data = ad(pathin,f)
        tob_obj = atob(data, tick_value)
    
        #uz_obj = uz(data,tick_value,start_time,end_time)
        if file_names[0]==f:
            output = tob_obj.get_tob_intensity_output()
        else:
            output.append(tob_obj.get_tob_intensity_output())
        stop = timeit.default_timer()
        print('Time Spent: ', round(stop - start), ' seconds')
        print('--END-------')
        
    
        
    intensity = output.get_aggregated_bid_ask()
    intensity.plot_intensities(pathout, True)
    intensity.plot_proba_stat(pathout, True)
    #output.print2file_df_uz_stats(pathout)
    #output.plot_html_uz_stats(pathout)


def runc_multi_days(pathin, pathout, tick_value, start_time, end_time, file_names = []):
    
    tick_value = TS
    filepaths = [pathout]
    #create directories if do not exist
    for path in filepaths:
        if not os.path.exists(path):
            os.makedirs(path)
            
    #either explicitly set file_names or get file_names from data path
    if len(file_names)==0:
        for file in os.listdir(pathin):
            if file.endswith("csv") or file.endswith(".zip"):
                file_names.append(file)
    
    for f in file_names:
        start = timeit.default_timer()
        print('--START------')
        data = ad(pathin,f)
        uz_obj = uz(data,tick_value,start_time,end_time)
        if file_names[0]==f:
            output = uz_obj.get_Armada_UZModel_output()
        else:
            output.append(uz_obj.get_Armada_UZModel_output())
        stop = timeit.default_timer()
        print('Time Spent: ', round(stop - start), ' seconds')
        print('--END-------')
        
    output.print2file_df_cont_alt_by_ticks(pathout)
    output.print2file_df_uz_stats(pathout)
    output.plot_html_uz_stats(pathout)
        
        

def run_unc_zones_read(pathin, pathout, file_name, tick_value, start_time,\
                  end_time, save_files=False):
    '''run_unc_zones(pathin, pathout, file_name, tick_value, end_of_time)
    returns the uncertainty zones data frame'''
    data = ad(pathin,file_name)
    data.plot_html_ohlc(pathout,'5min', pd.to_timedelta('00:00:00'),pd.to_timedelta('23:59:00'))
    #data.plot_html_ohlc(pathout,'1min', pd.to_timedelta('07:00:00'),pd.to_timedelta('10:00:00'))
    #data.plot_html_ohlc(pathout, '1S', pd.to_timedelta('07:27:00'),pd.to_timedelta('07:33:00'))
    uz_obj = uz(data,tick_value,start_time,end_time)
    #ohlc = uz_obj.ohlc(pathout)
    uz_obj.print2file_df_cont_alt_by_ticks(pathout)
    uz_obj.print2file_df_uz_stats(pathout)
    data.plot_html_1mintick(pathout,pd.to_timedelta('07:29:50'))

def run_tob(pathin, pathout, file_name, tick_value, start_time,\
                  end_time, save_files=False):
    '''run_unc_zones(pathin, pathout, file_name, tick_value, end_of_time)
    returns the uncertainty zones data frame'''
    data = ad(pathin,file_name)
    print(data.get_processing_date())
    tob_obj = atob(data, tick_value)
    #start = '07:29:50'
    #end = '07:50:10'
    tob_obj.plot_html_tob_1mintick(pathout,pd.to_timedelta('07:29:50'))
    #agg = tob_obj.get_net_number_aggression(pd.to_timedelta(start),pd.to_timedelta(end))
    #print('# of aggression between ',start, end,'is ',  agg)
    #tob_obj.plot_html_1sec_rolling_number_aggression(pd.to_timedelta(start), pd.to_timedelta(end), pathout)
    #data.plot_html_1mintick(pathout,pd.to_timedelta('07:29:50'))
    #tob_obj.print2file_df_tob(pathout, start_time, end_time)

def run_benchmark(pathin, pathout, file_name, tick_value, start_time,\
                  end_time):

    import armadauzdf
    
    armadauzdf.run_unc_zones(pathin, pathout, file_name, tick_value, start_time,\
                  end_time, 9.25, False)

def run_BFM_tob(pathin, pathout, file_name, tick_value, start_time,\
                  end_time, save_files=False):
    data = ad(pathin,file_name, 'BMF')
    print(data.file_name)
    print(data.processing_date) # test if bug if fixed
    data.plot_html_ohlc(pathout,'1min', pd.to_timedelta('00:00:00'),pd.to_timedelta('23:59:00'))
    tob_obj = atob(data, tick_value)
    tob_obj.print2file_df_tob(pathout, start_time, end_time)
    data.plot_html_1mintick(pathout,pd.to_timedelta('08:59:55'))
    uz_obj = uz(data,tick_value,start_time,end_time)
    uz_obj.print2file_df_cont_alt_by_ticks(pathout)
    uz_obj.print2file_df_uz_stats(pathout)

    print('done')
    
def run_compare_tob(pathin, pathout, file_names = []):
    
    file_before = 'tob_before.zip'
    file_after = 'tob.zip'
    tob_before = pd.read_csv(pathout+file_before)
    tob_after = pd.read_csv(pathout+file_after)
    tob_after2 = tob_after[3:699774]
    
    tob_before.rename(columns = {'Bid 1 Qty':'bid_1_qty',\
                                       'Bid 2 Qty':'bid_2_qty',\
                                       'Bid 1 Price':'bid_1_price',\
                                       'Bid 2 Price':'bid_2_price',\
                                       'Bid 1 Ord':'bid_1_ord',\
                                       'Ask 1 Ord':'ask_1_ord',\
                                       'Bid 2 Ord':'bid_2_ord',\
                                       'Ask 2 Ord':'ask_2_ord',\
                                       'Ask 1 Qty':'ask_1_qty',\
                                       'Ask 2 Qty':'ask_2_qty',\
                                       'Ask 1 Price':'ask_1_price',\
                                       'Ask 2 Price':'ask_2_price',\
                                       'Trade Price':'trade_price',\
                                       'Trade Qty':'trade_qty',\
                                    'Aggression':'aggression',\
                                           }, inplace = True)
    #tob_before.drop(['bid_2_qty', 'bid_2_price','bid_2_ord','ask_2_ord',\
    #            'ask_2_qty', 'ask_2_price', 'OT'],axis=1, inplace=True)
    #tob_after2.drop(['bid_1_price_last', 'ask_1_price_last','bid_price_traded'\
    #                 ,'ask_price_traded'],axis=1, inplace=True)
    
    tob_before = \
            tob_before.set_index(tob_before['DateTime'])
    tob_after2 = \
            tob_after2.set_index(tob_after2['DateTime'])
            
        
    bid_1_pr_match = pd.DataFrame( np.where(tob_after2['bid_1_price'] == tob_before['Bid_Price'], True, False), index= tob_before.index)
    #df_bid_1_price['before']=tob_before['Bid_Price'].copy()
    #df_bid_1_price['after']=tob_after2['bid_1_price'].copy()
    df = pd.concat([tob_after2, tob_before, bid_1_pr_match], axis=1)
    #print(df_bid_1_price.sum())
    
# %% Run test
run_intensity_one_days(PATHIN, PATHOUT, TS, FILE1)
#run_intensity_multi_days(PATHIN, PATHOUT, TS)
#run_event_data(PATHIN, PATHOUT, FILE1, TS, START_TIME, END_TIME)

#run_BFM_tob(PATHIN, PATHOUT, FILE_BMF, TS, START_TIME, END_TIME)
#run_compare_tob(PATHIN, PATHOUT)
#run_tob(PATHIN, PATHOUT, FILE1, TS, START_TIME, END_TIME)
#run_benchmark(PATHIN, PATHOUT, FILE1, TS, START_TIME, END_TIME)
#run_unc_zones_read(PATHIN, PATHOUT, FILE1, TS, START_TIME, END_TIME)
#runc_multi_days(PATHIN, PATHOUT, TS, START_TIME, END_TIME)
