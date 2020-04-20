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
#from armadaClass import Armada_UZModel_output as auo


PATHPROJ = os.path.join(os.path.expanduser("~"), "Documents", "GitHub",\
                       "UZStats")
PATHIN = PATHPROJ+'/CLOB_data/'
PATHOUT = PATHPROJ+'/UZClass/'

TS = 0.5
START_TIME = pd.to_timedelta('07:30:00')
END_TIME = pd.to_timedelta('18:15:00')

FILE1 = '20180105_6EH8.zip'
#FILE1 = '20190514_6EM9.zip'

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
    tob_obj = atob(data)
    start = '07:29:50'
    end = '07:50:10'
    agg = tob_obj.get_net_number_aggression(pd.to_timedelta(start),pd.to_timedelta(end))
    print('# of aggression between ',start, end,'is ',  agg)
    #tob_obj.plot_html_1sec_rolling_number_aggression(pd.to_timedelta(start), pd.to_timedelta(end), pathout)
    #data.plot_html_1mintick(pathout,pd.to_timedelta('07:29:50'))
    tob_obj.print2file_df_tob(pathout)

def run_benchmark(pathin, pathout, file_name, tick_value, start_time,\
                  end_time):

    import armadauzdf
    
    armadauzdf.run_unc_zones(pathin, pathout, file_name, tick_value, start_time,\
                  end_time, 9.25, False)
        
    
    
# %% Run test
#run_tob(PATHIN, PATHOUT, FILE1, TS, START_TIME, END_TIME)
run_benchmark(PATHIN, PATHOUT, FILE1, TS, START_TIME, END_TIME)
#run_unc_zones_read(PATHIN, PATHOUT, FILE1, TS, START_TIME, END_TIME)
#runc_multi_days(PATHIN, PATHOUT, TS, START_TIME, END_TIME)
