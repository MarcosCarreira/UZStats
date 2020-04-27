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
import matplotlib.pyplot as plt 
import Plotting as pltg
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




def Build_Q_no_regen(IntensVal,q,Qmax0):

    " IntensVal structure is pd.DataFrame(np.zeros((Qmax0*Qmax0,4)),columns=['BidQtyBefore','AskQtyBefore','lambdaCancel','lambdaIns']) "

    " q is the minimum order size "

    " Qmax0 is the maximum order size level "

    " RegenVect structure is np.zeros((2*Qmax0,Qmax0*Qmax0))"
    ## Build transition matrix finite difference scheme

    Matrix0 = np.zeros((Qmax0*Qmax0,Qmax0*Qmax0))

    for qsame in range(Qmax0) : # QSame Loop // qsame = 1

        for qopp in range(Qmax0) : # QOpp Loop // qopp = 1

            CumIntens = 0.

            ## Cancellation order bid side : 

            if (qsame > 0) : ## the limit is not totally consumed  // No regeneration

                 CumIntens +=  IntensVal['lambdaCancel'][qsame*Qmax0+qopp]  

                 Matrix0[qsame*Qmax0+qopp][(qsame-1)*Qmax0+qopp] += IntensVal['lambdaCancel'][qsame*Qmax0+qopp] 

            ## Cancellation order ask side :

            if (qopp > 0) : ## the limit is not totally consumed // no regeneration

                 CumIntens +=  IntensVal['lambdaCancel'][qopp*Qmax0+qsame]  

                 Matrix0[qsame*Qmax0+qopp][qsame*Qmax0+qopp-1] += IntensVal['lambdaCancel'][qopp*Qmax0+qsame] 

            ## Insertion order bid side :

            if (qsame < Qmax0-1) : ## when qsame = Qmax -1  no more order can be added to the bid limit

                 CumIntens +=  IntensVal['lambdaIns'][qsame*Qmax0+qopp]  

                 Matrix0[qsame*Qmax0+qopp][(qsame+1)*Qmax0+qopp] += IntensVal['lambdaIns'][qsame*Qmax0+qopp]            

            ## Insertion oder ask side

            if (qopp < Qmax0-1) : ## when qopp = Qmax -1  no more order can be added to the ask limit

                 CumIntens +=  IntensVal['lambdaIns'][qopp*Qmax0+qsame]  

                 Matrix0[qsame*Qmax0+qopp][qsame*Qmax0+qopp+1] += IntensVal['lambdaIns'][qopp*Qmax0+qsame]  

            ## Nothing happen 

            Matrix0[qsame*Qmax0+qopp][qsame*Qmax0+qopp] += - CumIntens  

    return Matrix0

## Building matrices ::
def Compute_intens_val_bis(IntensVal_whole_market, Qmax0 = 30, q = 1):
    res_sub = IntensVal_whole_market[IntensVal_whole_market['size_before'] <= Qmax0]
    
    IntensVal = pd.DataFrame(np.zeros((Qmax0*Qmax0,4)), columns = ['BidQtyBefore', 'AskQtyBefore', 'lambdaCancel', 'lambdaIns'])
    IntensVal['BidQtyBefore'] = np.repeat(np.arange(1,Qmax0+1)*q,Qmax0)
    IntensVal['AskQtyBefore'] = np.tile(np.arange(1,Qmax0+1)*q,Qmax0)
    IntensVal['lambdaCancel'] = np.repeat(res_sub[(res_sub['order_type'] == 'Consumption') ]['Intensity'].values,Qmax0)
    IntensVal['lambdaIns'] = np.repeat(res_sub[(res_sub['order_type'] == 'Insertion') ]['Intensity'].values,Qmax0)
    return IntensVal

## compute stationary probabilities

def Proba_stat(Tilde_Q,Qmax0): 

    size = Qmax0*Qmax0
    Tilde_Q_inv = np.array(Tilde_Q[:-1,:-1]) 
    for j in range(size-1):

        Tilde_Q_inv[:,j] -= Tilde_Q[-1,j]  
    F_inv = -Tilde_Q[size-1,:-1]
    ## Compute the stat proba
    Proba2 = np.zeros((size))
    Proba2[:-1] = np.linalg.solve(Tilde_Q_inv.transpose(),F_inv.transpose());Proba2[-1]  = 1-sum(Proba2)
    return Proba2
    
def run_intensity_process(pathin, bid_or_ask='bid', option_save = False):
    
    if bid_or_ask == 'bid':
        file_input = 'df_intensity_bid.csv'
    else:
        file_input = 'df_intensity_ask.csv'
    intens = pd.read_csv(pathin+file_input)
    
    ### Compute intensities
    ##### Initialize parameters
    q = 1
    Qmax0 = 30

    ##### Compute intensities
    IntensVal = Compute_intens_val_bis(intens, Qmax0 = Qmax0, q = q)
    
    ##### Test result :: plot values 
    ###### Initialize parameters
    order_type_1 = 'lambdaCancel'
    indexes_limit_1 = np.arange(0,Qmax0)*Qmax0
    order_type_2 = 'lambdaIns'
    #option_save = ""#save"

    ###### Plot values
    plt.plot(IntensVal['BidQtyBefore'][indexes_limit_1],IntensVal[order_type_1][indexes_limit_1], label ='Liquidity consumption', linewidth= 3.0)
    plt.plot(IntensVal['BidQtyBefore'][indexes_limit_1],IntensVal[order_type_2][indexes_limit_1], label = 'Liquidity provision', linewidth= 3.0)
    plt.title('')
    plt.grid()
    plt.legend(loc = 2, bbox_to_anchor = (.01,.92))
    if option_save == "save" :
        plt.savefig(pathin+"\\intensity_all_f_"+".pdf", bbox_inches='tight')
    plt.show()


    ### Compute Q matrix/proba stationary for all market agent :: without regeneration 
    ###### Initialize the parameters
    #option_save = True
    path = pathin; ImageName = "\\Proba_stat_all_agents_ff"
    
    ##### Computation
    Q_no_regen = Build_Q_no_regen(IntensVal,q,Qmax0)
    
    ### Compute stationary probabilities 
    ##### Computation
    proba = Proba_stat(Q_no_regen,Qmax0)
    
    ##### Plot the values
    ####### First method
    xpos1 = np.repeat(q*np.arange(1,Qmax0+1),Qmax0)
    ypos1 = np.tile(q*np.arange(1,Qmax0+1),Qmax0)
    data_frame = pd.DataFrame(np.zeros((Qmax0*Qmax0,3)),columns=['x','y','Prob'])
    data_frame['x'] = xpos1
    data_frame['y'] = ypos1
    data_frame['Prob'] = proba
    pltg.Plot_sns(data_frame.pivot("y", "x", "Prob"),option_save,path,ImageName, cbar = True, annot = False)

    
    ####### Second method
    Resx = q*np.arange(Qmax0);Resy = q*np.arange(Qmax0); Resz=proba; xlabel='Bid size';ylabel='Ask size'; zlabel='Joint distribution'; optionXY=2
    elev0= 20; azim0=20; dist0= 12; bins =Qmax0 
    path = pathin; ImageName ="\\Proba_stat_3D"; xtitle = ""
    #option="save";path ="D:\\etude\\charles-albert\\Past_Trades_Influence\\Estimator_Cont_Annul\\CompareModelQR_NewMod\\Model_PastTrades_ModelPropg\\Image";ImageName="\\ProbStatRegen_2f";xtitle="" 
    pltg.Plot3D(Resx,Resy,Resz,Qmax0,xlabel,ylabel,zlabel,option_save,path,ImageName,xtitle,elev0, azim0, dist0,optionXY)
    
    ####### Third method
    labels = [""]
    path = pathin; ImageName ="\\Proba_stat_all"; xtitle = ""
    xpos1 = np.repeat(q*np.arange(1,Qmax0+1),Qmax0)
    ypos1 = np.tile(q*np.arange(1,Qmax0+1),Qmax0)
    data_frame = pd.DataFrame(np.zeros((Qmax0*Qmax0,3)),columns=['x','y','Prob'])
    data_frame['x'] = xpos1
    data_frame['y'] = ypos1
    data_frame['Prob'] = proba
    res_bis = data_frame.groupby(['x']).agg({'Prob':'median'})
    df = [[res_bis.index.values, res_bis.values.flatten()/res_bis.values.sum()]]
    pltg.Plot_plot(df,labels,option_save,path,ImageName,xtitle = bid_or_ask, Nset_tick_x = False)

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
#run_event_data(PATHIN, PATHOUT, FILE1, TS, START_TIME, END_TIME)
run_intensity_process(PATHOUT, 'ask')
run_intensity_process(PATHOUT, 'bid')

#run_BFM_tob(PATHIN, PATHOUT, FILE_BMF, TS, START_TIME, END_TIME)
#run_compare_tob(PATHIN, PATHOUT)
#run_tob(PATHIN, PATHOUT, FILE1, TS, START_TIME, END_TIME)
#run_benchmark(PATHIN, PATHOUT, FILE1, TS, START_TIME, END_TIME)
#run_unc_zones_read(PATHIN, PATHOUT, FILE1, TS, START_TIME, END_TIME)
#runc_multi_days(PATHIN, PATHOUT, TS, START_TIME, END_TIME)
