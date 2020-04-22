
# -*- coding: utf-8 -*-
'''
The Robert and Rosenbaum Uncertainty Zones model
Applied to Armada Top of Book Level 2 with trades

Implementation by
Marcos Costa Santos Carreira (École Polytechnique - CMAP)
and
Florian Huchedé (CME)
May-2019

'''

# %% Import packages
import timeit
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# %% Armada Data Class

class Armada_Data():

# %% Main Functions    
    
    def __init__(self, file_path, file_name, exchange = 'CME'):
        self.__exchange = exchange
        self.__file_name = file_name
        self.__file_path = file_path
        self.__df = pd.DataFrame()
        self.__measures = pd.DataFrame()
        
        print('Reading file '+ self.file_entire_path)
        self.__read_ArmadaData()
        print('Re-formatting data')
        self.__column_datetime()
        self.__rename_columns()
        print('Construct Data Frame with Basic Measures')        
        self.__calc_basic_measures() # Marcos to check if needed here or get() function
        print('Remove data outside exchange trading hours')
        self.__filter_exchange_data_by_time()
        print('Armada Data Object Construction Successfull')
        
    
    @property
    def file_name(self):
            return self.__file_name
    @property
    def file_name_long(self):
            return self.__file_name[:-4]
    @property
    def file_entire_path(self):
        return (self.__file_path+self.__file_name)
    @property
    def exchange(self):
        return self.__exchange
    @property
    def df(self):
        return self.__df
    @property
    def measures(self):
        return self.__measures
    
    # %% Public Functions
    
    def get_processing_date(self):
        if self.__exchange == 'CME':
            return pd.to_datetime(self.file_name[0:8],format='%Y%m%d')
        if self.__exchange == 'BMF':
            return pd.to_datetime(self.file_name[6:-4],format='%Y%m%d')
        
    def get_exchange_starting_time(self):
        if self.__exchange == 'CME':
            return self.get_processing_date()\
                                + pd.to_timedelta('00:00:00')
        if self.__exchange == 'BMF':
            return self.get_processing_date()\
                                + pd.to_timedelta('09:00:00')
                                
    def get_exchange_end_time(self):
        if self.__exchange == 'CME':
            return self.get_processing_date()\
                                + pd.to_timedelta('16:00:00')
        if self.__exchange == 'BMF':
            return self.get_processing_date()\
                                + pd.to_timedelta('23:59:59')
# %% Main Functions
    
    def __read_ArmadaData(self):
        self.__df = pd.read_csv(self.file_entire_path \
                                          ,dtype={'Index':int,
                                                'Date':str,
                                                'Time':str,
                                                'Bid 1 Qty':float,
                                                'Bid 2 Qty':float,
                                                'Bid 1 Price':float,
                                                'Bid 2 Price':float,
                                                'Bid 1 Ord':float,
                                                'Bid 2 Ord':float,
                                                'Trade Price':float,
                                                'Trade Qty':float,
                                                'Ask 1 Qty':float,
                                                'Ask 2 Qty':float,
                                                'Ask 1 Price':float,
                                                'Ask 2 Price':float,
                                                'Ask 1 Ord':float,
                                                'Ask 2 Ord':float
                                                })
    def __rename_columns(self):
        self.__df.rename(columns = {'Bid 1 Qty':'bid_1_qty',\
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
                                           }, inplace = True)
        self.__df.drop(['Time.1', 'Date.1','Index','Index.1',\
                              'Date', 'Time'],axis=1, inplace=True)   
    def __calc_basic_measures(self):
        self.__get_ba_spread()
        self.__get_delta_t()
        self.__get_mid_price()
        self.__trade_indicator()  
    
    def __filter_exchange_data_by_time(self):
        # set start and end time
        start_time = self.get_exchange_starting_time()
        end_time = self.get_exchange_end_time()
        
        # truncating df
        df_tmp = self.__df.copy()
        df_tmp = df_tmp.set_index(['DateTime'])
        df_tmp = df_tmp.loc[start_time:end_time]
        df_tmp = df_tmp.reset_index()
        self.__df = df_tmp        
        # truncating df
        measures_tmp = self.__measures.copy()
        measures_tmp = measures_tmp.set_index(self.__df.DateTime)
        measures_tmp = measures_tmp.loc[start_time:end_time]
        measures_tmp = measures_tmp.reset_index()
        self.__measures = measures_tmp

# %% Measures Functions 
        
    def __trade_indicator(self):
        self.measures['trade_indicator'] = self.__df.trade_price.isnull().copy()
    
    def __get_ba_spread(self, data=None):
        if data is not None:
            spread = (data.ask_1_price - data.bid_1_price)
            return spread
        else:
            df = self.__df.copy()
            self.measures['ba_spread'] = (df.ask_1_price - df.bid_1_price)    
    
    def __get_delta_t(self, data=None):
        if data is not None:
            delta_t = self.df.DateTime.diff().shift(-1)
            return delta_t
        else:
            self.measures['dt'] = self.__df.DateTime.diff().shift(-1) 
    
    def __get_mid_price(self, data=None):
        if data is not None:
            mid_price = (data.ask_1_price + data.bid_1_price)/2
            return mid_price
        else:
            self.measures['mid_price'] = \
                (self.__df.ask_1_price + self.__df.bid_1_price)/2
    
    def __rlz_vol_log(self,prices):
        '''rlzvollog(prices) calculates the realized volatility of a time series
    of prices using logreturns
    Inputs: Time Series of prices
    Outputs: scalar with the volatility for the period'''
        pxs = np.log(prices/prices.shift(1))
        self.measures['realized_vol']=np.sqrt(np.sum(pxs*pxs))
        
    def __custom_resampler_abs_sum(self,array):
        return np.sum(np.abs(array))
    
    def __get_time_weighted_spread(self, data=None):
        if data is not None:    
            return ((data['ba_spread']*data['dt']).sum())/\
                (data['dt'].sum())
            return ((self.__df['ba_spread']*self.__df['dt']).sum())/\
                (self.__df['dt'].sum())
    
    def __column_datetime(self):
        '''column_datetime(data_frame) creates the column 'DateTime'
        concatenating 'Date' and 'Time' and converting it to datetime.
        Inputs: Data Frame from Armada
        Outputs: Copy of DataFrame with 'DateTime' column added'''
        data_framec = self.__df.copy()
        data_framec['Date'] = pd.to_datetime(data_framec['Date'],\
            format="%m/%d/%Y")
        data_framec['Time'] = pd.to_timedelta(data_framec['Time'], unit='ns')
        data_framec['DateTime'] = data_framec['Date']+data_framec['Time']
        self.__df = data_framec

# %%  Plots Functions

    def plot_html_1mintick(self,path_out, start_xaxis = pd.to_timedelta('07:30:00')):
        #layout = self.__plot_html_layout('Bid Price', 'Ask Price', 'Price Time Series')
        data = self.__df.copy()
        data = data.set_index(['DateTime'])
        
        start_xaxis = self.get_processing_date() + start_xaxis
        end_xaxis = start_xaxis + pd.to_timedelta('00:01:00')
        
        data = data.loc[start_xaxis:end_xaxis]
        
        #data['trade_price_minus_1']=data.trade_price.shift(-1)
        #data['buy']=np.where((data['trade_price_minus_1']==data.bid_1_price)
        #                     & (data['trade_price_minus_1'].notna())
        #                     & (data.bid_1_price.notna())
        #                     , True, False)
        
        fig = make_subplots(rows=3, cols=1, row_width=[0.2, 0.2, 0.4], 
                            shared_xaxes=True)
        fig.add_trace(go.Scatter(x=data.index, y=data.bid_1_price, 
                            name = 'Bid'), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data.ask_1_price, 
                            name = 'Ask'), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data.trade_qty, 
                            name = 'Trade Qty'), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=self.__get_ba_spread(data), 
                            name = 'Bid-ask spread'), row=3, col=1)
        
        
        file = path_out+self.file_name_long+"_price.html"
        print('saving html plot to ', file)
        fig.write_html(file)
        
    def plot_html_qty(self,path_out):
        layout = self.__plot_html_layout('Bid Qty', 'Ask Qty', 'Quantity Time Series')
        bid_data = go.Scatter(x=self.__df.DateTime,
                         y=self.__df.bid_1_qty)
        ask_data = go.Scatter(x=self.__df.DateTime,
                        y=self.__df.ask_1_qty)
        fig = go.Figure(data=[bid_data, ask_data], layout=layout)
        fig.update_layout(height=600, width=600, title_text="Stacked Subplots")
        file = path_out+self.file_name_long+"_qty.html"
        print('saving html plot to ', file)
        fig.write_html(file)  
    
    
    def plot_html_ohlc(self, pathout, 
                       freq='1min',start_xaxis = pd.to_timedelta('09:00:00'),
                       end_xaxis = pd.to_timedelta('16:00:00')):
        
        data = self.__df.copy()
        data = data.set_index(['DateTime'])
        
        start_xaxis = self.get_processing_date() + start_xaxis
        end_xaxis = self.get_processing_date() + end_xaxis
        
        # prepare OHLC data
        data['mid_price'] = (data.ask_1_price + data.bid_1_price)/2
        ohlc = data.mid_price.resample(freq).ohlc()
        ohlc = ohlc.loc[start_xaxis:end_xaxis]
        
        
        # prepare Volume data
        volume = pd.DataFrame(data.trade_qty.resample(freq)\
                              .apply(self.__custom_resampler_abs_sum))
        #volume = volume.set_index(ohlc.index)    
        volume = volume.loc[start_xaxis:end_xaxis]
        
        # prepare time weigthed spread data
        data = data.assign(ba_spread=self.measures.ba_spread.values)
        data = data.assign(dt=self.measures.dt .values)
        data_masked = data[data.ba_spread > 0]
        time_weighted_spread = pd.DataFrame(\
            data_masked.groupby(pd.Grouper(freq=freq)).apply(self.__get_time_weighted_spread))
        #time_weighted_spread = time_weighted_spread.set_index(ohlc.index)    
        time_weighted_spread.columns = ['time_weighted_spread']
        time_weighted_spread = time_weighted_spread.loc[start_xaxis:end_xaxis]
        
        
        # plotting
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
        row_width=[0.2, 0.2, 0.4],vertical_spacing=0.1,
        subplot_titles=('Open High Low Close Candlestick', 'Volume', 
                        'Time Weighted Spread'))
        
        fig.add_trace(go.Candlestick(name='OHLC',
                x=ohlc.index,
                open=ohlc.open, high=ohlc.high,
                low=ohlc.low, close=ohlc.close,
                increasing_line_color= 'green', decreasing_line_color= 'red'),
                row=1, col=1)
        
    
        fig.add_trace(go.Bar(name='Volume', x=volume.index, \
                y=volume.trade_qty), row=2, col=1)
            
        fig.add_trace(go.Bar(x=time_weighted_spread.index, \
                y=time_weighted_spread.time_weighted_spread , 
                name = 'Time Weighted Spread', marker_color='black'), 
                row=3,col=1)
        
        fig.update_layout(
                xaxis=dict(rangeslider=dict(visible=False),type="date", 
                       )) 

          
        file = pathout+self.get_processing_date().strftime('%Y%m%d')\
                +'_'+freq+"_ohlc.html"
        print('saving html plot to ', file)
        fig.write_html(file)


# %% Armada Top-Of-Book Class
        
class Armada_TOB(Armada_Data):
    tick_value = float()
    def __init__(self, Armada_Data, tick_value):
        start = timeit.default_timer()
        self.Armada_Data = Armada_Data
        self.tick_value = tick_value
        self.tob = pd.DataFrame()
        #to delete once algo completed
        self.tob['traded_price'] = self.Armada_Data.df.trade_price.copy() 
        self.tob['traded_qty'] = self.Armada_Data.df.trade_qty.copy() 
        ######
        print('Filling intermediate values in order book')
        self.__fill_tob_data()
        self.__fill_price_traded()
        #self.__fill_aggression()
        self.num_consecutive_trade = self.__find_num_consecutive_trade()
        print(self.Armada_Data.file_name+' max_runs = '+str(self.num_consecutive_trade))
        self.__fill_tob()
        stop = timeit.default_timer()
        print('Time spent on top-of-book filling: ', round(stop - start), ' seconds')
        self.__depletion()
        print('Depletion completed')
        print('Calculating order indicators')
        self.__order_indicators()
        print('Order indicators completed')
        
    def get_data_for_tob_intensity(self):
        bid_qty_before = self.tob.bid_1_qty.shift(+1).copy()
        ask_qty_before = self.tob.ask_1_qty.shift(+1).copy()
        isconsumption = self.depl.depl_or_fill
        delta_t = self.Armada_Data.df.DateTime.diff().shift(+1)
        
        
        
    def __fill_tob(self):
        # initialize to original not filled tob
        self.tob['bid_1_price'] = self.Armada_Data.df.bid_1_price.copy()
        self.tob['ask_1_price'] = self.Armada_Data.df.ask_1_price.copy()
        self.tob['bid_1_qty'] = self.Armada_Data.df.bid_1_qty.copy()
        self.tob['ask_1_qty'] = self.Armada_Data.df.ask_1_qty.copy()
        
        self.tob['bool_trade'] = self.Armada_Data.measures.trade_indicator.copy() == False
        
        
        # if trade happened        
        for i in reversed(range(self.num_consecutive_trade + 1)):
            
            self.tob['bool_idx'] = self.tob['cumsum'].astype('int64') == i
            shift_bid_price = self.tob.bid_1_price.shift(-1)
            shift_bid_qty = self.tob.bid_1_qty.shift(-1)
            shift_ask_price = self.tob.ask_1_price.shift(-1)
            shift_ask_qty = self.tob.ask_1_qty.shift(-1)
            
            #If ask or bid price t+1 = trade price t 
            self.tob['bool_bid_D'] = self.Armada_Data.df.trade_price == shift_bid_price
            self.tob['bool_ask_D'] = self.Armada_Data.df.trade_price == shift_ask_price      
            
            self.tob.bid_1_qty.loc[    \
                self.tob.bool_idx &    \
                self.tob.bool_trade &  \
                self.tob.bool_bid_D] = \
                pd.Series(np.sum([self.Armada_Data.df.trade_qty , shift_bid_qty], axis=0))
            
            self.tob.bid_1_price.loc[    \
                self.tob.bool_idx &    \
                self.tob.bool_trade &  \
                self.tob.bool_bid_D] = self.Armada_Data.df.trade_price
                
            self.tob.ask_1_qty.loc[    \
                self.tob.bool_idx &    \
                self.tob.bool_trade &  \
                self.tob.bool_ask_D] = \
                pd.Series(np.sum([self.Armada_Data.df.trade_qty , shift_ask_qty], axis=0))
                
            self.tob.ask_1_price.loc[    \
                self.tob.bool_idx &    \
                self.tob.bool_trade &  \
                self.tob.bool_ask_D] = self.Armada_Data.df.trade_price

            #If ask or bid price t+1 != trade price t 
            self.tob['bool_bid_C'] = self.Armada_Data.df.trade_price != shift_bid_price
            self.tob['bool_ask_C'] = self.Armada_Data.df.trade_price != shift_ask_price      
            
            self.tob.ask_1_qty.loc[    \
                self.tob.bool_idx &    \
                self.tob.bool_trade &  \
                self.tob.bool_ask_C] = self.Armada_Data.df.trade_qty
                
            self.tob.ask_1_price.loc[    \
                self.tob.bool_idx &    \
                self.tob.bool_trade &  \
                self.tob.bool_ask_C] = self.Armada_Data.df.trade_price
                
            self.tob.bid_1_qty.loc[    \
                self.tob.bool_idx &    \
                self.tob.bool_trade &  \
                self.tob.bool_bid_C] = self.Armada_Data.df.trade_qty
                
            self.tob.bid_1_price.loc[    \
                self.tob.bool_idx &    \
                self.tob.bool_trade &  \
                self.tob.bool_bid_C] = self.Armada_Data.df.trade_price
            
            # if side not traded: 
            #{Qty_Other_Side[t] , Price_Other_Side[t]} = 
            #{Qty_Other_Side[t+1] , Price_Other_Side[t+1]} # side not traded

            self.tob.ask_1_price.loc[    \
                self.tob.bid_price_traded & \
                self.tob.bool_idx] = np.nan #shift_ask_price
                
            self.tob.ask_1_qty.loc[
                self.tob.bid_price_traded & 
                self.tob.bool_idx] = np.nan #shift_ask_qty
            
            self.tob.bid_1_price.loc[    \
                self.tob.ask_price_traded & \
                self.tob.bool_idx] = np.nan #shift_bid_price
                
            self.tob.bid_1_qty.loc[
                self.tob.ask_price_traded & 
                self.tob.bool_idx] = np.nan #shift_bid_qty
        
        #self.tob.drop(['bool_bid_C', 'bool_ask_C','bool_ask_D','bool_bid_D',\
        #                      'bool_idx', 'bool_trade', 'cumsum'],axis=1, inplace=True)
        self.tob.ask_1_price.fillna(method='ffill', inplace=True)
        self.tob.ask_1_qty.fillna(method='ffill', inplace=True)
        self.tob.bid_1_price.fillna(method='ffill', inplace=True)
        self.tob.bid_1_qty.fillna(method='ffill', inplace=True)
        
    def __fill_tob_data(self):
        self.tob['bid_1_price_last'] = self.Armada_Data.df.bid_1_price.copy()
        self.tob.bid_1_price_last.fillna(method='ffill', inplace=True)
        
        self.tob['ask_1_price_last'] = self.Armada_Data.df.ask_1_price.copy()
        self.tob.ask_1_price_last.fillna(method='ffill', inplace=True)
       
    def __fill_price_traded(self):
        self.tob['bid_price_traded'] = self.tob.bid_1_price_last >=\
        self.Armada_Data.df.trade_price
        
        self.tob['ask_price_traded'] = self.tob.ask_1_price_last <=\
        self.Armada_Data.df.trade_price
    
    def __fill_aggression(self):
        self.tob['aggression'] = np.vectorize(self.aggression_id)\
        (self.tob.bid_price_traded, self.tob.ask_price_traded,\
        self.Armada_Data.measures.trade_indicator)
            
    def get_net_number_aggression(self, start_time, end_time):
        start = self.Armada_Data.get_processing_date() + start_time
        end = self.Armada_Data.get_processing_date() + end_time
        df_aggression = pd.DataFrame(self.tob.aggression)
        df_aggression = \
            df_aggression.set_index(self.Armada_Data.df.DateTime.copy())
        df_aggression = df_aggression.loc[start:end]
        number_aggression = np.nansum(df_aggression)
        return number_aggression
    
    def get_rolling_number_aggression(self, start_time, end_time):
        date = self.Armada_Data.get_processing_date()
        start_lag =  date + start_time - pd.to_timedelta('00:00:01')
        start = date + start_time
        end = date + end_time
        
        df_aggression = pd.DataFrame(self.tob.aggression)
        df_aggression = \
            df_aggression.set_index(self.Armada_Data.df.DateTime.copy())
        df_aggression = df_aggression.loc[start_lag:end]
        #df_aggression = abs(df_aggression)
        number_aggression = df_aggression.rolling('1s').sum()
        
        number_aggression = number_aggression.loc[start:end]
        return number_aggression
    
    def plot_html_1sec_rolling_number_aggression(self, start_time, end_time, path_out):
        number_aggression = self.get_rolling_number_aggression(start_time, end_time)
        
        #layout = self.__plot_html_layout('Bid Qty', 'Ask Qty', 'Quantity Time Series')
        fig = go.Figure([go.Scatter(x=number_aggression.index, y=number_aggression.aggression)])
        fig.update_layout(title_text="1 Second Rolling Net Number of Aggression")
        file = path_out+self.Armada_Data.file_name+"_1sec_roll_net_number_aggression.html"
        print('saving html plot to ', file)
        fig.write_html(file)  
        
    def __find_num_consecutive_trade(self):
        boolean_series = self.Armada_Data.measures.trade_indicator == False
        cumsum = boolean_series.cumsum()
        temp = cumsum.sub(cumsum.mask(boolean_series).ffill().fillna(0)).astype(int)
        num_consecutive_trade = temp.max()
        ###can be deleted later#
        self.tob['cumsum'] = temp
        ###
        return num_consecutive_trade
    
    # %% Depletions

    def __depletion(self):
        '''depletions(data_frame, tick_value) flags depletions and fills on the
        queue calculated by df_previous_tob returning new fields that indicate
        which side (Bid or Ask) was depleted or filled and whether the
        depletions were caused by a trade or a cancel'''
        tick_value = self.tick_value
        orders_idx_shift = self.Armada_Data.measures.trade_indicator.shift(+1)
        orders_idx_shift.fillna(False, inplace=True)
        bid_diff = self.tob.bid_1_price.diff()/tick_value
        ask_diff = self.tob.ask_1_price.diff()/tick_value
        
        bid_depl_trade = (bid_diff < 0) & (~orders_idx_shift) 
        bid_depl_cancel = (bid_diff < 0) & (orders_idx_shift) 
        
        ask_depl_trade = (ask_diff < 0) & (~orders_idx_shift) 
        ask_depl_cancel = (ask_diff < 0) & (orders_idx_shift) 
        
        bid_fill = bid_diff > 0
        ask_fill = ask_diff > 0
        
        bdt_and_af = bid_depl_trade & ask_fill
        adt_and_bf = ask_depl_trade & bid_fill
        
        bid_depl_trade = bid_depl_trade & (~ bdt_and_af)
        ask_fill = ask_fill & bdt_and_af
        
        ask_depl_trade = ask_depl_trade & (~adt_and_bf)
        bid_fill = bid_fill & (~adt_and_bf)
        
        depl_or_fill = bid_depl_trade | bid_depl_cancel | ask_depl_trade \
            | ask_depl_cancel | bid_fill | ask_fill | \
                bdt_and_af | adt_and_bf
        
        self.depl = pd.DataFrame()
        self.depl['bid_depl_trade'] = bid_depl_trade
        self.depl['bid_depl_cancel'] = bid_depl_cancel
        self.depl['ask_depl_trade'] = ask_depl_trade
        self.depl['ask_depl_cancel'] = ask_depl_cancel
        self.depl['bid_fill'] = bid_fill
        self.depl['ask_fill'] = ask_fill
        self.depl['bdt_and_af'] = bdt_and_af
        self.depl['adt_and_bf'] = adt_and_bf
        self.depl['depl_or_fill'] = depl_or_fill
        
    def __order_indicators(self):
        '''order_indicators(data_frame,tick_value) returns the midprice, the
        microprice, the level 1 and level 2 spreads in ticks, the weighted
        execution price and the associated distance of the average execution
        price on each side to the midprice, the imbalance and a discretization
        of the imbalance given:
        the data_frame
        the tick value'''
        tick_value = self.tick_value
        bid_1_price = self.Armada_Data.df.bid_1_price.copy()
        ask_1_price = self.Armada_Data.df.ask_1_price.copy()
        bid_1_qty = self.Armada_Data.df.bid_1_qty.copy()
        ask_1_qty = self.Armada_Data.df.ask_1_qty.copy()
        bid_2_price = self.Armada_Data.df.bid_2_price.copy()
        ask_2_price = self.Armada_Data.df.ask_2_price.copy()
        bid_2_qty = self.Armada_Data.df.bid_2_qty.copy()
        ask_2_qty = self.Armada_Data.df.ask_2_qty.copy()
        
        
        mid_price = (bid_1_price + ask_1_price) / 2
        
        micro_price = self.__microprice(self.tob.bid_1_qty,\
            self.tob.bid_1_price, self.tob.ask_1_qty,\
            self.tob.ask_1_price)
            
        spread_1_ticks = (ask_1_price-bid_1_price)/tick_value
        spread_2_ticks = (ask_2_price-bid_2_price)/tick_value
        
        bid_to_midprice = (mid_price- bid_1_price)/tick_value
        ask_to_midprice = (mid_price- ask_1_price)/tick_value
        
        bid_12_price = (bid_1_price * bid_1_qty + bid_2_price * bid_2_qty)  \
            /(bid_1_qty + bid_2_qty)
        
        bid_12_to_midprice = (mid_price - bid_12_price)/tick_value
        
        ask_12_price = (ask_1_price * ask_1_qty + ask_2_price * ask_2_qty)  \
            /(ask_1_qty + ask_2_qty)    
        
        ask_12_to_midprice = (mid_price - ask_12_price)/tick_value
        
        spread_ticks = self.__spread_ticks(self.tob.bid_1_price, \
                                           self.tob.ask_1_price\
                                               , tick_value)
        imbalance = self.__imbalance(self.tob.bid_1_qty,\
            self.tob.ask_1_qty)
        imbal_sign = pd.cut(imbalance, [-0.5, -0.2, +0.2, +0.5],\
                            labels=[-1, 0, 1])
        
        self.ord_indic = pd.DataFrame()
        self.ord_indic['mid_price']= mid_price
        self.ord_indic['micro_price']= micro_price
        self.ord_indic['spread_1_ticks']= spread_1_ticks
        self.ord_indic['spread_2_ticks']= spread_2_ticks
        self.ord_indic['bid_to_midprice']= bid_to_midprice
        self.ord_indic['ask_to_midprice']= ask_to_midprice
        self.ord_indic['bid_12_price']= bid_12_price
        self.ord_indic['bid_12_to_midprice']= bid_12_to_midprice
        self.ord_indic['ask_12_price']= ask_12_price
        self.ord_indic['ask_12_to_midprice']= ask_12_to_midprice
        self.ord_indic['spread_ticks']= spread_ticks
        self.ord_indic['imbalance']= imbalance
        self.ord_indic['imbal_sign']= imbal_sign
        
            
    def __microprice(self,qbid, pbid, qask, pask):
        ''' fwp(qbid,pbid,qask,pask) returns the microprice
        (pbid*qask+pask*qbid)/(qbid+qask) given:
        amount on bid qbid, price on bid pbid,
        amount on ask qask, price on ask pask'''
        return (pbid*qask+pask*qbid)/(qbid+qask)
    
    def __spread_ticks(self,pbid, pask, tick_value):
        '''fsp(pbid,pask,tick_value) returns the spread between bid and ask prices
        in ticks given:
        price on bid pbid,
        price on ask pask,
        tick value'''
        return np.round((pask-pbid)/tick_value)
    
    def __imbalance(self,qbid, qask):
        '''imbalance(qbid,qask) returns the imbalance qbid/(qbid+qask)-1/2 given:
        amount on bid qbid, amount on ask qask'''
        return qbid/(qbid+qask)-1/2
    
    
# %% Order statistics

    def get_time_weighted_tob(self):
        '''time_weighted_spread(data_frame) returns the time weighted spread
        in ticks, given the data_frame'''
        df_columns = ['tw_spr1', 'tw_spr2', 'tw_bid1tomid', 'tw_ask1tomid', 'tw_bid1qty',\
            'tw_ask1qty', 'tw_bid12qty', 'tw_ask12qty', 'tw_bid12tomid', 'tw_ask12tomid']
        df_dict = dict(zip(df_columns, len(df_columns)*[0]))
        
        #df_orders = df_imbl[df_imbl['OT']].copy()
        mask1 = self.Armada_Data.measures.trade_indicator.copy()
        mask2 = self.ord_indic.spread_1_ticks > 0
        #data_frame_masked = data_framec[data_framec['Spread_Lvl_1_Ticks'] > 0]
        
        df_masked = self.ord_indic[mask1 & mask2]
        df_masked['delta_t'] = \
            self.Armada_Data.df.DateTime[mask1 & mask2].diff().shift(-1)
        
        def delta_t_weigh(field):
            return ((df_masked[field]*df_masked['delta_t']).sum())/\
                (df_masked['delta_t'].sum())
                
        df_dict['tw_spr1'] = delta_t_weigh('spread_1_ticks')
        df_dict['tw_spr2'] = delta_t_weigh('spread_2_ticks')
        df_dict['tw_bid1tomid'] = delta_t_weigh('bid_to_midprice')
        df_dict['tw_ask1tomid'] = delta_t_weigh('ask_to_midprice')
        df_dict['tw_bid12tomid'] = delta_t_weigh('bid_12_to_midprice')
        df_dict['tw_ask12tomid'] = delta_t_weigh('ask_12_to_midprice')
        
        df_masked = self.Armada_Data.df[mask1 & mask2]
        
        df_dict['tw_bid1qty'] = delta_t_weigh('bid_1_qty')
        df_dict['tw_ask1qty'] = delta_t_weigh('ask_1_qty')
        df_dict['tw_bid12qty'] = delta_t_weigh('bid_1_qty')+delta_t_weigh('bid_2_qty')
        df_dict['tw_ask12qty'] = delta_t_weigh('ask_1_qty')+delta_t_weigh('ask_2_qty')
        data_frame_stats = pd.DataFrame(df_dict)
        return data_frame_stats

    
    def print2file_df_tob(self,pathout, start_time, end_time):
        file_name = pathout+'tob.csv'
        zip_name = pathout + 'tob.zip'
        print('Saving file: ',file_name)
        date = self.Armada_Data.get_processing_date()
        start = (date + start_time)
        end = (date + end_time)
        data_to_print = self.tob.copy()
        data_to_print['DateTime'] = self.Armada_Data.df.DateTime.copy()
        data_to_print = \
            data_to_print.set_index(self.Armada_Data.df.DateTime.copy())
        data_to_print = data_to_print.loc[start:end]
        #date_filter = (data_to_print['DateTime'].to_timedelta() >start & data_to_print['DateTime'].to_timedelta()<end)
        #data_to_print = data_to_print.loc[data_to_print['DateTime'].dt > start & data_to_print['DateTime'].dt < end]
        #self.data_to_print.to_csv(file_name)
        compression_opts = dict(method='zip', archive_name=file_name)
        data_to_print.to_csv(zip_name, index=False, compression=compression_opts) 
    
    def print2file_df_time_weighted_tob(self,pathout):
        file_name = pathout+'df_time_weighted_tob.csv'
        print('Saving file: ',file_name)
        data_to_print = self.get_time_weighted_tob()
        data_to_print.to_csv(file_name)
        


# %% Armada UZ Model Output Class
    
class Armada_UZModel_output:
    df_cont_alt_by_ticks = pd.DataFrame()
    df_uz_stats = pd.DataFrame()
    
    def __init__(self, df_cont_alt_by_ticks=pd.DataFrame(), df_uz_stats= pd.DataFrame()):
        self.df_cont_alt_by_ticks = df_cont_alt_by_ticks
        self.df_uz_stats = df_uz_stats
    
    def append(self, Armada_UZModel_output):
        self.df_cont_alt_by_ticks = self.df_cont_alt_by_ticks.append(\
            Armada_UZModel_output.df_cont_alt_by_ticks, ignore_index = True)
        self.df_uz_stats = self.df_uz_stats.append(\
            Armada_UZModel_output.df_uz_stats, ignore_index = True)
    
    def print2file_df_cont_alt_by_ticks(self,pathout):
        file_name = pathout+'CAticks_timeSeries.csv'
        print('Saving file: ',file_name)
        self.df_cont_alt_by_ticks.to_csv(file_name)
        
    def print2file_df_uz_stats(self,pathout):
        file_name = pathout+'UZstats_timeSeries.csv'
        print('Saving file: ',file_name)
        self.df_uz_stats.to_csv(file_name)
        
    def plot_html_uz_stats(self, path_out):
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,)
        #bid_data = go.Scatter(x=self.data_frame.DateTime,
        #                 y=self.data_frame.bid_1_price)
        #ask_data = go.Scatter(x=self.data_frame.DateTime,
        #                y=self.data_frame.ask_1_price)
        
        fig.add_trace(go.Scatter(x=self.df_uz_stats.Date, y=self.df_uz_stats.rvp,mode='markers',name = 'Realized Volatility - log price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.df_uz_stats.Date, y=self.df_uz_stats.dt_avg,mode='markers',name='Average Duration'), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.df_uz_stats.Date,y=self.df_uz_stats.chgavg,mode='markers',name= 'Average Price Move'), row=3, col=1)
        fig.add_trace(go.Scatter(x=self.df_uz_stats.Date,y=self.df_uz_stats.ndfpr,mode='markers',name='Number of Price Change'), row=4, col=1)

        
        file = path_out+"uz_stats.html"
        print('saving html plot to ', file)
        fig.write_html(file)
        
        
    
# %% Armada UZ Model Class

class ArmadaData_UZModel():
    df = pd.DataFrame()
    df_trades = pd.DataFrame()
    df_cont_alt_by_ticks = pd.DataFrame()
    df_trades_adduz = pd.DataFrame()
    df_uz_stats = pd.DataFrame()
    trading_hours = float()
    n_trades = float()
    volume = float()
    tick_value = float()
    #date = datetime.Date()
    
    def __init__(self, Armada_Data, tick_value \
                 , start_time = pd.to_timedelta('00:00:00')\
                 , end_time = pd.to_timedelta('23:59:59')):
        self.df = Armada_Data.df
        self.file_out = Armada_Data.file_name_long
        self.processing_date = Armada_Data.get_processing_date()
        #self.df = data_frame
        self.tick_value = tick_value
        #self.__column_datetime()
        self.__column_ot()
        self.select_times(start_time, end_time)
        self.trading_hours = float((end_time-start_time).seconds/3600)
        print('Running Uncertainty Zones statistics')
        print('Input Tick Value Parameter: ', tick_value)
        self.calculate()
        print('UZ Model Object Construction Sucessfull')
    
    
    def print2file_df_cont_alt_by_ticks(self,pathout):
        file_name = pathout+self.file_out+'_CAticks.csv'
        print('Saving file: ',file_name)
        self.df_cont_alt_by_ticks.to_csv(file_name)
    
    def print2file_df_uz_stats(self,pathout):
        file_name = pathout+self.file_out+'_UZstats.csv'
        print('Saving file: ',file_name)
        self.df_uz_stats.to_csv(file_name)
        
    
    def calculate(self):
        self.__calc_trades()
        self.__calc_uz_coal_byk()
        self.__calc_uz_stats()

    def get_trades(self):
        #write here to check if field empty then calc_trades, if not return
        self.__calc_trades()
        return self.df_trades
    
    def ohlc(self, pathout):
        ohlc = pd.DataFrame()
        self.df_trades = self.df_trades.set_index(['DateTime'])
        ohlc = self.df_trades.trade_price.resample('1min').ohlc()
        
        fig = go.Figure(data=[go.Candlestick(
                x=ohlc.index,
                open=ohlc.open, high=ohlc.high,
                low=ohlc.low, close=ohlc.close,
                increasing_line_color= 'cyan', decreasing_line_color= 'gray'
        )])
        file = pathout+"ohlc.html"
        print('saving html plot to ', file)
        fig.write_html(file)
        return ohlc

    def get_uz_coal_byk(self):
        # if empty then
        self.__calc_uz_coal_byk()
        return self.df_cont_alt_by_ticks

    def get_uz_stats(self):
        self.__calc_uz_stats()
        return self.df_uz_stats

    def __add_date_to_df_cont_alt_by_ticks(self):
        self.df_cont_alt_by_ticks['Date']=self.processing_date
        
    def __add_date_to_df_uz_stats(self):
        self.df_uz_stats['Date']=self.processing_date
        
    def __calc_trades(self):
        self.df_trades = self.__get_trades_remove_blocktrades()
        self.volume = float(self.df_trades.trade_qty.sum())
        self.__collapse_time()
        self.n_trades = float(len(self.df_trades))
        self.__collapse_price()
        self.__trades_min_increment()
        print('Mininum Non Zero Trade Price Increment: '\
              , self.__trades_min_increment)
    
    def __calc_uz_coal_byk(self):
        self.__adduz()
        self.__uz_coal_byk()

    def __calc_uz_stats(self):
        self.__uz_stats()
        self.__uz_coal_byk()
        self.df_uz_stats['M'] = self.n_trades
        self.df_uz_stats['S1']= self.get_percentspread_at1tick_priortrade()
        self.df_uz_stats['Volume'] = self.volume
        
    def __trades_min_increment(self):
        diff = abs(self.df_trades.trade_price-\
                   self.df_trade.trade_price.shift(1))
        diff_no0 = diff[diff!=0]
        self.__trades_min_increment = np.nanmin(diff_no0)
        
    def __get_trades_remove_blocktrades(self):
        '''function returns a trades matrix with no trade_price above or 
        below the bid ask'''
        df_fill = self.df.copy()
        ''' fill dataframe with previous value trades have nan bid and ask
        and will get previous bid price or ask price'''
        df_fill = df_fill.ffill(axis=0)
        df_trades = df_fill[~df_fill['OT']].copy()
        'remove block trades - they are above the ask and below the bid prices'
        df_bidprice = df_trades.bid_1_price
        df_askprice = df_trades.ask_1_price
        df_trades_price = df_trades.trade_price
        df_trades_orig = self.df[~self.df.OT].copy()
        df_trades_orig = df_trades_orig.drop(df_trades[(df_trades_price <  \
                        df_bidprice)|(df_trades_price > df_askprice)].index)
        return df_trades_orig
        
# %% Functions applied to the initial data (selection, orders vs trades)
    def get_percentspread_at1tick_priortrade(self):
        tick_value = self.tick_value
        sub_df = self.df.copy()[['bid_1_price', 'ask_1_price', 'OT']]
        sub_df['nextOT']= ~(sub_df['OT'].shift(-1).fillna(True))
        sub_df['OandT']= sub_df['nextOT']&sub_df['OT']
        sub_df['sprOandT']=np.round((sub_df['ask_1_price'] - \
                                     sub_df['bid_1_price'])/tick_value)
        spreads_OT = sub_df[sub_df['OandT']]['sprOandT']
        spreads_OT_1 = (spreads_OT.value_counts()).loc[1]/len(spreads_OT)
        return spreads_OT_1

    def __column_ot(self):
        '''column_ot(data_frame) creates the column 'OT'
        where 'OT' is True if the row corresponds to a change in the top of
        the book from either a limit order or a cancellation and 'OT' is
        False if the row is a trade.
        Inputs: Data Frame from Armada
        Outputs: Copy of DataFrame with 'OT' column added'''
        data_frame = self.df
        data_framec = data_frame.copy()
        data_framec['OT'] = data_framec.trade_price.isnull().copy()
        self.df = data_framec

    def select_times(self, start_time, end_time):
        '''select_times(data_frame, start_time, end_time) selects entries
        between start_time and end_time (inclusive for both cases) and
        then drops the order book events before the first trades
        (tipically the setup for the opening auction)
        Inputs: Data Frame from Armada preprocessed by column_datetime and
        column_ot, start and end times
        Outputs: Copy of DataFrame with events between the selected times
        and dropping orders before the first trade, and with redundant
        date, time and index columns deleted'''
        #data_frame = self.df
        data_framec = self.df.copy()
        date = pd.to_datetime(data_framec['DateTime'].values[0]).date()
        # Find first trade after start_time
        first_time = pd.to_datetime(date) + start_time
        mask_first = data_framec['DateTime'] >= first_time
        # Find last trade before end_time
        last_time = pd.to_datetime(date) + end_time
        mask_last = data_framec['DateTime'] <= last_time
        # Combine time masks
        mask_time = np.logical_and(mask_first, mask_last)
        data_framec = data_framec[mask_time]
        # Find first trade
        first_trade = data_framec[~data_framec['OT']].index[0]
        mask_first_trade = data_framec.index >= first_trade
        data_framec = data_framec[mask_first_trade]
        # Drop any leftover row from auction
        del_index = data_framec[(data_framec.ask_1_price <=\
                                 data_framec.bid_1_price)].index
        data_framec.drop(del_index, inplace=True)
        #data_framec.drop(columns=['Date', 'Time', 'Time.1',\
        #    'Date.1', 'Index.1'], inplace=True)
        self.df = data_framec
    
    # %% Collapsing functions
    
    def __collapse_time(self):
        '''collapse_time(data_frame) takes a Data Frame of trades at various
        times and prices that might have different amounts traded at the same
        time(s) and price(s) and collapses these trades into these same times
        and prices, adding the 'Trade Qty' field for each collapsed group.
        In short: Group trades by time and price, sum(Trade Qty)
        Inputs: Data Frame from Armada preprocessed by column_datetime and
        column_ot
        Outputs: Smaller (or equal) DataFrame with unique time, price pairs
        and the values of 'Trade Qty' grouped with sum()'''
        data_frame = self.df_trades
        data_framecol = data_frame.columns.tolist()
        data_framec = data_frame.copy()
        data_framecl = data_framec.columns.tolist()
        data_framecl.remove('trade_qty')
        data_framec['DateTimem'] = data_framec['DateTime'] ==\
            data_frame['DateTime'].shift()
        data_framec['Trade Pricem'] = data_framec.trade_price ==\
            data_frame.trade_price.shift()
        data_framec['m'] = ~(data_framec['DateTimem']&data_framec['Trade Pricem'])
        data_framec['nm'] = np.nan
        data_framec['nm'] = np.where(data_framec['m'],\
                   data_framec['m'].cumsum()-1, data_framec['nm'])
        data_framec['nm'].fillna(method='ffill', inplace=True)
        df_grouped = data_framec[['trade_qty', 'nm']].groupby('nm').sum()
        df_first_traded = data_framec[data_framecl][data_framec['m']]
        df_first_traded.index = df_grouped.index
        df_grouped_at_first = pd.concat([df_first_traded, df_grouped], axis=1)
        df_grouped_at_first = df_grouped_at_first[data_framecol]
        df_grouped_at_first.index.names = ['gindex']
        self.df_trades = df_grouped_at_first

    def __collapse_price(self):
        '''collapse_time(data_frame) takes a Data Frame of trades at various
        prices that might have different amounts traded at the same price(s)
        and collapses these trades into these same prices, adding the 'Trade Qty'
        field for each collapsed group.
        In short: Group trades by price, sum(Trade Qty)
        Inputs: Data Frame from Armada preprocessed by column_datetime and
        column_ot
        Outputs: Smaller (or equal) DataFrame with different consecutive prices
        and the values of 'Trade Qty' grouped with sum()'''
        data_frame = self.df_trades
        data_framecol = data_frame.columns.tolist()
        data_framec = data_frame.copy()
        data_framecl = data_framec.columns.tolist()
        data_framecl.remove('trade_qty')
        data_framec['Trade Pricem'] = data_framec.trade_price ==\
            data_frame.trade_price.shift()
        data_framec['m'] = ~(data_framec['Trade Pricem'])
        data_framec['nm'] = np.nan
        data_framec['nm'] = np.where(data_framec['m'],\
                   data_framec['m'].cumsum()-1, data_framec['nm'])
        data_framec['nm'].fillna(method='ffill', inplace=True)
        df_grouped = data_framec[['trade_qty', 'nm']].groupby('nm').sum()
        df_first_traded = data_framec[data_framecl][data_framec['m']]
        df_first_traded.index = df_grouped.index
        df_grouped_at_first = pd.concat([df_first_traded, df_grouped], axis=1)
        df_grouped_at_first = df_grouped_at_first[data_framecol]
        df_grouped_at_first.index.names = ['gindex']
        self.df_trade = df_grouped_at_first

    # %% Trades UZ fields
    
    def __adduz(self):
        '''adduz(data_frame,alpha) returns a data frame with
        the fields necessary for the UZ stats
        Inputs: Data Frame of trades collapsed by collapse_time and by
        collapse_price
        Outputs: Collapsed Data Frame with additional fields for the UZ model'''
        alpha = self.tick_value
        data_frame = self.df_trades
        data_framec = data_frame.copy()
        data_framec['dPj'] = data_framec.trade_price.diff()
        data_framec['sign'] = np.sign(data_framec['dPj'])
        data_framec['Li'] = np.round(np.abs((data_framec['dPj'])/alpha), 4)
        data_framec['dtTj'] = data_framec['DateTime'].diff()
        data_framec['Co'] = data_framec['sign'].diff() == 0
        data_framec['Al'] = data_framec['sign'].diff().abs() == 2
        data_framec['Co'] = data_framec['Co']*1.
        data_framec['Al'] = data_framec['Al']*1.
        self.df_trades_adduz = data_framec

# %% Eta components
    
    def __uz_coal_byk(self):
        '''uz_coal_byk(data_frame_trades) returns the uncertainty zones
        data frame for the different values of k (price changes in ticks)'''
        data_frame_trades = self.df_trades_adduz
        k_array = np.sort(data_frame_trades['Li'].dropna().unique())
        data_frame_k = pd.DataFrame(k_array, columns=['Li'])
        data_frame_k.set_index('Li', drop=False, inplace=True)
        coal_group_co = data_frame_trades['Co'].groupby(data_frame_trades['Li'])
        coal_group_al = data_frame_trades['Al'].groupby(data_frame_trades['Li'])
        data_frame_k['lamb'] = (coal_group_al.count()/\
                    np.sum(coal_group_al.count()))
        data_frame_k['Co'] = coal_group_co.sum()
        data_frame_k['Al'] = coal_group_al.sum()
        data_frame_k['u'] = (data_frame_k['Li']*(\
                    (data_frame_k['Co']/data_frame_k['Al']).fillna(0)-1)+1)/2
        data_frame_k['eta'] = data_frame_k['lamb']*data_frame_k['u']
        data_frame_k.reset_index(drop=True, inplace=True)
        self.df_cont_alt_by_ticks = data_frame_k
        self.__add_date_to_df_cont_alt_by_ticks()

# %% UZ stats

    def __rlzvollog(self,prices):
        '''rlzvollog(prices) calculates the realized volatility of a time series
        of prices using logreturns
        Inputs: Time Series of prices
        Outputs: scalar with the volatility for the period'''
        pxs = np.log(prices/prices.shift(1))
        return np.sqrt(np.sum(pxs*pxs))
    
    def __uz_duration(self, spot, tick_value, eta, vol):
        '''dur(spot, alpha, eta, vol) calculates the estimated durations
        of an asset given microstructure parameters
        Inputs:
        Spot price
        alpha = tick value
        eta
        vol = volatility of the efficient prices scaled for the desired time unit
        Outputs: scalar with the estimated duration according to the time scale
        of the volatility'''
        return 2*eta*(tick_value/(spot*vol))**2
    
    def __uz_stats(self):
        '''uz_stats(data_frame, file_out, tick_value) returns the uncertainty
        zones statistics)'''
        tick_value = self.tick_value
        trading_hours = self.trading_hours
        data_frame_trades = self.df_trades_adduz
        df_columns = ['eta1', 'rvp', 'rvx', 'rvxe', 'dt_cont', 'dt_alt', \
                      'dt_avg', 'duration', 'chgavg', 'ndfpr', 'spot_avg']
        df_dict = dict(zip(df_columns, len(df_columns)*[0]))
        uz_df = self.df_cont_alt_by_ticks#uz_coal_byk(data_frame_trades)
        uz_df.set_index('Li', drop=False, inplace=True)
        #df_dict['eta1'] = uz_df['eta'].loc[1]
        df_dict['eta1'] = uz_df['eta'].values[0]
        effic_prices = self.__effprpath(tick_value, df_dict['eta1'],\
                                 data_frame_trades['trade_price'])
        df_dict['rvp'] = self.__rlzvollog(data_frame_trades['trade_price'])
        df_dict['rvx'] = self.__rlzvollog(effic_prices)
        df_dict['rvxe'] = np.sqrt(2*df_dict['eta1'])*df_dict['rvp']
        df_dict['dt_cont'] = np.mean(data_frame_trades[data_frame_trades['Co']\
            == 1]['dtTj']).total_seconds()
        df_dict['dt_alt'] = np.mean(data_frame_trades[data_frame_trades['Al']\
            == 1]['dtTj']).total_seconds()
        df_dict['dt_avg'] = np.mean(data_frame_trades['dtTj']).total_seconds()
        df_dict['spot_avg'] = data_frame_trades['trade_price'].mean()
        df_dict['duration'] = self.__uz_duration(df_dict['spot_avg'], tick_value,\
              df_dict['eta1'], df_dict['rvxe']/np.sqrt(trading_hours*3600))
        df_dict['chgavg'] = np.mean(np.abs(data_frame_trades['dPj']))
        df_dict['ndfpr'] = np.float(len(data_frame_trades['dPj'].copy().dropna()))
        data_frame_stats = pd.DataFrame(df_dict, index=[self.file_out])
        self.df_uz_stats = data_frame_stats
        self.__add_date_to_df_uz_stats()
    
    def __effprpath(self,alpha,eta,path):
        '''returns the efficient price P(t) path given the traded price P(t) 
        path, the tick size alpha and the parameter eta'''
        xpath=path.copy()
        for k in range(1,len(xpath)):
            xpath[k]=path[k]-alpha*(0.5-eta)*np.sign(path[k]-path[k-1])
        return xpath
    
    def plot_html_cont_alt(self,path_out):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,)
        #bid_data = go.Scatter(x=self.data_frame.DateTime,
        #                 y=self.data_frame.bid_1_price)
        #ask_data = go.Scatter(x=self.data_frame.DateTime,
        #                y=self.data_frame.ask_1_price)
        
        fig.add_trace(go.Scatter(x=self.data_frame_trades.DateTime, y=self.data_frame_trades.Al), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.data_frame_trades.DateTime, y=self.data_frame_trades.Co), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.__df.DateTime,y=self.__get_BAspread(self.__df)), row=2, col=1)
        fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
        
        file = path_out+self.file_name(True)+"_price.html"
        print('saving html plot to ', file)
        fig.write_html(file)
    
    def get_Armada_UZModel_output(self):
        return Armada_UZModel_output(self.df_cont_alt_by_ticks, self.df_uz_stats)
        
