
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
        self.__processing_date = self.get_processing_date()
        self.__file_path = file_path
        self.__df = pd.DataFrame()
        #self.__measures = pd.DataFrame()
        
        print('Reading file '+ self.file_entire_path)
        self.__read_ArmadaData()
        print('Re-formatting data')
        self.__column_datetime()
        self.__rename_columns()
        print('Remove data outside exchange trading hours')
        self.__filter_exchange_data_by_time()  
        self.__filter_exchange_data_prior_first_trade()
        print('Armada Data Object Construction Successfull')
    
    @property
    def processing_date(self):
        return self.get_processing_date()
    
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

    def select_times(self,start_time = pd.to_timedelta('00:30:00'), \
                     end_time = pd.to_timedelta('23:59:59')):
        self.__filter_exchange_data_by_time([start_time,end_time])
        self.__filter_exchange_data_prior_first_trade()
        return self
    
    # %% Public Functions
    
    def get_processing_date(self):
        if self.__exchange == 'CME':
            return pd.to_datetime(self.file_name[0:8],format='%Y%m%d')
        if self.__exchange == 'BMF':
            return pd.to_datetime(self.file_name[6:-4],format='%Y%m%d')
        
    def get_exchange_starting_time(self):
        if self.__exchange == 'CME':
            return self.processing_date\
                                + pd.to_timedelta('00:00:00')
        if self.__exchange == 'BMF':
            return self.processing_date\
                                + pd.to_timedelta('09:00:00')
                                
    def get_exchange_end_time(self):
        if self.__exchange == 'CME':
            return self.processing_date\
                                + pd.to_timedelta('16:00:00')
        if self.__exchange == 'BMF':
            return self.processing_date\
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
    def __filter_exchange_data_prior_first_trade(self):
        first_trade = self.df[self.get_trade_indicator()].index[0]
        df_tmp = self.__df.copy()
        #df_tmp = df_tmp.set_index(['DateTime'])
        df_tmp = df_tmp.loc[first_trade:]
        #df_tmp = df_tmp.reset_index()
        self.__df = df_tmp  
    
    def __filter_exchange_data_by_time(self, arg = None):
        if arg == None:
            start_time = self.get_exchange_starting_time()
            end_time = self.get_exchange_end_time()
        else:
            start_time = self.processing_date + arg[0]
            end_time = self.processing_date + arg[1]
        
        # truncating df
        df_tmp = self.__df.copy()
        df_tmp = df_tmp.set_index(['DateTime'])
        df_tmp = df_tmp.loc[start_time:end_time]
        df_tmp = df_tmp.reset_index()
        self.__df = df_tmp        

# %% Measures Functions 
        
    def get_order_indicator(self):
        return self.__df.trade_price.isnull().copy()
    
    def get_trade_indicator(self):
        return ~(self.get_order_indicator())
    
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
        #data = data.assign(ba_spread=self.measures.ba_spread.values)
        #data = data.assign(dt=self.measures.dt .values)
        #data_masked = data[data.ba_spread > 0]
        #time_weighted_spread = pd.DataFrame(\
        #    data_masked.groupby(pd.Grouper(freq=freq)).apply(self.__get_time_weighted_spread))   
        #time_weighted_spread.columns = ['time_weighted_spread']
        #time_weighted_spread = time_weighted_spread.loc[start_xaxis:end_xaxis]
        
        
        # plotting
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
        row_width=[0.2, 0.4],vertical_spacing=0.1,
        subplot_titles=('Open High Low Close Candlestick', 'Volume'))
        
        fig.add_trace(go.Candlestick(name='OHLC',
                x=ohlc.index,
                open=ohlc.open, high=ohlc.high,
                low=ohlc.low, close=ohlc.close,
                increasing_line_color= 'green', decreasing_line_color= 'red'),
                row=1, col=1)
        
    
        fig.add_trace(go.Bar(name='Volume', x=volume.index, \
                y=volume.trade_qty), row=2, col=1)
            
        #fig.add_trace(go.Bar(x=time_weighted_spread.index, \
        #        y=time_weighted_spread.time_weighted_spread , 
        #        name = 'Time Weighted Spread', marker_color='black'), 
        #        row=3,col=1)
        
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
        self.__Armada_Data = Armada_Data
        self.__exchange = Armada_Data.exchange
        self.__file_name = Armada_Data.file_name
        self.__file_name_long = Armada_Data.file_name_long
        self.__file_entire_path = Armada_Data.file_entire_path
        self.__processing_date = Armada_Data.processing_date
        self.__tick_value = tick_value
        self.__tob = pd.DataFrame()
        ######to delete once algo completed
        self.__tob['traded_price'] = self.__Armada_Data.df.trade_price.copy() 
        self.__tob['traded_qty'] = self.__Armada_Data.df.trade_qty.copy() 
        ######
        print('Filling intermediate values in order book')
        self.__fill_tob_price_last()
        self.__fill_price_traded()
        #self.__fill_aggression()
        self.num_consecutive_trade = self.__find_num_consecutive_trade()
        print(self.__Armada_Data.file_name+' max_runs = '+str(self.num_consecutive_trade))
        self.__fill_tob()
        stop = timeit.default_timer()
        print('Time spent on top-of-book filling: ', round(stop - start), ' seconds')
        self.__depletion()
        print('Depletion completed')
        print('Calculating order indicators')
        self.__order_indicators()
        print('Order indicators completed')
        self.get_data_for_tob_intensity()

    @property
    def file_name(self):
        return self.__file_name
    @property
    def file_name_long(self):
        return self.__file_name_long
    @property
    def file_entire_path(self):
        return self.__file_entire_path

    @property
    def tick_value(self):
        return self.__tick_value
    
    @property
    def tob(self):
        return self.__tob
        
    @property
    def exchange(self):
        return self.__exchange
    
    @property
    def processing_date(self):
        return self.__processing_date
    
    #def get_processing_date(self):
    #    return Armada_Data.get_processing_date
    
    def get_tob_intensity_output(self):
        return TOB_Intensity_Output(self.processing_date, self.results_bid, self.results_ask)
    
    def get_event_count(self):
        self.event = pd.DataFrame()
        # get data we need
        df = self.__tob[['bid_1_price', 'bid_1_qty', 'ask_1_price', 'ask_1_qty']].copy()
        df['trade_price'] = self.__Armada_Data.df.trade_price.copy()
        df['trade_qty'] = self.__Armada_Data.df.trade_qty.copy()
        df['DateTime'] = self.__Armada_Data.df.DateTime.copy()
        df['order_idx'] = self.__Armada_Data.get_order_indicator()
        df[['bid_traded','ask_traded']] = \
            self.__tob[['bid_price_traded', 'ask_price_traded']].copy()
        
        # run unique - collapse time and remove second level bid / ask
        df_unique = df.drop_duplicates(subset = ['bid_1_price', 'bid_1_qty', 'ask_1_price', 'ask_1_qty', 'trade_price', 'trade_qty'])
        
        # get bid and ask before attributes
        bid_qty_before = df_unique.bid_1_qty.shift(+1).copy()
        ask_qty_before = df_unique.ask_1_qty.shift(+1).copy()
        
        # get delta_t attribute
        delta_t = df_unique.DateTime.diff().shift(+1)
        
        # get consumption / insertion boolean attribute
        #istrade = ~(df_unique.traded_price.isnull().copy())
        
        orders_idx_shift = df_unique.order_idx.shift(+1)
        orders_idx_shift.fillna(False, inplace=True)
        
        # price diff  
        bid_price_diff = df_unique.bid_1_price.diff()/self.tick_value
        ask_price_diff = df_unique.ask_1_price.diff()/self.tick_value
        
        #spread 
        spread_diff =  (df_unique.ask_1_price - df_unique.bid_1_price).diff()
        
        # combinations
        bool_b_neg = (bid_price_diff < 0)
        bool_b_pos = (bid_price_diff > 0)
        bool_b_cst = bool_b_neg & bool_b_pos
        
        bool_s_neg = spread_diff < 0
        bool_s_pos = spread_diff > 0
        bool_s_pos = bool_s_neg & bool_s_pos
        
        bool_a_neg = (ask_price_diff < 0)
        bool_a_pos = (ask_price_diff > 0)
        bool_a_cst = bool_a_neg & bool_a_pos
        
        e_b1_s1_a1 = bool_b_neg & bool_s_neg & bool_a_neg
        e_b1_s1_a2 = bool_b_neg & bool_s_neg & bool_a_pos
        e_b1_s1_a3 = bool_b_neg & bool_s_neg & bool_a_cst
        
        e_b1_s2_a1 = bool_b_neg & bool_s_pos & bool_a_neg
        e_b1_s2_a2 = bool_b_neg & bool_s_pos & bool_a_pos
        e_b1_s2_a3 = bool_b_neg & bool_s_pos & bool_a_cst
        
        
    def get_data_for_tob_intensity(self):
        self.event = pd.DataFrame()
        # get data we need
        df = self.__tob[['bid_1_price', 'bid_1_qty', 'ask_1_price', 'ask_1_qty']].copy()
        df['trade_price'] = self.__Armada_Data.df.trade_price.copy()
        df['trade_qty'] = self.__Armada_Data.df.trade_qty.copy()
        df['DateTime'] = self.__Armada_Data.df.DateTime.copy()
        df['order_idx'] = self.__Armada_Data.get_order_indicator()
        df[['bid_traded','ask_traded']] = \
            self.__tob[['bid_price_traded', 'ask_price_traded']].copy()
        
        # run unique - collapse time and remove second level bid / ask
        df_unique = df.drop_duplicates(subset = ['bid_1_price', 'bid_1_qty', 'ask_1_price', 'ask_1_qty', 'trade_price', 'trade_qty'])
        
        # get bid and ask before attributes
        bid_qty_before = df_unique.bid_1_qty.shift(+1).copy()
        ask_qty_before = df_unique.ask_1_qty.shift(+1).copy()
        
        # get delta_t attribute
        delta_t = df_unique.DateTime.diff().shift(+1)
        
        # get consumption / insertion boolean attribute
        #istrade = ~(df_unique.traded_price.isnull().copy())
        
        orders_idx_shift = df_unique.order_idx.shift(+1)
        orders_idx_shift.fillna(False, inplace=True)
        
        # bid  
        bid_price_diff = df_unique.bid_1_price.diff()/self.tick_value
        ask_price_diff = df_unique.ask_1_price.diff()/self.tick_value
        bid_depl_trade = (bid_price_diff < 0) & (~orders_idx_shift) 
        bid_depl_cancel = (bid_price_diff < 0) & (orders_idx_shift) 
        
        bid_qty_diff = df_unique.bid_1_qty.diff().copy()
        ask_qty_diff = df_unique.ask_1_qty.diff().copy()
        # no trade, decrease qty, price the same 
        bid_qty_less = (bid_qty_diff <0) & (bid_price_diff ==0) & df_unique.order_idx
        bid_qty_add = (bid_qty_diff >0) & (bid_price_diff ==0) & df_unique.order_idx
        
        bid_inspread_add = (bid_price_diff > 0) & df_unique.order_idx
        
        bid_event = bid_depl_trade | bid_depl_cancel \
                       | bid_qty_less | bid_qty_add | bid_inspread_add
        # Ask 
        
        ask_depl_trade = (ask_price_diff < 0) & (~orders_idx_shift) 
        ask_depl_cancel = (ask_price_diff < 0) & (orders_idx_shift) 
        
        # no trade, decrease qty, price the same 
        ask_qty_less = (ask_qty_diff <0) & (ask_price_diff ==0) & df_unique.order_idx
        ask_qty_add = (ask_qty_diff >0) & (ask_price_diff ==0) & df_unique.order_idx
        
        ask_inspread_add = (ask_price_diff < 0) & df_unique.order_idx
        ask_outspread_less = (ask_price_diff > 0) & df_unique.order_idx
        
        
        ask_event = ask_depl_trade | ask_depl_cancel | \
                       ask_qty_less | ask_qty_add | ask_inspread_add | ask_outspread_less
                       
        
        ## finding insertion (add) and consumption (less)
        consumption_bid = bid_depl_trade | bid_depl_cancel | bid_qty_less 
        consumption_ask = ask_depl_trade | ask_depl_cancel | ask_qty_less | ask_outspread_less
       
        #insertion = bid_qty_add | bid_inspread_add \
        #        | ask_qty_add | ask_inspread_add
        self.event['bid_consumption'] = bid_event & consumption_bid
        self.event['bid_insertion'] = bid_event & (~consumption_bid)
        self.event['bid_size'] = df_unique.bid_1_qty
        self.event['bid_delta_t'] = delta_t
        
        
        self.event['ask_consumption'] = consumption_ask
        self.event['ask_insertion'] = ask_event & (~consumption_ask)
        self.event['ask_size'] = df_unique.ask_1_qty
        self.event['ask_delta_t'] = delta_t
    
        self.event['spread'] = df_unique.ask_1_price - df_unique.bid_1_price 
        self.event = self.event.set_index(df_unique.DateTime.copy())
        
    
        #self.event['ask_event'] = ask_event
        #self.event['consumption'] = consumption
        #self.event['insertion'] = insertion
        #self.event['bid_qty_before'] = bid_qty_before
        #self.event['ask_qty_before'] = ask_qty_before 
        #self.event['delta_t'] = delta_t
        event_bid = pd.DataFrame()
        event_bid['bid_event']= bid_event
        event_bid['consumption'] = consumption_bid
        event_bid['size_before'] = bid_qty_before
        event_bid['delta_t'] = delta_t
        
        event_bid = event_bid[event_bid.bid_event==True]
        
        event_ask = pd.DataFrame()
        
        event_ask['ask_event']= ask_event
        event_ask['consumption'] = consumption_ask
        event_ask['size_before'] = ask_qty_before
        event_ask['delta_t'] = delta_t
        
        event_ask = event_ask[event_ask.ask_event==True]
        
        #data = result.groupby(['ISIN', 'BuyMemberID','Newspread', 'Bid_BBO_Qty_AES']).agg({'Limit':'sum', 'Market':'sum','Cancel':'sum', 'DiffTime':'sum','count':'sum'})    
        self.results_bid = pd.DataFrame()
        self.results_ask = pd.DataFrame()
        
        self.results_bid = event_bid.groupby(['consumption','size_before']).agg({'delta_t':'sum', 'bid_event':'sum'}) 
        self.results_ask = event_ask.groupby(['consumption','size_before']).agg({'delta_t':'sum', 'ask_event':'sum'}) 
        self.results_bid.delta_t  = self.results_bid.delta_t.dt.total_seconds()
        self.results_ask.delta_t  = self.results_ask.delta_t.dt.total_seconds()
        
        self.results_bid['intensity'] = self.results_bid.bid_event\
            / self.results_bid.delta_t
        self.results_ask['intensity'] = self.results_ask.ask_event \
            / self.results_ask.delta_t
        
        self.results_bid = self.results_bid.reset_index()
        self.results_ask = self.results_ask.reset_index()
        
        
        # if all true, we have all event cases
        all_event = (~df_unique.order_idx) | ask_event | bid_event
        print(all_event.sum() / len(all_event))
        ''' output for debudgging 
        df_unique['bid_depl_trade'] =  bid_depl_trade
        df_unique['bid_depl_cancel'] = bid_depl_cancel
        df_unique['bid_qty_less'] = bid_qty_less
        df_unique['bid_qty_add'] = bid_qty_add
        df_unique['bid_inspread_add'] = bid_inspread_add
        df_unique['ask_depl_trade'] = ask_depl_trade
        df_unique['ask_depl_cancel'] =ask_depl_cancel
        df_unique['ask_qty_less'] = ask_qty_less
        df_unique['ask_qty_add'] = ask_qty_add
        df_unique['ask_inspread_add'] = ask_inspread_add
        df_unique['ask_outspread_less'] = ask_outspread_less
        
        df_unique['all_event'] = all_event

        test = df_unique[0:100] # for debug'''
        #test_1 = self.event[0:100]
        # bid consu,ption
        # remove level 2 order book lines due to 
        # bid qty before
        # detla_t 
        # count the number of event 
        # column event (qty added, price change, )

    def get_rolling_event(self, start_time, end_time):
        date = self.processing_date
        start_lag =  date + start_time - pd.to_timedelta('00:00:01')
        start = date + start_time
        end = date + end_time
        
        df = pd.DataFrame(self.event)
        df = df.loc[start_lag:end]
        df_roll = df.rolling('1s').sum()
        
        df_roll = df_roll.loc[start:end]
        df_roll['bid_size'] = df.bid_size.loc[start:end]
        df_roll['ask_size'] = df.ask_size.loc[start:end]
        df_roll['spread'] = df.spread.loc[start:end]
        return df_roll 
    
    def __fill_tob(self):
        # initialize to original not filled tob
        self.__tob['bid_1_price'] = self.__Armada_Data.df.bid_1_price.copy()
        self.__tob['bid_1_qty'] = self.__Armada_Data.df.bid_1_qty.copy()
        self.__tob['ask_1_price'] = self.__Armada_Data.df.ask_1_price.copy()
        self.__tob['ask_1_qty'] = self.__Armada_Data.df.ask_1_qty.copy()
        
        self.__tob['bool_trade'] = self.__Armada_Data.get_trade_indicator()
        
        # if trade happened        
        for i in reversed(range(self.num_consecutive_trade + 1)):
            
            self.__tob['bool_idx'] = self.__tob['cumsum'].astype('int64') == i
            shift_bid_price = self.__tob.bid_1_price.shift(-1)
            shift_bid_qty = self.__tob.bid_1_qty.shift(-1)
            shift_ask_price = self.__tob.ask_1_price.shift(-1)
            shift_ask_qty = self.__tob.ask_1_qty.shift(-1)
            
            #If ask or bid price t+1 = trade price t 
            self.__tob['bool_bid_D'] = self.__Armada_Data.df.trade_price == shift_bid_price
            self.__tob['bool_ask_D'] = self.__Armada_Data.df.trade_price == shift_ask_price      
            
            self.__tob.bid_1_qty.loc[    \
                self.__tob.bool_idx &    \
                self.__tob.bool_trade &  \
                self.__tob.bool_bid_D] = \
                pd.Series(np.sum([self.__Armada_Data.df.trade_qty , shift_bid_qty], axis=0))
            
            self.__tob.bid_1_price.loc[    \
                self.__tob.bool_idx &    \
                self.__tob.bool_trade &  \
                self.__tob.bool_bid_D] = self.__Armada_Data.df.trade_price
                
            self.__tob.ask_1_qty.loc[    \
                self.__tob.bool_idx &    \
                self.__tob.bool_trade &  \
                self.__tob.bool_ask_D] = \
                pd.Series(np.sum([self.__Armada_Data.df.trade_qty , shift_ask_qty], axis=0))
                
            self.__tob.ask_1_price.loc[    \
                self.__tob.bool_idx &    \
                self.__tob.bool_trade &  \
                self.__tob.bool_ask_D] = self.__Armada_Data.df.trade_price

            #If ask or bid price t+1 != trade price t 
            self.__tob['bool_bid_C'] = self.__Armada_Data.df.trade_price != shift_bid_price
            self.__tob['bool_ask_C'] = self.__Armada_Data.df.trade_price != shift_ask_price      
            
            self.__tob.ask_1_qty.loc[    \
                self.__tob.bool_idx &    \
                self.__tob.bool_trade &  \
                self.__tob.bool_ask_C] = self.__Armada_Data.df.trade_qty
                
            self.__tob.ask_1_price.loc[    \
                self.__tob.bool_idx &    \
                self.__tob.bool_trade &  \
                self.__tob.bool_ask_C] = self.__Armada_Data.df.trade_price
                
            self.__tob.bid_1_qty.loc[    \
                self.__tob.bool_idx &    \
                self.__tob.bool_trade &  \
                self.__tob.bool_bid_C] = self.__Armada_Data.df.trade_qty
                
            self.__tob.bid_1_price.loc[    \
                self.__tob.bool_idx &    \
                self.__tob.bool_trade &  \
                self.__tob.bool_bid_C] = self.__Armada_Data.df.trade_price
            
            # if side not traded: 
            #{Qty_Other_Side[t] , Price_Other_Side[t]} = 
            #{Qty_Other_Side[t+1] , Price_Other_Side[t+1]} # side not traded

            self.__tob.ask_1_price.loc[    \
                self.__tob.bid_price_traded & \
                self.__tob.bool_idx] = np.nan #shift_ask_price
                
            self.__tob.ask_1_qty.loc[
                self.__tob.bid_price_traded & 
                self.__tob.bool_idx] = np.nan #shift_ask_qty
            
            self.__tob.bid_1_price.loc[    \
                self.__tob.ask_price_traded & \
                self.__tob.bool_idx] = np.nan #shift_bid_price
                
            self.__tob.bid_1_qty.loc[
                self.__tob.ask_price_traded & 
                self.__tob.bool_idx] = np.nan #shift_bid_qty
        
        # remove if debug
        self.tob.drop(['bool_bid_C', 'bool_ask_C','bool_ask_D','bool_bid_D',\
                              'bool_idx', 'bool_trade', 'cumsum'],axis=1, inplace=True)
            
        self.__tob.ask_1_price.fillna(method='ffill', inplace=True)
        self.__tob.ask_1_qty.fillna(method='ffill', inplace=True)
        self.__tob.bid_1_price.fillna(method='ffill', inplace=True)
        self.__tob.bid_1_qty.fillna(method='ffill', inplace=True)
        
    def __fill_tob_price_last(self):
        self.__tob['bid_1_price_last'] = self.__Armada_Data.df.bid_1_price.copy()
        self.__tob.bid_1_price_last.fillna(method='ffill', inplace=True)
        
        self.__tob['ask_1_price_last'] = self.__Armada_Data.df.ask_1_price.copy()
        self.__tob.ask_1_price_last.fillna(method='ffill', inplace=True)
       
    def __fill_price_traded(self):
        self.__tob['bid_price_traded'] = self.__tob.bid_1_price_last >=\
        self.__Armada_Data.df.trade_price
        
        self.__tob['ask_price_traded'] = self.__tob.ask_1_price_last <=\
        self.__Armada_Data.df.trade_price
    
    def __fill_aggression(self):
        self.__tob['aggression'] = np.vectorize(self.aggression_id)\
        (self.__tob.bid_price_traded, self.__tob.ask_price_traded,\
        self.__Armada_Data.get_order_indicator())
            
    def get_net_number_aggression(self, start_time, end_time):
        start = self.__Armada_Data.get_processing_date() + start_time
        end = self.__Armada_Data.get_processing_date() + end_time
        df_aggression = pd.DataFrame(self.__tob.aggression)
        df_aggression = \
            df_aggression.set_index(self.__Armada_Data.df.DateTime.copy())
        df_aggression = df_aggression.loc[start:end]
        number_aggression = np.nansum(df_aggression)
        return number_aggression
    
    def get_rolling_number_aggression(self, start_time, end_time):
        date = self.processing_date
        start_lag =  date + start_time - pd.to_timedelta('00:00:01')
        start = date + start_time
        end = date + end_time
        
        df_aggression = pd.DataFrame(self.__tob.aggression)
        df_aggression = \
            df_aggression.set_index(self.__Armada_Data.df.DateTime.copy())
        df_aggression = df_aggression.loc[start_lag:end]
        #df_aggression = abs(df_aggression)
        number_aggression = df_aggression.rolling('1s').sum()
        
        number_aggression = number_aggression.loc[start:end]
        return number_aggression 
        
    def __find_num_consecutive_trade(self):
        boolean_series = self.__Armada_Data.get_trade_indicator()
        cumsum = boolean_series.cumsum()
        temp = cumsum.sub(cumsum.mask(boolean_series).ffill().fillna(0)).astype(int)
        num_consecutive_trade = temp.max()
        ###can be deleted later#
        self.__tob['cumsum'] = temp
        ###
        return num_consecutive_trade
    
    # %% Depletions

    def __depletion(self):
        '''depletions(data_frame, tick_value) flags depletions and fills on the
        queue calculated by df_previous_tob returning new fields that indicate
        which side (Bid or Ask) was depleted or filled and whether the
        depletions were caused by a trade or a cancel'''
        tick_value = self.tick_value
        orders_idx_shift = self.__Armada_Data.get_order_indicator().shift(+1)
        orders_idx_shift.fillna(False, inplace=True)
        bid_diff = self.__tob.bid_1_price.diff()/tick_value
        ask_diff = self.__tob.ask_1_price.diff()/tick_value
        
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
        bid_1_price = self.__Armada_Data.df.bid_1_price.copy()
        ask_1_price = self.__Armada_Data.df.ask_1_price.copy()
        bid_1_qty = self.__Armada_Data.df.bid_1_qty.copy()
        ask_1_qty = self.__Armada_Data.df.ask_1_qty.copy()
        bid_2_price = self.__Armada_Data.df.bid_2_price.copy()
        ask_2_price = self.__Armada_Data.df.ask_2_price.copy()
        bid_2_qty = self.__Armada_Data.df.bid_2_qty.copy()
        ask_2_qty = self.__Armada_Data.df.ask_2_qty.copy()
        
        
        mid_price = (bid_1_price + ask_1_price) / 2
        
        micro_price = self.__microprice(self.__tob.bid_1_qty,\
            self.__tob.bid_1_price, self.__tob.ask_1_qty,\
            self.__tob.ask_1_price)
            
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
        
        spread_ticks = self.__spread_ticks(self.__tob.bid_1_price, \
                                           self.__tob.ask_1_price\
                                               , tick_value)
        imbalance = self.__imbalance(self.__tob.bid_1_qty,\
            self.__tob.ask_1_qty)
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
    def delta_t_weight_event(self,field):
        mask1 = self.__Armada_Data.get_order_indicator()
        mask2 = self.ord_indic.spread_1_ticks > 0
        df_masked = self.ord_indic[mask1 & mask2]
        df_masked['delta_t'] = \
            self.__Armada_Data.df.DateTime[mask1 & mask2].diff().shift(-1)
        return ((df_masked[field]*df_masked['delta_t']).sum())/\
            (df_masked['delta_t'].sum())

    def get_time_weighted_tob(self):
        '''time_weighted_spread(data_frame) returns the time weighted spread
        in ticks, given the data_frame'''
        df_columns = ['tw_spr1', 'tw_spr2', 'tw_bid1tomid', 'tw_ask1tomid', 'tw_bid1qty',\
            'tw_ask1qty', 'tw_bid12qty', 'tw_ask12qty', 'tw_bid12tomid', 'tw_ask12tomid']
        df_dict = dict(zip(df_columns, len(df_columns)*[0]))
        
        #df_orders = df_imbl[df_imbl['OT']].copy()
        mask1 = self.__Armada_Data.get_order_indicator()
        mask2 = self.ord_indic.spread_1_ticks > 0
        #data_frame_masked = data_framec[data_framec['Spread_Lvl_1_Ticks'] > 0]
        
        df_masked = self.ord_indic[mask1 & mask2]
        df_masked['delta_t'] = \
            self.__Armada_Data.df.DateTime[mask1 & mask2].diff().shift(-1)
        
        def delta_t_weigh(field):
            return ((df_masked[field]*df_masked['delta_t']).sum())/\
                (df_masked['delta_t'].sum())
                
        df_dict['tw_spr1'] = delta_t_weigh('spread_1_ticks')
        df_dict['tw_spr2'] = delta_t_weigh('spread_2_ticks')
        df_dict['tw_bid1tomid'] = delta_t_weigh('bid_to_midprice')
        df_dict['tw_ask1tomid'] = delta_t_weigh('ask_to_midprice')
        df_dict['tw_bid12tomid'] = delta_t_weigh('bid_12_to_midprice')
        df_dict['tw_ask12tomid'] = delta_t_weigh('ask_12_to_midprice')
        
        df_masked = self.__Armada_Data.df[mask1 & mask2]
        
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
        date = self.processing_date
        start = (date + start_time)
        end = (date + end_time)
        data_to_print = self.__tob.copy()
        data_to_print['DateTime'] = self.__Armada_Data.df.DateTime.copy()
        data_to_print = \
            data_to_print.set_index(self.__Armada_Data.df.DateTime.copy())
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
        
    def print2file_df_intensity(self,pathout):
        file_name = pathout+'df_intensity_bid.csv'
        print('Saving file: ',file_name)
        
        intensity_bid = self.results_bid.copy()
        intensity_ask = self.results_ask.copy()
        
        intensity_bid = intensity_bid.reset_index()
        intensity_ask = intensity_ask.reset_index()
        
        intensity_bid = intensity_bid.rename(columns={"consumption": "order_type", "intensity": "Intensity", "delta_t": "var_DateTime", "bid_event": "Number"})
        intensity_ask = intensity_ask.rename(columns={"consumption": "order_type", "intensity": "Intensity", "delta_t": "var_DateTime", "ask_event": "Number"})
        
        intensity_bid.order_type = np.where(intensity_bid.order_type == True, 'Consumption', 'Insertion')
        intensity_ask.order_type = np.where(intensity_ask.order_type == True, 'Consumption', 'Insertion')
        
        intensity_bid.to_csv(file_name) 
        file_name2 = pathout+'df_intensity_ask.csv'
        print('Saving file: ',file_name2)
        intensity_ask.to_csv(file_name2) 
        
        
        
# %% Plot Functions
        
    def plot_html_1sec_rolling_number_aggression(self, start_time, end_time, path_out):
        number_aggression = self.get_rolling_number_aggression(start_time, end_time)
        
        #layout = self.__plot_html_layout('Bid Qty', 'Ask Qty', 'Quantity Time Series')
        fig = go.Figure([go.Scatter(x=number_aggression.index, y=number_aggression.aggression)])
        fig.update_layout(title_text="1 Second Rolling Net Number of Aggression")
        file = path_out+self.__Armada_Data.file_name+"_1sec_roll_net_number_aggression.html"
        print('saving html plot to ', file)
        fig.write_html(file) 

    def plot_html_tob_event(self,path_out, \
                                 start_xaxis = pd.to_timedelta('07:00:00'),\
                                 start_end_delta = pd.to_timedelta('01:00:00'),\
                                 start_delta = pd.to_timedelta('01:00:00')):
        
        # time management
        date = self.processing_date
        start_data_delta = start_xaxis - start_delta
        #start_end_delta = pd.to_timedelta('01:30:00')
        end_xaxis = start_xaxis + start_end_delta
        #tob data
        data = self.__tob.copy()
        data = data.set_index(self.__Armada_Data.df.DateTime)
        traded_qty_signed = np.where(data.bid_price_traded, \
                                     -1*data.traded_qty, +1*data.traded_qty)
        #ord data
        data_ord = self.ord_indic.copy()
        data_ord['traded_qty_signed']= traded_qty_signed
        data_ord = data_ord.set_index(self.__Armada_Data.df.DateTime)
        #event data
        data_event = self.get_rolling_event(start_data_delta, end_xaxis)
        
        data_event['delta_t'] = \
            data_event.index.to_series().diff().shift(-1)
        data_event['bid_net_intensity'] = \
            data_event.bid_insertion - data_event.bid_consumption 
        data_event['ask_net_intensity'] = \
            data_event.ask_insertion - data_event.ask_consumption
        data_event['net_intensity'] = data_event.ask_net_intensity + \
            data_event.bid_net_intensity 
        
        data_event_tw = data_event.loc[(date + start_data_delta):(date+start_xaxis)]
        
        # time weighted average metrics
        def delta_t_weigh(field):
            return ((data_event_tw[field]*data_event_tw['delta_t']).sum())/\
                (data_event_tw['delta_t'].sum())
                
        data_event['tw_bid_size']= delta_t_weigh('bid_size')
        data_event['tw_ask_size']= delta_t_weigh('ask_size')
        data_event['tw_spread']= delta_t_weigh('spread')/self.tick_value
        data_event['tw_ask_net_intensity']= delta_t_weigh('ask_net_intensity')
        data_event['tw_bid_net_intensity']= delta_t_weigh('bid_net_intensity')
        data_event['tw_net_intensity']= delta_t_weigh('net_intensity')
        
        # data filtered for plot
        data_event = data_event.loc[(date + start_xaxis):(date+end_xaxis)]
        data = data.loc[(date + start_xaxis):(date+end_xaxis)]
        data_ord = data_ord.loc[(date + start_xaxis):(date+end_xaxis)]
        
        # plotting
        fig = make_subplots(rows=5, cols=1, row_width=[0.1,0.4,0.2, 0.2, 0.4], 
                            shared_xaxes=True, vertical_spacing=0.01)
        fig.update_layout(title_text="Top of Book from " + \
                         (date+start_xaxis).strftime('%Y-%m-%d %H:%M:%S') + " to " \
                        +(date+end_xaxis).strftime('%Y-%m-%d %H:%M:%S'),\
                        legend=dict(y=0.5, font_size=8))
        fig.add_trace(go.Scattergl(x=data.index, y=data.bid_1_price,
                                   line=dict(color='#1f77b4'),# muted blue
                            name = 'Bid', line_shape='hv', opacity= 0.5), row=1, col=1)
        fig.add_trace(go.Scattergl(x=data.index, y=data.ask_1_price, 
                                   line=dict(color='brown'),#brick red
                            name = 'Ask', line_shape='hv', opacity= 0.5), row=1, col=1)
        fig.add_trace(go.Scattergl(x=data.index, y=data_ord.micro_price, 
                                   line=dict(color='#7f7f7f'),visible = 'legendonly',# grey
                            name = 'Micro price', line_shape='hv'), row=1, col=1)
        ### 
        fig.add_trace(go.Scattergl(x=data_event.index, y= - data_event.bid_size,
                                   line=dict(color='#1f77b4'),
                            name = 'Bid qty', line_shape='hv', opacity= 0.5), row=2, col=1)
        fig.add_trace(go.Scattergl(x=data_event.index, y= - data_event.tw_bid_size,
                                   line=dict(color='midnightblue'),
                            name = 'Time Weighted Bid Qty (1h)', line_shape='hv', opacity= 0.9), row=2, col=1)
        
        fig.add_trace(go.Scattergl(x=data_event.index, y=data_event.ask_size, 
                                   line=dict(color='#d62728'),
                            name = 'Ask qty', line_shape='hv', opacity= 0.5), row=2, col=1)
        fig.add_trace(go.Scattergl(x=data_event.index, y= data_event.tw_ask_size,
                                   line=dict(color='firebrick'),
                            name = 'Time Weighted Ask Qty (1h)', line_shape='hv', opacity= 0.9), row=2, col=1)
        
        ###
        fig.add_trace(go.Scattergl(x=data.index, y=data_ord.spread_ticks, 
                                   line=dict(color='midnightblue'),
                            name = 'Bid-ask spread in ticks', line_shape='hv',opacity= 0.5), row=3, col=1)
        fig.add_trace(go.Scattergl(x=data.index, y=data_event.tw_spread, 
                                   line=dict(color='black'),
                            name = 'Time Weighted Spread (1h)', line_shape='hv',opacity= 0.5), row=3, col=1)
        
        ###
        fig.add_trace(go.Scattergl(x=data_event.index, y=data_event.bid_consumption, 
                                   line=dict(color='#1f77b4'),
                            name = 'Bid Consumption', line_shape='hv', visible = 'legendonly'), row=4, col=1)
        fig.add_trace(go.Scattergl(x=data_event.index, y=data_event.ask_consumption, 
                                   line=dict(color='#d62728'),
                            name = 'Ask Consumption', line_shape='hv', visible = 'legendonly'), row=4, col=1)
        fig.add_trace(go.Scattergl(x=data_event.index, y=data_event.bid_insertion, 
                                   line=dict(color='#1f77b4'),
                            name = 'Bid Insertion', line_shape='hv', visible = 'legendonly'), row=4, col=1)
        fig.add_trace(go.Scattergl(x=data_event.index, y=data_event.ask_insertion, 
                                   line=dict(color='#d62728'),
                            name = 'Ask Insertion', line_shape='hv', visible = 'legendonly'), row=4, col=1)
        fig.add_trace(go.Scattergl(x=data_event.index, y=data_event.bid_net_intensity, 
                                   line=dict(color='#1f77b4'),
                            name = 'Bid Net Intensity', line_shape='hv',opacity= 0.5), row=4, col=1)
        fig.add_trace(go.Scattergl(x=data_event.index, y=data_event.ask_net_intensity, 
                                   line=dict(color='#d62728'),
                            name = 'Ask Net Intensity', line_shape='hv',opacity= 0.5), row=4, col=1)
        fig.add_trace(go.Scattergl(x=data_event.index, y=data_event.net_intensity, 
                                   line=dict(color='#7f7f7f'),
                            name = 'Net Intensity', line_shape='hv'), row=4, col=1)
        fig.add_trace(go.Scattergl(x=data_event.index, y=data_event.tw_net_intensity, 
                                   line=dict(color='black'),
                            name = 'Time Weighted Net Intensity (1h)', line_shape='hv'), row=4, col=1)
        fig.add_trace(go.Scattergl(x=data_event.index, y=data_event.tw_ask_net_intensity, 
                                   line=dict(color='red'),
                            name = 'Time Weighted Ask Net Intensity (1h)', line_shape='hv', visible = 'legendonly'), row=4, col=1)
        fig.add_trace(go.Scattergl(x=data_event.index, y=data_event.tw_bid_net_intensity, 
                                   line=dict(color='blue'),
                            name = 'Time Weighted Bid Net Intensity (1h)', line_shape='hv',visible = 'legendonly'), row=4, col=1)
        
        ###
        fig.add_trace(go.Scattergl(x=data.index, y=data_ord.traded_qty_signed, 
                                   line=dict(color='midnightblue'), 
                            name = 'Signed trade qty', mode='markers', opacity= 0.5), row=5, col=1)
        
        fig.update_yaxes(title_text="bid, ask, micro price", row=1, col=1)
        fig.update_yaxes(title_text="Top Order Book Quantities", row=2, col=1)
        fig.update_yaxes(title_text="Bid-ask spread", row=3, col=1)
        fig.update_yaxes(title_text="Intensities per second", row=4, col=1)
        fig.update_yaxes(title_text="Signed Traded Quantities", row=5, col=1)
        
        #fig.update_layout(shapes=[dict(type= 'line',yref= 'paper', y0= 0, y1= tw_metrics.bid_size, xref= 'x', x0= 5, x1= 5)])
        #fig['data'][2].update(yaxis='y5')
        #fig['layout']['yaxis5'] = dict(overlaying='y2', anchor='x2', side='right', showgrid=False, title='Trades')
        file = path_out+self.file_name_long+"_event.html"
        print('saving html plot to ', file)
        fig.write_html(file)        
        
    def plot_html_tob_1mintick(self,path_out, start_xaxis = pd.to_timedelta('07:30:00')):
        #layout = self.__plot_html_layout('Bid Price', 'Ask Price', 'Price Time Series')
        data = self.__tob.copy()
        data = data.set_index(self.__Armada_Data.df.DateTime)
        
        traded_qty_signed = np.where(data.bid_price_traded, \
                                     -1*data.traded_qty, +1*data.traded_qty)
        
        
        data_ord = self.ord_indic.copy()
        data_ord['traded_qty_signed']= traded_qty_signed
        data_ord = data_ord.set_index(self.__Armada_Data.df.DateTime)
        
        start_xaxis = self.processing_date + start_xaxis
        end_xaxis = start_xaxis + pd.to_timedelta('00:01:00')
        
        data = data.loc[start_xaxis:end_xaxis]
        data_ord = data_ord.loc[start_xaxis:end_xaxis]
        
        #data['trade_price_minus_1']=data.trade_price.shift(-1)
        #data['buy']=np.where((data['trade_price_minus_1']==data.bid_1_price)
        #                     & (data['trade_price_minus_1'].notna())
        #                     & (data.bid_1_price.notna())
        #                     , True, False)
        
        fig = make_subplots(rows=4, cols=1, row_width=[0.2,0.2, 0.2, 0.4], 
                            shared_xaxes=True)
        fig.update_layout(title_text="One Minute Top of Book from " + \
                         start_xaxis.strftime('%Y-%m-%d %H:%M:%S') + " to " \
                        +end_xaxis.strftime('%Y-%m-%d %H:%M:%S'),\
                        legend=dict(y=0.5, traceorder='reversed', font_size=10))
        fig.add_trace(go.Scatter(x=data.index, y=data.bid_1_price, 
                            name = 'Bid', line_shape='hv'), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data.ask_1_price, 
                            name = 'Ask', line_shape='hv'), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data_ord.micro_price, 
                            name = 'Micro price', line_shape='hv'), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data_ord.traded_qty_signed, 
                            name = 'Signed trade qty', line_shape='hv'), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data_ord.spread_ticks, 
                            name = 'Bid-ask spread in ticks', line_shape='hv'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data_ord.imbalance, 
                            name = 'Imbalance', line_shape='hv'), row=4, col=1)
        
        
        file = path_out+self.file_name_long+"_tob_1min_tick.html"
        print('saving html plot to ', file)
        fig.write_html(file)


# %% Top-Of-Book Intensity Output Class, Allow multi days stats
    
class TOB_Intensity_Output:
    
    def __init__(self, processing_date, results_bid =pd.DataFrame(), results_ask= pd.DataFrame()):
        self.results_bid = results_bid
        self.results_ask = results_ask
        self.processing_date = processing_date
        self.results_bid['Date']=self.processing_date
        self.results_ask['Date']=self.processing_date
    
    def append(self, TOB_Intensity_Output):
        self.results_bid = self.results_bid.append(\
            TOB_Intensity_Output.results_bid, ignore_index = True)
        self.results_ask = self.results_ask.append(\
            TOB_Intensity_Output.results_ask, ignore_index = True)
    
    def __get_intensity(self,bid_ask):
        
        
        if bid_ask == 'bid':
            intensity_bid = self.results_bid.copy()
            intensity_bid = intensity_bid.rename(columns={'bid_event': "Number"})
            intensity = intensity_bid
        
        elif bid_ask == 'ask':
            intensity_ask = self.results_ask.copy()
            intensity_ask = intensity_ask.rename(columns={'ask_event': "Number"})
            intensity = intensity_ask
            
        else:
            intensity_bid = self.results_bid.copy()
            intensity_ask = self.results_ask.copy()
            intensity_bid = intensity_bid.rename(columns={'bid_event': "Number"})
            intensity_ask = intensity_ask.rename(columns={'ask_event': "Number"})
            intensity = intensity_bid.append(intensity_ask, ignore_index = True)
        
        intensity = intensity.rename(columns={"consumption": "order_type", "intensity": "Intensity", "delta_t": "var_DateTime"})
        
        intensity.order_type = np.where(intensity.order_type == True, 'Consumption', 'Insertion')
        
        #intensity_bid = intensity_bid.reset_index()
        #intensity_ask = intensity_ask.reset_index()
        #intensity_group = Intensity(intensity)
        intensity_group = intensity.groupby(['size_before', 'order_type']).agg({'Number':'sum', 'var_DateTime':'sum'}) 
        intensity_group = intensity_group.reset_index()
        
        intensity_group['intensity'] = intensity_group.Number\
            / intensity_group.var_DateTime
        intensity_group = intensity_group.rename(columns={"intensity": "Intensity"})
        return Intensity(intensity_group)
            
        
    
    def get_bid_intensity(self):
        return self.__get_intensity('bid')
    
    def get_ask_intensity(self):
        return self.__get_intensity('ask')
    
    def get_aggregated_bid_ask(self):
        return self.__get_intensity('both')
        

    
# %% Class Intensity to Compute intensities
        
class Intensity:
    
    def __init__(self,df_intensity=pd.DataFrame()):
        self.intensity = pd.DataFrame()
        self.intensity = df_intensity
        self.q = 1
        self.Qmax0 = 40 # HARDCODED #int(max(df_intensity.size_before))
        ##### Compute intensities
        self.intensity_values = self.__compute_intens_val_bis()
        ##### Compute Q matrix
        self.q_no_regen = self.__build_Q_no_regen()
        ### Compute stationary probabilities 
        self.proba = self.__proba_stat()
    
            
    def __build_Q_no_regen(self):
        " IntensVal structure is pd.DataFrame(np.zeros((Qmax0*Qmax0,4)),columns=['BidQtyBefore','AskQtyBefore','lambdaCancel','lambdaIns']) "
        " q is the minimum order size "
        " Qmax0 is the maximum order size level "
        " RegenVect structure is np.zeros((2*Qmax0,Qmax0*Qmax0))"
        ## Build transition matrix finite difference scheme
        IntensVal = self.intensity_values.copy()
        Qmax0 = self.Qmax0
    
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
    def __compute_intens_val_bis(self):
        IntensVal_whole_market = self.intensity.copy()
        q = self.q
        Qmax0 = self.Qmax0
        res_sub = IntensVal_whole_market[IntensVal_whole_market['size_before'] <= Qmax0]
        
        IntensVal = pd.DataFrame(np.zeros((Qmax0*Qmax0,4)), columns = ['BidQtyBefore', 'AskQtyBefore', 'lambdaCancel', 'lambdaIns'])
        IntensVal['BidQtyBefore'] = np.repeat(np.arange(1,Qmax0+1)*q,Qmax0)
        IntensVal['AskQtyBefore'] = np.tile(np.arange(1,Qmax0+1)*q,Qmax0)
        IntensVal['lambdaCancel'] = np.repeat(res_sub[(res_sub['order_type'] == 'Consumption') ]['Intensity'].values,Qmax0)
        IntensVal['lambdaIns'] = np.repeat(res_sub[(res_sub['order_type'] == 'Insertion') ]['Intensity'].values,Qmax0)
        return IntensVal
    
    ## compute stationary probabilities
    def __proba_stat(self):
        Qmax0 = self.Qmax0
        Tilde_Q = self.q_no_regen
        size = Qmax0*Qmax0
        Tilde_Q_inv = np.array(Tilde_Q[:-1,:-1]) 
        for j in range(size-1):
    
            Tilde_Q_inv[:,j] -= Tilde_Q[-1,j]  
        F_inv = -Tilde_Q[size-1,:-1]
        ## Compute the stat proba
        Proba2 = np.zeros((size))
        Proba2[:-1] = np.linalg.solve(Tilde_Q_inv.transpose(),F_inv.transpose());Proba2[-1]  = 1-sum(Proba2)
        return Proba2
    
              
    def plot_intensities(self,pathout,second_intens = None, file_name=''):
        Qmax0 = self.Qmax0
        q = self.q
        xpos1 = np.repeat(q*np.arange(1,Qmax0+1),Qmax0)
        ypos1 = np.tile(q*np.arange(1,Qmax0+1),Qmax0)
        
        if second_intens is not None:
            # intensity
            IntensVal_2 = second_intens.intensity_values.copy()
            x_ = IntensVal_2.BidQtyBefore
            y_= IntensVal_2.lambdaCancel
            y_2_ = IntensVal_2.lambdaIns
            #proba
            proba_ = second_intens.proba.copy()
            data_frame_ = pd.DataFrame(np.zeros((Qmax0*Qmax0,3)),columns=['x','y','Prob'])
            data_frame_['x'] = xpos1
            data_frame_['y'] = ypos1
            data_frame_['Prob'] = proba_
            res_bis_ = data_frame_.groupby(['x']).agg({'Prob':'median'})
        
    
        # instensity plot data preparation
        IntensVal = self.intensity_values.copy()
        x = IntensVal.BidQtyBefore
        y = IntensVal.lambdaCancel
        y_2 = IntensVal.lambdaIns
        
        # Probability plot data prepatation
        proba = self.proba.copy()
        data_frame = pd.DataFrame(np.zeros((Qmax0*Qmax0,3)),columns=['x','y','Prob'])
        data_frame['x'] = xpos1
        data_frame['y'] = ypos1
        data_frame['Prob'] = proba
        res_bis = data_frame.groupby(['x']).agg({'Prob':'median'})
        
        # Plot prepare
        fig = make_subplots(rows=2, cols=1, row_width=[0.2, 0.4], 
                            shared_xaxes=True, vertical_spacing=0.01)
        
        fig.add_trace(go.Scatter(x=res_bis.index.values, \
                    y=res_bis.values.flatten()/res_bis.values.sum(), line=dict(color='black', width=1),name = 'Probability Distribution', mode='lines+markers'), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=y, line=dict(color='royalblue', width=1),name ='Liquidity Consumption' , mode='lines+markers'), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=y_2,line=dict(color='red', width=1), name ='Liquidity Provision', mode='lines+markers'), row=1, col=1)
        
        fig.update_xaxes(title_text="", zeroline=True,row=1, col=1)
        fig.update_xaxes(title_text="Quantity", zeroline=True, row=2, col=1)
        fig.update_yaxes(title_text="Liquidity Intensities",rangemode="tozero",row=1, col=1, )
        fig.update_yaxes(title_text="Intensity Probability Distribution",rangemode="tozero",row=2, col=1, )
        
        if second_intens is not None:
            fig.add_trace(go.Scatter(x=res_bis_.index.values, \
                    y=res_bis_.values.flatten()/res_bis_.values.sum(),line=dict(color='black', width=1, dash='dot'), name = 'Probability Distribution 2', mode='lines+markers'), row=2, col=1)
            fig.add_trace(go.Scatter(x=x_, y=y_, line=dict(color='royalblue', width=1, dash = 'dot'),name ='Liquidity Consumption 2' , mode='lines+markers'), row=1, col=1)
            fig.add_trace(go.Scatter(x=x_, y=y_2_, line=dict(color='red', width=1, dash='dot'),name ='Liquidity Provision 2', mode='lines+markers'), row=1, col=1)
        
        
        fig.update_layout(\
            title={
        'text': "Liquidity Provision and Consumption with Probability Distribution",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},legend=dict(y=0.5, font_size=10))
        
        #fig = go.Figure(data=go.Scatter(x=res_bis.index.values, y=res_bis.values.flatten()/res_bis.values.sum()))
        #fig.update_layout(title_text="Probability Distribution of Intensities",
        #          title_font_size=30)
        file = pathout+"proba_"+file_name+".html"
        print('saving html plot to ', file)
        fig.write_html(file)
        
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
        data_frame = self.df_trades.copy()
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
        
        ''' def __collapse_time2(self):
        collapse_time(data_frame) takes a Data Frame of trades at various
        times and prices that might have different amounts traded at the same
        time(s) and price(s) and collapses these trades into these same times
        and prices, adding the 'Trade Qty' field for each collapsed group.
        In short: Group trades by time and price, sum(Trade Qty)
        Inputs: Data Frame from Armada preprocessed by column_datetime and
        column_ot
        Outputs: Smaller (or equal) DataFrame with unique time, price pairs
        and the values of 'Trade Qty' grouped with sum()
        data_frame = self.df_trades.copy()
        data_framecol = data_frame.columns.tolist()
        data_framec = data_frame.copy()
        data_framecl = data_framec.columns.tolist()
        data_framecl.remove('trade_qty')
        df_unique = data_frame.drop_duplicates(subset = ['DateTime', 'trade_price', 'trade_qty'])
        df_grouped = data_frame[['DateTime','trade_price']].groupby('trade_qty').sum()
        
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
        self.df_trades = df_grouped_at_first   '''

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
        
