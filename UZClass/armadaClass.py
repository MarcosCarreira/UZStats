
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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# %% Armada Data Class

class Armada_Data():
    
    def __init__(self, file_path, file_name):
        self.file_name = file_name
        self.file_path = file_path
        print('Reading file '+file_path+file_name)
        self.__read_ArmadaData()
        print('Re-formatting Data')
        self.__column_datetime()
        print('Armada Data Object Construction Sucessfull')
        self.__rename_columns()
        
    def __read_ArmadaData(self):
        self.data_frame = pd.read_csv(self.file_path+self.file_name \
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
        self.data_frame.rename(columns = {'Bid 1 Qty':'bid_1_qty',\
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
        self.data_frame.drop(['Time.1', 'Date.1','Index','Index.1',\
                              'Date', 'Time'],axis=1, inplace=True)
        
        
    def plot_html_1mintick(self,path_out, start_xaxis = pd.to_timedelta('07:30:00')):
        #layout = self.__plot_html_layout('Bid Price', 'Ask Price', 'Price Time Series')
        data = self.data_frame.copy()
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
        fig.add_trace(go.Scatter(x=data.index, y=self.__get_BAspread(data), 
                            name = 'Bid-ask spread'), row=3, col=1)
        
        
        file = path_out+self.get_file_name()+"_price.html"
        print('saving html plot to ', file)
        fig.write_html(file)
        
    def plot_html_qty(self,path_out):
        layout = self.__plot_html_layout('Bid Qty', 'Ask Qty', 'Quantity Time Series')
        bid_data = go.Scatter(x=self.data_frame.DateTime,
                         y=self.data_frame.bid_1_qty)
        ask_data = go.Scatter(x=self.data_frame.DateTime,
                        y=self.data_frame.ask_1_qty)
        fig = go.Figure(data=[bid_data, ask_data], layout=layout)
        fig.update_layout(height=600, width=600, title_text="Stacked Subplots")
        file = path_out+self.get_file_name+"_qty.html"
        print('saving html plot to ', file)
        fig.write_html(file)  
        
    def __plot_html_layout(self,yaxis_text, yaxis2_text, title_text):
        layout = go.Layout(height=1000, width=1400,
                   title=title_text,
                   # Same x and first y
                   xaxis=dict(title='Date'),
                   yaxis=dict(title=yaxis_text, color='red'),
                   # Add a second yaxis to the right of the plot
                   yaxis2=dict(title=yaxis2_text, color='blue',
                               overlaying='y', side='right')
                   )
        return layout
    
    def __custom_resampler_abs_sum(self,array):
        return np.sum(np.abs(array))
    
    def _time_weighted_spread(self, data):
        return ((data['ba_spread']*data['dt']).sum())/\
            (data['dt'].sum())
            
    def _get_ba_spread(self):
        df = self.data_frame.copy()
        df['ba_spread'] = (df.ask_1_price - df.bid_1_price)
        df = df.set_index(self.data_frame['DateTime'])
        return  df.ba_spread
        
    def _get_delta_t(self):
        df = self.data_frame.copy()
        df['dt'] = self.data_frame.DateTime.diff().shift(-1)
        df = df.set_index(self.data_frame['DateTime'])
        return  df.dt
        
    
    def plot_html_ohlc(self, pathout, 
                       freq='1min',start_xaxis = pd.to_timedelta('09:00:00'),
                       end_xaxis = pd.to_timedelta('16:00:00')):
        
        data = self.data_frame.copy()
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
        data['ba_spread'] = self._get_ba_spread()
        data['dt'] = self._get_delta_t() 
        data_masked = data[data.ba_spread > 0]
        time_weighted_spread = pd.DataFrame(\
            data_masked.groupby(pd.Grouper(freq=freq)).apply(self._time_weighted_spread))
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

          
        file = pathout+self.get_processing_date().strftime('%Y%m%d')+freq+"ohlc.html"
        print('saving html plot to ', file)
        fig.write_html(file)

    
    def __get_BAspread(self, data):
        spread = (data.ask_1_price - data.bid_1_price)
        return spread
    
    def __column_datetime(self):
        '''column_datetime(data_frame) creates the column 'DateTime'
        concatenating 'Date' and 'Time' and converting it to datetime.
        Inputs: Data Frame from Armada
        Outputs: Copy of DataFrame with 'DateTime' column added'''
        data_framec = self.data_frame.copy()
        data_framec['Date'] = pd.to_datetime(data_framec['Date'],\
            format="%m/%d/%Y")
        data_framec['Time'] = pd.to_timedelta(data_framec['Time'], unit='ns')
        data_framec['DateTime'] = data_framec['Date']+data_framec['Time']
        self.data_frame = data_framec
        
    def get_file_name(self):
        return self.file_name[:-4]
    
    def get_processing_date(self):
        return pd.to_datetime(self.file_name[0:8],format='%Y%m%d')
    
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
        self.df = Armada_Data.data_frame
        self.file_out = Armada_Data.get_file_name()
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
        fig.add_trace(go.Scatter(x=self.data_frame.DateTime,y=self.__get_BAspread(self.data_frame)), row=2, col=1)
        fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
        
        file = path_out+self.get_file_name()+"_price.html"
        print('saving html plot to ', file)
        fig.write_html(file)
    
    def get_Armada_UZModel_output(self):
        return Armada_UZModel_output(self.df_cont_alt_by_ticks, self.df_uz_stats)
        


# %% Agression functions

def aggression_id(bid_traded, ask_traded, order_flag):
    '''aggression_id(bid_traded, ask_traded, order_flag) returns
    aggression_value, which is equal to:
    0 for an order event (no trade)
    +1 for a trade at the ask
    -1 for a trade at the bid
    nan for a trade without a previous bid/ask'''
    if order_flag:
        aggression_value = 0
    else:
        if bid_traded:
            aggression_value = -1
        elif ask_traded:
            aggression_value = +1
        else:
            aggression_value = np.nan
    return aggression_value

def fill_aggression_id(data_frame):
    '''fill_aggression_id(data_frame) applies aggression_id to a data_frame,
    with the last bid and ask prices bracketing trade prices for the
    aggression flag'''
    data_framec = data_frame.copy()
    new_columns = list(data_frame.columns)+['Aggression']
    data_framec['Bid_Price_Last'] = data_framec['Bid 1 Price'].copy()
    data_framec['Ask_Price_Last'] = data_framec['Ask 1 Price'].copy()
    data_framec['Bid_Price_Last'].fillna(method='ffill', inplace=True)
    data_framec['Ask_Price_Last'].fillna(method='ffill', inplace=True)
    data_framec['Bid_Price_Traded'] = data_framec['Bid_Price_Last'] >=\
        data_framec['Trade Price']
    data_framec['Ask_Price_Traded'] = data_framec['Ask_Price_Last'] <=\
        data_framec['Trade Price']
    data_framec['Aggression'] = np.vectorize(aggression_id)\
        (data_framec['Bid_Price_Traded'], data_framec['Ask_Price_Traded'],\
        data_framec['OT'])
    return data_framec[new_columns]

def previous_tob(aggression_flag, trade_price, trade_quantity,\
        bid_quantity, bid_price, ask_price, ask_quantity):
    '''previous_tob_core(aggression_flag, trade_price, trade_quantity,\
    bid_quantity, bid_price, ask_price, ask_quantity) returns\
    a list with the new top of the book: bid_quantity, bid_price,\
    ask_price, ask_quantity'''
    top_of_book = {'Bid_Qty': bid_quantity, 'Bid_Price': bid_price,\
                   'Ask_Price': ask_price, 'Ask_Qty': ask_quantity}
    if bid_price > -1:
        if aggression_flag == -1:
            if trade_price == bid_price:
                top_of_book['Bid_Qty'] += trade_quantity
            else:
                top_of_book['Bid_Qty'] = trade_quantity
                top_of_book['Bid_Price'] = trade_price
            if trade_price == ask_price:
                top_of_book['Ask_Qty'] = np.nan
                top_of_book['Ask_Price'] = np.nan
        elif aggression_flag == 1:
            if trade_price == ask_price:
                top_of_book['Ask_Qty'] += trade_quantity
            else:
                top_of_book['Ask_Qty'] = trade_quantity
                top_of_book['Ask_Price'] = trade_price
            if trade_price == bid_price:
                top_of_book['Bid_Qty'] = np.nan
                top_of_book['Bid_Price'] = np.nan
    return list(top_of_book.values())

def init_previous_tob(data_frame):
    '''init_previous_tob(data_frame) applies previous_tob to a data_frame,
    returning new fields that indicate the previous top of book
    ('Bid_Qty', 'Bid_Price', 'Ask_Price', 'Ask_Qty')'''
    data_framec = data_frame.copy()
    data_framec['Bid_Qty'] = data_framec['Bid 1 Qty'].shift(-1)
    data_framec['Bid_Price'] = data_framec['Bid 1 Price'].shift(-1)
    data_framec['Ask_Price'] = data_framec['Ask 1 Price'].shift(-1)
    data_framec['Ask_Qty'] = data_framec['Ask 1 Qty'].shift(-1)
    new_columns = list(data_framec.columns)
    processed_list = list(map(previous_tob,\
        data_framec['Aggression'], data_framec['Trade Price'],\
        data_framec['Trade Qty'], data_framec['Bid_Qty'],\
        data_framec['Bid_Price'], data_framec['Ask_Price'],\
        data_framec['Ask_Qty']))
    new_data_frame = pd.DataFrame(np.array(processed_list),\
        columns=['Bid_Qty', 'Bid_Price', 'Ask_Price', 'Ask_Qty'],\
        index=data_framec.index)
    data_framec['Bid_Qty'] = np.where(data_framec['OT'],\
        data_framec['Bid 1 Qty'], new_data_frame['Bid_Qty'])
    data_framec['Bid_Price'] = np.where(data_framec['OT'],\
        data_framec['Bid 1 Price'], new_data_frame['Bid_Price'])
    data_framec['Ask_Price'] = np.where(data_framec['OT'],\
        data_framec['Ask 1 Price'], new_data_frame['Ask_Price'])
    data_framec['Ask_Qty'] = np.where(data_framec['OT'],\
        data_framec['Ask 1 Qty'], new_data_frame['Ask_Qty'])
    beg_orders = data_framec[data_framec['OT']].index[0]
    new_cols = ['Bid_Qty', 'Bid_Price', 'Ask_Price', 'Ask_Qty']
    data_framec.loc[:max(beg_orders-1, 0), new_cols] = np.nan
    return data_framec[new_columns]

def rec_previous_tob(data_frame):
    '''rec_previous_tob(data_frame) applies previous_tob to a data_frame,
    returning new fields that indicate the previous top of book
    ('Bid_Qty', 'Bid_Price', 'Ask_Price', 'Ask_Qty')'''
    data_framec = data_frame.copy()
    new_columns = list(data_framec.columns)
    data_framec['Bid_Qty_0'] = data_framec['Bid_Qty'].shift(-1)
    data_framec['Bid_Price_0'] = data_framec['Bid_Price'].shift(-1)
    data_framec['Ask_Price_0'] = data_framec['Ask_Price'].shift(-1)
    data_framec['Ask_Qty_0'] = data_framec['Ask_Qty'].shift(-1)
    processed_list = list(map(previous_tob,\
        data_framec['Aggression'], data_framec['Trade Price'],\
        data_framec['Trade Qty'], data_framec['Bid_Qty_0'],\
        data_framec['Bid_Price_0'], data_framec['Ask_Price_0'],\
        data_framec['Ask_Qty_0']))
    new_data_frame = pd.DataFrame(np.array(processed_list),\
        columns=['Bid_Qty', 'Bid_Price', 'Ask_Price', 'Ask_Qty'],\
        index=data_framec.index)
    data_framec['Bid_Qty'] = np.where(data_framec['OT'],\
        data_framec['Bid_Qty'], new_data_frame['Bid_Qty'])
    data_framec['Bid_Price'] = np.where(data_framec['OT'],\
        data_framec['Bid_Price'], new_data_frame['Bid_Price'])
    data_framec['Ask_Price'] = np.where(data_framec['OT'],\
        data_framec['Ask_Price'], new_data_frame['Ask_Price'])
    data_framec['Ask_Qty'] = np.where(data_framec['OT'],\
        data_framec['Ask_Qty'], new_data_frame['Ask_Qty'])
    beg_orders = data_framec[data_framec['OT']].index[0]
    new_cols = ['Bid_Qty', 'Bid_Price', 'Ask_Price', 'Ask_Qty']
    data_framec.loc[:max(beg_orders-1, 0), new_cols] = np.nan
    return data_framec[new_columns]

def find_max_run(boolean_series):
    '''find_max_run(boolean_series) finds the maximum number of
    consecutive trades ('OT' False)'''
    series_string = np.where(boolean_series, ' ', '1')
    string_boolean = pd.Series(series_string).sum()
    list_split = string_boolean.split()
    lengths_runs = np.vectorize(len)(pd.Series(list_split))
    return lengths_runs.max()

def loop_previous_tob(data_frame, max_runs=1):
    '''loop_previous_tob(data_frame, max_runs=1) applies
    previous_tob to a data_frame max_runs+2 times, returning
    new fields that indicate the previous top of book
    ('Bid_Qty', 'Bid_Price', 'Ask_Price', 'Ask_Qty')'''
    data_framec = data_frame.copy()
    print('original')
    loop_1_df = init_previous_tob(data_framec)
    print('init 0')
    loop_0_df = data_frame.copy()
    j = 0
    while j <= max_runs:
        loop_0_df = loop_1_df.copy()
        loop_1_df = rec_previous_tob(loop_0_df)
        if j%10 == 0:
            print('loop '+str(j))
        j += 1
    prev_tob_df = loop_1_df.copy()
    prev_tob_df['Bid_Qty'].fillna(method='ffill', inplace=True)
    prev_tob_df['Bid_Price'].fillna(method='ffill', inplace=True)
    prev_tob_df['Ask_Price'].fillna(method='ffill', inplace=True)
    prev_tob_df['Ask_Qty'].fillna(method='ffill', inplace=True)
    beg_orders = prev_tob_df[prev_tob_df['OT']].index[0]
    new_cols = ['Bid_Qty', 'Bid_Price', 'Ask_Price', 'Ask_Qty']
    prev_tob_df.loc[:max(beg_orders-1, 0), new_cols] = np.nan
    print('end')
    return prev_tob_df

# %% Depletions

def depletions(data_frame, tick_value):
    '''depletions(data_frame, tick_value) flags depletions and fills on the
    queue calculated by df_previous_tob returning new fields that indicate
    which side (Bid or Ask) was depleted or filled and whether the
    depletions were caused by a trade or a cancel'''
    data_framec = data_frame.copy()
    data_framec['OT_shift'] = data_framec['OT'].shift(+1)
    data_framec['OT_shift'].fillna(False, inplace=True)
    data_framec['Bid_Diff'] = data_framec['Bid_Price'].diff()/tick_value
    data_framec['Ask_Diff'] = data_framec['Ask_Price'].diff()/tick_value
    data_framec['Bid_Depl_Trade'] = (data_framec['Bid_Diff'] < 0) &\
        (~ data_framec['OT_shift'])
    data_framec['Bid_Depl_Cancel'] = (data_framec['Bid_Diff'] < 0) &\
        data_framec['OT_shift']
    data_framec['Ask_Depl_Trade'] = (data_framec['Ask_Diff'] > 0) &\
        (~ data_framec['OT_shift'])
    data_framec['Ask_Depl_Cancel'] = (data_framec['Ask_Diff'] > 0) &\
        data_framec['OT_shift']
    data_framec['Bid_Fill'] = data_framec['Bid_Diff'] > 0
    data_framec['Ask_Fill'] = data_framec['Ask_Diff'] < 0
    data_framec['BDT_And_AF'] = data_framec['Bid_Depl_Trade'] &\
        data_framec['Ask_Fill']
    data_framec['ADT_And_BF'] = data_framec['Ask_Depl_Trade'] &\
        data_framec['Bid_Fill']
    data_framec['Bid_Depl_Trade'] = data_framec['Bid_Depl_Trade'] &\
        (~ data_framec['BDT_And_AF'])
    data_framec['Ask_Fill'] = data_framec['Ask_Fill'] &\
        (~ data_framec['BDT_And_AF'])
    data_framec['Ask_Depl_Trade'] = data_framec['Ask_Depl_Trade'] &\
        (~ data_framec['ADT_And_BF'])
    data_framec['Bid_Fill'] = data_framec['Bid_Fill'] &\
        (~ data_framec['ADT_And_BF'])
    data_framec['Depl_Or_Fill'] = data_framec['Bid_Depl_Trade'] |\
        data_framec['Bid_Depl_Cancel'] | data_framec['Ask_Depl_Trade'] |\
        data_framec['Ask_Depl_Cancel'] | data_framec['Bid_Fill'] |\
        data_framec['Ask_Fill'] | data_framec['BDT_And_AF'] |\
        data_framec['ADT_And_BF']
    return data_framec

# %% Orders

def microprice(qbid, pbid, qask, pask):
    ''' fwp(qbid,pbid,qask,pask) returns the microprice
    (pbid*qask+pask*qbid)/(qbid+qask) given:
    amount on bid qbid, price on bid pbid,
    amount on ask qask, price on ask pask'''
    return (pbid*qask+pask*qbid)/(qbid+qask)

def spread_ticks(pbid, pask, tick_value):
    '''fsp(pbid,pask,tick_value) returns the spread between bid and ask prices
    in ticks given:
    price on bid pbid,
    price on ask pask,
    tick value'''
    return np.round((pask-pbid)/tick_value)

def imbalance(qbid, qask):
    '''imbalance(qbid,qask) returns the imbalance qbid/(qbid+qask)-1/2 given:
    amount on bid qbid, amount on ask qask'''
    return qbid/(qbid+qask)-1/2

def order_indicators(data_frame, tick_value):
    '''order_indicators(data_frame,tick_value) returns the midprice, the
    microprice, the level 1 and level 2 spreads in ticks, the weighted
    execution price and the associated distance of the average execution
    price on each side to the midprice, the imbalance and a discretization
    of the imbalance given:
    the data_frame
    the tick value'''
    data_framec = data_frame.copy()
    data_framec['MidPrice'] = (data_framec['Bid 1 Price']+\
        data_framec['Ask 1 Price'])/2
    data_framec['Micro_Price'] = microprice(data_framec['Bid_Qty'],\
        data_framec['Bid_Price'], data_framec['Ask_Qty'],\
        data_framec['Ask_Price'])
    data_framec['Spread_Lvl_1_Ticks'] = (data_framec['Ask 1 Price']-\
        data_framec['Bid 1 Price'])/tick_value
    data_framec['Bid_to_MidPrice'] = (data_framec['MidPrice']-\
        data_framec['Bid 1 Price'])/tick_value
    data_framec['Ask_to_MidPrice'] = (data_framec['Ask 1 Price']-\
        data_framec['MidPrice'])/tick_value
    data_framec['Spread_Lvl_2_Ticks'] = (data_framec['Ask 2 Price']-\
        data_framec['Bid 2 Price'])/tick_value
    data_framec['Bid 12 Price'] = (\
        data_framec['Bid 1 Price']*data_framec['Bid 1 Qty']+\
        data_framec['Bid 2 Price']*data_framec['Bid 2 Qty'])/\
        (data_framec['Bid 1 Qty']+data_framec['Bid 2 Qty'])
    data_framec['Bid_12_to_MidPrice'] = (data_framec['MidPrice']-\
        data_framec['Bid 12 Price'])/tick_value
    data_framec['Ask 12 Price'] = (\
        data_framec['Ask 1 Price']*data_framec['Ask 1 Qty']+\
        data_framec['Ask 2 Price']*data_framec['Ask 2 Qty'])/\
        (data_framec['Ask 1 Qty']+data_framec['Ask 2 Qty'])
    data_framec['Ask_12_to_MidPrice'] = (data_framec['Ask 12 Price']-\
        data_framec['MidPrice'])/tick_value
    data_framec['Spread_Ticks'] = spread_ticks(data_framec['Bid_Price'],\
        data_framec['Ask_Price'], tick_value)
    data_framec['Imbalance'] = imbalance(data_framec['Bid_Qty'],\
        data_framec['Ask_Qty'])
    data_framec['Imbal_Sign'] = pd.cut(data_framec['Imbalance'],\
               [-0.5, -0.2, +0.2, +0.5], labels=[-1, 0, 1])
    return data_framec

# %% States functions

def ot_states(data_frame, margins=False):
    '''ot_states(data_frame, normalize=False) returns a transition matrix
    for the states 'Aggression' (for trades) and 'Imbal_Sign' (for orders);
    'Aggression' is multiplied by 2, so -2 indicates a trade at the Bid and
    +2 indicates a trade at the Ask; 'Imbal_Sign' is -1 for a low imbalance
    value (more Qty on Ask) and 'Imbal_Sign' is -1 for a high imbalance
    value (more Qty on Bid)'''
    state_1 = pd.Series(np.where(data_frame['Aggression'] == 0,\
        data_frame['Imbal_Sign'], data_frame['Aggression']*2)).dropna()
    trans_matrix_1 = pd.crosstab(state_1, state_1.shift(),\
        rownames=['t'], colnames=['t+1'], margins=margins)
    return trans_matrix_1

def depl_fill_states(data_frame, margins=False):
    '''depl_fill_states(data_frame, normalize=False) returns a transition
    matrix for the different states of Depletion and Fill (caused by trades
    or cancels, on the Bid or on the Ask):
    ADC    = Ask_Depl_Cancel
    ADT    = Ask_Depl_Trade
    ADT+BF = ADT_And_BF
    AF     = Ask_Fill
    BDC    = Bid_Depl_Cancel
    BDT    = Bid_Depl_Trade
    BDT+AF = BDT_And_AF
    BF     = Bid_Fill'''
    trans_cols = ['Bid_Depl_Trade', 'Bid_Depl_Cancel', 'Ask_Depl_Trade',\
        'Ask_Depl_Cancel', 'Bid_Fill', 'Ask_Fill', 'BDT_And_AF', 'ADT_And_BF']
    depl_fill_df = data_frame[data_frame['Depl_Or_Fill']][trans_cols].copy()+0
    depl_fill_df.columns = ['BDT   ', 'BDC   ', 'ADT   ', 'ADC   ',\
        'BF    ', 'AF    ', 'BDT+AF', 'ADT+BF']
    depl_fill_series = pd.Series(\
        depl_fill_df.columns[np.where(depl_fill_df != 0)[1]])
    trans_matrix_2 = pd.crosstab(depl_fill_series, depl_fill_series.shift(),\
        rownames=['t'], colnames=['t+1'], margins=margins)
    return trans_matrix_2

def reduce_matrix(crostab_matrix):
    '''reduce_matrix(crostab_matrix) returns a reduced transition matrix for
    the different states of Depletion and Fill (caused by trades or cancels,
    on the same side or the opposite side for both Bid and Ask):
    DC   = Depl_Cancel
    DT   = Depl_Trade
    DT+F = DT_And_F
    F    = Fill'''
    sub_index = ['DC  ', 'DT  ', 'DT+F', 'F   ']
    depl_cancel_same = pd.Series(\
        crostab_matrix.iloc[0, :4].values+\
        crostab_matrix.iloc[4, 4:].values)
    depl_cancel_opps = pd.Series(\
        crostab_matrix.iloc[0, 4:].values+\
        crostab_matrix.iloc[4, :4].values)
    depl_cancel_row = pd.concat([depl_cancel_same, depl_cancel_opps]).values
    depl_traded_same = pd.Series(\
        crostab_matrix.iloc[1, :4].values+\
        crostab_matrix.iloc[5, 4:].values)
    depl_traded_opps = pd.Series(\
        crostab_matrix.iloc[1, 4:].values+\
        crostab_matrix.iloc[5, :4].values)
    depl_traded_row = pd.concat([depl_traded_same, depl_traded_opps]).values
    depl_trfill_same = pd.Series(\
        crostab_matrix.iloc[2, :4].values+\
        crostab_matrix.iloc[6, 4:].values)
    depl_trfill_opps = pd.Series(\
        crostab_matrix.iloc[2, 4:].values+\
        crostab_matrix.iloc[6, :4].values)
    depl_trfill_row = pd.concat([depl_trfill_same, depl_trfill_opps]).values
    fill_same = pd.Series(\
        crostab_matrix.iloc[3, :4].values+\
        crostab_matrix.iloc[7, 4:].values)
    fill_opps = pd.Series(\
        crostab_matrix.iloc[3, 4:].values+\
        crostab_matrix.iloc[7, :4].values)
    fill_row = pd.concat([fill_same, fill_opps]).values
    cols_iterables = [['same', 'opps'], sub_index]
    cols_index = pd.MultiIndex.from_product(cols_iterables,\
        names=['side', 'event'])
    sub_data_frame = pd.DataFrame({\
        'DC  ':depl_cancel_row, 'DT  ':depl_traded_row,\
        'DT+F':depl_trfill_row, 'F   ':fill_row})
    sub_data_frame.index = cols_index
    return sub_data_frame.transpose()

# %% Statistics functions

def ptox(traded_price, signs, alpha, eta):
    '''ptox(traded_price, signs, alpha, eta) returns the series of
    efficient prices given the traded prices, alpha and eta'''
    return traded_price-alpha*(0.5-eta)*(signs.fillna(0))

# %% Order statistics

def time_weighted_tob(data_frame, file_out):
    '''time_weighted_spread(data_frame) returns the time weighted spread
    in ticks, given the data_frame'''
    df_columns = ['twspr1', 'twspr2', 'bid1tomid', 'ask1tomid', 'bid1qty',\
        'ask1qty', 'bid12tomid', 'ask12tomid', 'bid12qty', 'ask12qty']
    df_dict = dict(zip(df_columns, len(df_columns)*[0]))
    data_framec = data_frame.copy()
    data_framec['dtO'] = data_framec['DateTime'].diff().shift(-1)
    data_frame_masked = data_framec[data_framec['Spread_Lvl_1_Ticks'] > 0]
    def dt0_weigh(field):
        return ((data_frame_masked[field]*data_frame_masked['dtO']).sum())/\
            (data_frame_masked['dtO'].sum())
    df_dict['twspr1'] = dt0_weigh('Spread_Lvl_1_Ticks')
    df_dict['twspr2'] = dt0_weigh('Spread_Lvl_2_Ticks')
    df_dict['bid1tomid'] = dt0_weigh('Bid_to_MidPrice')
    df_dict['ask1tomid'] = dt0_weigh('Ask_to_MidPrice')
    df_dict['bid1qty'] = dt0_weigh('Bid 1 Qty')
    df_dict['ask1qty'] = dt0_weigh('Ask 1 Qty')
    df_dict['bid12tomid'] = dt0_weigh('Bid_12_to_MidPrice')
    df_dict['ask12tomid'] = dt0_weigh('Ask_12_to_MidPrice')
    df_dict['bid12qty'] = dt0_weigh('Bid 1 Qty')+dt0_weigh('Bid 2 Qty')
    df_dict['ask12qty'] = dt0_weigh('Ask 1 Qty')+dt0_weigh('Ask 2 Qty')
    data_frame_stats = pd.DataFrame(df_dict, index=[file_out])
    return data_frame_stats

# %% Cost of Trades

def cost_of_trades(data_frame):
    '''cost_of_trades(data_frame) returns a data frame
    with trade amounts and costs'''
    data_framec = data_frame.copy()
    data_framec['MidPrice_Fill'] = data_framec['MidPrice'].copy()\
        .fillna(method='ffill')
    data_framec_trades = data_framec[~data_framec['OT']].copy()
    data_framec_trades['Cost'] = data_framec_trades['Aggression']*\
        data_framec_trades['Trade Qty']*\
        (data_framec_trades['Trade Price']-data_framec_trades['MidPrice_Fill'])
    data_framec_trades.set_index(['DateTime', 'Aggression', 'MidPrice_Fill'],\
        inplace=True)
    cost_data = data_framec_trades[['Trade Qty', 'Cost']].sum(level=[0, 1])\
        .reset_index()
    cost_data['Avg_Cost'] = cost_data['Cost']/cost_data['Trade Qty']
    return cost_data

def mean_cost_of_trades(data_frame):
    '''mean cost_of_trades(data_frame) returns a data frame
    with the mean cost of trade for each trade amount'''
    data_framec = data_frame.copy()
    cost_series = data_framec.groupby('Trade Qty')['Avg_Cost'].mean()
    return pd.DataFrame(cost_series).reset_index()

# %% Main script

def run_unc_zones(pathin, pathout, file_name, tick_value, start_time,\
                  end_time, trading_hours, save_files=False):
    '''run_unc_zones(pathin, pathout, file_name, tick_value, end_of_time)
    returns the uncertainty zones data frame'''
    print('Reading file '+pathin+file_name)
    data_frame = pd.read_csv(pathin+file_name)
    file_out = file_name[:-4]
    print('Preparing data_frame')
    data_frame = column_datetime(column_ot(data_frame))
    data_frame = select_times(data_frame, start_time, end_time)
    print('Running Uncertainty Zones statistics')
    df_trades = data_frame[~data_frame['OT']].copy()
    df_trades_time = collapse_time(df_trades)
    df_trades_price = adduz(collapse_price(df_trades_time), tick_value)
    df_cont_alt_by_ticks = uz_coal_byk(df_trades_price)
    df_uz_stats = uz_stats(df_trades_price, file_out, tick_value,\
                           trading_hours)
    print('Filling intermediate values in order book')
    df_agg = fill_aggression_id(data_frame)
    max_runs = find_max_run(df_agg['OT'])
    print(file_out+' max_runs = '+str(max_runs))
    df_tob = loop_previous_tob(df_agg, max_runs)
    print('Calculating order indicators')
    df_depl = depletions(df_tob, tick_value)
    df_imbl = order_indicators(df_depl, tick_value)
    df_orders = df_imbl[df_imbl['OT']].copy()
    df_time_weighted_tob = time_weighted_tob(df_orders, file_out)
    df_ot_trans = ot_states(df_imbl)
    df_red_deplfill_trans = reduce_matrix(depl_fill_states(df_imbl))
    df_cost_of_trades = mean_cost_of_trades(cost_of_trades(df_imbl))
    if save_files:
        print('Saving files for '+file_name)
        df_cont_alt_by_ticks.to_hdf(pathout+file_out+'_CAticks.h5',\
                                    key=file_out, mode='w')
        df_uz_stats.to_hdf(pathout+file_out+'_UZstats.h5',\
                           key=file_out, mode='w')
        df_time_weighted_tob.to_hdf(pathout+file_out+'_OBstats.h5',\
                           key=file_out, mode='w')
        df_ot_trans.to_hdf(pathout+file_out+'_OTtrans.h5',\
                           key=file_out, mode='w')
        df_red_deplfill_trans.to_hdf(pathout+file_out+'_RDFtrans.h5',\
                           key=file_out, mode='w')
        df_cost_of_trades.to_hdf(pathout+file_out+'_COSTtrades.h5',\
                           key=file_out, mode='w')
#        df_imbl.to_hdf(pathout+file_out+'_data.h5',\
#                      key=file_out, mode='w', format='table')
    print('Finished '+file_name)
    # no need to return data_frame

# %% StatsUZ

#run_unc_zones(PATHIN, PATHOUT, FILE1, TS, START_TIME, END_TIME,\
#             TRADING_HOURS, True)

# %% StatsUZ

#run_unc_zones(PATHIN, PATHOUT, FILE2, TS, START_TIME, END_TIME,\
#             TRADING_HOURS, True)

# %% k DF

#print('Continuations and Alternations: DOL')
#print('')
#print(pd.read_hdf(PATHOUT+FILE1[:-4]+'_CAticks.h5'))
#print('')

# %% k DF

#print('Continuations and Alternations: WDO')
#print('')
#print(pd.read_hdf(PATHOUT+FILE2[:-4]+'_CAticks.h5'))
#print('')

# %% Eta and vols

#print('Uncertainty zones statistics: DOL')
#print('')
#print(pd.read_hdf(PATHOUT+FILE1[:-4]+'_UZstats.h5'))
#print('')

# %% Eta and vols

#print('Uncertainty zones statistics: WDO')
#print('')
#print(pd.read_hdf(PATHOUT+FILE2[:-4]+'_UZstats.h5'))
#print('')

# %% TOB

#print('Top of Book statistics: DOL')
#print('')
#print(pd.read_hdf(PATHOUT+FILE1[:-4]+'_OBstats.h5'))
#print('')

# %% TOB

#print('Top of Book statistics: WDO')
#print('')
#print(pd.read_hdf(PATHOUT+FILE2[:-4]+'_OBstats.h5'))
#print('')

# %% States - Orders and Trades

#print('OT states DOL')
#print('')
#print(pd.read_hdf(PATHOUT+FILE1[:-4]+'_OTtrans.h5'))
#print('')

# %% States - Orders and Trades

#print('OT states WDO')
#print('')
#print(pd.read_hdf(PATHOUT+FILE2[:-4]+'_OTtrans.h5'))
#print('')

# %% States - Depletions and Fills

#print('DeplFill states DOL')
#print('')
#print(pd.read_hdf(PATHOUT+FILE1[:-4]+'_RDFtrans.h5'))
#print('')

# %% States - Depletions and Fills

#print('DeplFill states WDO')
#print('')
#print(pd.read_hdf(PATHOUT+FILE2[:-4]+'_RDFtrans.h5'))
#print('')