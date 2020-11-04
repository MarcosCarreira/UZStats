
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

# %% Constants - CME


TS = 0.5
MOSCME = 1
MINDTCME = 0.0000001
MAXEVDTCME = 10
DTEVSHIFTCME = -pd.Timedelta(MINDTCME, 's')*0
DTCUMADDCME = pd.Timedelta(MINDTCME, 's')/MAXEVDTCME
START_TIME = pd.to_timedelta('00:00:00')
END_TIME = pd.to_timedelta('23:59:59')

# %% Constants - BMF


TS1 = 0.5
MOSDOL = 5
MOSWDO = 1
MINDT1 = 0.001
MAXEVDT1 = 40
DTEVSHIFT1 = -pd.Timedelta(MINDT1, 's')/2
DTCUMADD1 = pd.Timedelta(MINDT1, 's')/MAXEVDT1
START_TIME1 = pd.to_timedelta('09:00:00')
END_TIME1 = pd.to_timedelta('18:15:00')

# %% Constants - tick

EV_14_LBLS = ['L_B', 'C_A', 'M_A', 'I_B', 'DmI_A', 'Dm_A', 'Dc_A',
              'L_A', 'C_B', 'M_B', 'I_A', 'DmI_B', 'Dm_B', 'Dc_B']

# %% Armada Data Class


class Armada_Data():

# %% init

    def __init__(self, file_path, file_name, exchange='CME'):
        self.__exchange = exchange
        self.__file_name = file_name
        self.__processing_date = self.get_processing_date()
        self.__file_path = file_path
        self.__df = pd.DataFrame()
        # self.__measures = pd.DataFrame()

        print('Reading file ' + self.file_entire_path)
        self.__read_ArmadaData()
        print('Re-formatting data')
        self.__column_datetime()
        self.__rename_columns()
        print('Remove data outside exchange trading hours')
        self.__filter_exchange_data_by_time()
        self.__filter_exchange_data_prior_first_trade()
        print('Armada Data Object Construction Successful')

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

# %% Public Functions

    def get_processing_date(self):
        if self.__exchange == 'CME':
            return pd.to_datetime(self.file_name[0:8], format='%Y%m%d')
        if self.__exchange == 'BMF':
            return pd.to_datetime(self.file_name[6:-4], format='%Y%m%d')

    def get_exchange_starting_time(self):
        if self.__exchange == 'CME':
            return self.processing_date\
                                + pd.to_timedelta('00:00:00')
        if self.__exchange == 'BMF':
            return self.processing_date\
                                + pd.to_timedelta('09:00:00')

    def get_exchange_end_time(self):
        if self.__exchange == 'CME':
            return self.processing_date + pd.to_timedelta('16:00:00')
        if self.__exchange == 'BMF':
            return self.processing_date + pd.to_timedelta('18:15:00')

# %% Main Functions

    def select_times(self, start_time=pd.to_timedelta('00:30:00'), \
                     end_time=pd.to_timedelta('23:59:59')):
        self.__filter_exchange_data_by_time([start_time, end_time])
        self.__filter_exchange_data_prior_first_trade()
        return self

    def __read_ArmadaData(self):
        self.__df = pd.read_csv(
            self.file_entire_path,
            dtype={'Index': int, 'Date': str, 'Time': str,
                   'Bid 1 Qty': float, 'Bid 2 Qty': float,
                   'Bid 1 Price': float, 'Bid 2 Price': float,
                   'Bid 1 Ord': float, 'Bid 2 Ord': float,
                   'Trade Price': float, 'Trade Qty': float,
                   'Ask 1 Qty': float, 'Ask 2 Qty': float,
                   'Ask 1 Price': float, 'Ask 2 Price': float,
                   'Ask 1 Ord': float, 'Ask 2 Ord': float})

    def __column_datetime(self):
        '''column_datetime(data_frame) creates the column 'DateTime'
        concatenating 'Date' and 'Time' and converting it to datetime.
        Inputs: Data Frame from Armada
        Outputs: Copy of DataFrame with 'DateTime' column added'''
        data_framec = self.__df.copy()
        data_framec['Date'] = pd.to_datetime(data_framec['Date'],\
            format="%m/%d/%Y")
        data_framec['Time'] = pd.to_timedelta(data_framec['Time']) #  , unit='ns')
        data_framec['DateTime'] = data_framec['Date'] + data_framec['Time']
        self.__df = data_framec

    def __rename_columns(self):
        self.__df.rename(
            columns={
                'Bid 1 Qty': 'bid_1_qty', 'Bid 2 Qty': 'bid_2_qty',
                'Bid 1 Price': 'bid_1_price', 'Bid 2 Price': 'bid_2_price',
                'Bid 1 Ord': 'bid_1_ord', 'Ask 1 Ord': 'ask_1_ord',
                'Bid 2 Ord': 'bid_2_ord', 'Ask 2 Ord': 'ask_2_ord',
                'Ask 1 Qty': 'ask_1_qty', 'Ask 2 Qty': 'ask_2_qty',
                'Ask 1 Price': 'ask_1_price', 'Ask 2 Price': 'ask_2_price',
                'Trade Price': 'trade_price', 'Trade Qty': 'trade_qty'},
            inplace=True)
        self.__df.drop(
            ['Time.1', 'Date.1', 'Index', 'Index.1', 'Date', 'Time'],
            axis=1, inplace=True)

    def __filter_exchange_data_prior_first_trade(self):
        first_trade = self.df[self.get_trade_indicator()].index[0]
        df_tmp = self.__df.copy()
        # df_tmp = df_tmp.set_index(['DateTime'])
        df_tmp = df_tmp.loc[first_trade:]
        # df_tmp = df_tmp.reset_index()
        self.__df = df_tmp

    def __filter_exchange_data_by_time(self, arg=None):
        if arg is None: # is in place of ==
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
        pxs = np.log(prices / prices.shift(1))
        self.measures['realized_vol']=np.sqrt(np.sum(pxs * pxs))
        
    def __custom_resampler_abs_sum(self,array):
        return np.sum(np.abs(array))
    
    def __get_time_weighted_spread(self, data=None):
        if data is not None:    
            return ((data['ba_spread'] * data['dt']).sum()) /\
                (data['dt'].sum())
            return ((self.__df['ba_spread'] * self.__df['dt']).sum()) /\
                (self.__df['dt'].sum())

# %%  Plots Functions

    def plot_html_1mintick(self,path_out,
                           start_xaxis=pd.to_timedelta('07:30:00')):
        # layout = self.__plot_html_layout('Bid Price', 'Ask Price',
        #                                  'Price Time Series')
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
                                 name='Bid'), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data.ask_1_price,
                                 name='Ask'), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data.trade_qty,
                                 name='Trade Qty'), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=self.__get_ba_spread(data),
                                 name='Bid-ask spread'), row=3, col=1)
        file = path_out + self.file_name_long + "_price.html"
        print('saving html plot to ', file)
        fig.write_html(file)

    def plot_html_qty(self,path_out):
        layout = self.__plot_html_layout('Bid Qty', 'Ask Qty',
                                         'Quantity Time Series')
        bid_data = go.Scatter(x=self.__df.DateTime,
                         y=self.__df.bid_1_qty)
        ask_data = go.Scatter(x=self.__df.DateTime,
                        y=self.__df.ask_1_qty)
        fig = go.Figure(data=[bid_data, ask_data], layout=layout)
        fig.update_layout(height=600, width=600,
                          title_text="Stacked Subplots")
        file = path_out + self.file_name_long + "_qty.html"
        print('saving html plot to ', file)
        fig.write_html(file)

    def plot_html_ohlc(self, pathout, freq='1min',
                       start_xaxis=pd.to_timedelta('09:00:00'),
                       end_xaxis=pd.to_timedelta('18:15:00')):
        data = self.__df.copy()
        data = data.set_index(['DateTime'])
        start_xaxis = self.get_processing_date() + start_xaxis
        end_xaxis = self.get_processing_date() + end_xaxis
        # prepare OHLC data
        data['mid_price'] = (data.ask_1_price + data.bid_1_price) / 2
        ohlc = data.mid_price.resample(freq).ohlc()
        ohlc = ohlc.loc[start_xaxis:end_xaxis]
        # prepare Volume data
        volume = pd.DataFrame(data.trade_qty.resample(freq)
                              .apply(self.__custom_resampler_abs_sum))
        # volume = volume.set_index(ohlc.index)    
        volume = volume.loc[start_xaxis:end_xaxis]
        # prepare time weigthed spread data
        data = data.assign(ba_spread=self.measures.ba_spread.values)
        data = data.assign(dt=self.measures.dt .values)
        data_masked = data[data.ba_spread > 0]
        time_weighted_spread = pd.DataFrame(
            data_masked.groupby(
                pd.Grouper(freq=freq)).apply(self.__get_time_weighted_spread))
        time_weighted_spread.columns = ['time_weighted_spread']
        time_weighted_spread = time_weighted_spread.loc[start_xaxis:end_xaxis]
        # plotting
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
        row_width=[0.2, 0.4],vertical_spacing=0.1,
        subplot_titles=('Open High Low Close Candlestick', 'Volume'))
        fig.add_trace(
            go.Candlestick(
                name='OHLC', x=ohlc.index, open=ohlc.open, high=ohlc.high,
                low=ohlc.low, close=ohlc.close,
                increasing_line_color= 'green', decreasing_line_color= 'red'),
            row=1, col=1)
        fig.add_trace(go.Bar(name='Volume', x=volume.index,
                             y=volume.trade_qty), row=2, col=1)
        #fig.add_trace(go.Bar(x=time_weighted_spread.index, \
        #        y=time_weighted_spread.time_weighted_spread , 
        #        name = 'Time Weighted Spread', marker_color='black'), 
        #        row=3,col=1)
        fig.update_layout(
                xaxis=dict(rangeslider=dict(visible=False),type="date"))
        file = pathout+self.get_processing_date().strftime('%Y%m%d') + '_'\
            + freq+"_ohlc.html"
        print('saving html plot to ', file)
        fig.write_html(file)

# %% Armada Level 1 clean and collapsed class

class Armada_Lvl1(Armada_Data):
    def __init__(self, Armada_Data, start_time, end_time, file_type='CME',
                 min_dt=MINDTCME):
        start = timeit.default_timer()
        self.__Armada_Data = Armada_Data
        self.__exchange = Armada_Data.exchange
        self.__file_name = Armada_Data.file_name
        self.__file_name_long = Armada_Data.file_name_long
        self.__file_entire_path = Armada_Data.file_entire_path
        self.__processing_date = Armada_Data.processing_date
        # Select times
        self.__df = self.__Armada_Data.select_times(
            pd.to_timedelta(start_time), pd.to_timedelta(end_time)).df
        # Add order flags and count
        self.__df['OrderQ'] = self.__df['trade_price'].isnull().copy()
        self.__df['OrderN'] = self.__df['OrderQ'].copy().cumsum()
        self.__df['last_trade'] = self.__df['trade_price'].copy().fillna(
            method='ffill')
        # Group trades by time and price
        self.__df = self.__df.groupby(
            ['OrderN', 'DateTime', 'OrderQ', 'last_trade'], sort=False
            ).sum(min_count=1)
        # Reset columns from groupby
        self.__df = self.__df.reset_index()
        self.__df['trade_price'] = np.where(self.__df['OrderQ'], np.nan,
                                          self.__df['last_trade'].copy())
        self.__df = self.__df.drop(columns=['OrderN', 'last_trade'])
        # Clear trades without book update or sweep not instantaneous
        def clear_invalid_trades(df, dt=min_dt):
            dfc = df.copy()
            dfc['Prev_Trade'] = (~dfc['OrderQ']).shift()
            dfc['Signif_dt'] = dfc['DateTime'].diff().dt.total_seconds() > dt
            dfc['Check'] = (dfc['Prev_Trade'] & (dfc['Signif_dt'])).shift(
                periods=-1, fill_value=False)
            return dfc[~dfc['Check']].copy()\
                .drop(['Prev_Trade', 'Signif_dt', 'Check'], axis=1)
        # Clear trades without book update or sweep not instantaneous
        cleaned_df = clear_invalid_trades(self.__df)
        print(str(len(self.__df) - len(cleaned_df)) + ' trades deleted')
        while len(cleaned_df) != len(self.__df):
            self.__df = cleaned_df
            cleaned_df = clear_invalid_trades(self.__df)
            print(str(len(self.__df) - len(cleaned_df)) + ' trades deleted')
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
        self.__df = lvldiff(self.__df)
        self.__df = self.__df[self.__df['lvl2']]
        self.__df = self.__df.drop(['bid_2_qty', 'bid_2_ord', 'bid_2_price',
                                      'bid_1_ord', 'ask_1_ord', 'ask_2_price',
                                      'ask_2_ord', 'ask_2_qty', 'lvl2'],
                                     axis=1)
        self.__df['bid_traded'] = self.__df['bid_1_price'].copy().fillna(
            method='ffill') >= self.__df['trade_price']
        self.__df['ask_traded'] = self.__df['ask_1_price'].copy().fillna(
            method='ffill') <= self.__df['trade_price']
        # Clear trades without book update or sweep not instantaneous
        cleaned_df = clear_invalid_trades(self.__df)
        while len(cleaned_df) != len(self.__df):
            self.__df = cleaned_df
            cleaned_df = clear_invalid_trades(self.__df)
            print(str(len(self.__df) - len(cleaned_df)) + ' trades deleted')
        print('Armada_Lvl1 finished')
        stop = timeit.default_timer()
        print('Time spent on Armada Data cleaning: ',
              round(stop - start), ' seconds')

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
        return (self.__file_entire_path)

    @property
    def exchange(self):
        return self.__exchange

    @property
    def df(self):
        return self.__df
        
# %% Public Functions

    def get_processing_date(self):
        if self.__exchange == 'CME':
            return pd.to_datetime(self.file_name[0:8], format='%Y%m%d')
        if self.__exchange == 'BMF':
            return pd.to_datetime(self.file_name[6:-4], format='%Y%m%d')

    def get_exchange_starting_time(self):
        if self.__exchange == 'CME':
            return self.processing_date\
                                + pd.to_timedelta('00:00:00')
        if self.__exchange == 'BMF':
            return self.processing_date\
                                + pd.to_timedelta('09:00:00')

    def get_exchange_end_time(self):
        if self.__exchange == 'CME':
            return self.processing_date + pd.to_timedelta('16:00:00')
        if self.__exchange == 'BMF':
            return self.processing_date + pd.to_timedelta('18:15:00')

# %% Armada Level 1 fully collapsed class (trade levels)

class Armada_Collapsed(Armada_Lvl1):
    def __init__(self, Armada_Lvl1, tick_value, min_order_size):
        start = timeit.default_timer()
        self.__Armada_Lvl1 = Armada_Lvl1
        self.__exchange = Armada_Lvl1.exchange
        self.__file_name = Armada_Lvl1.file_name
        self.__file_name_long = Armada_Lvl1.file_name_long
        self.__file_entire_path = Armada_Lvl1.file_entire_path
        self.__processing_date = Armada_Lvl1.processing_date
        # Define key for groupby
        self.__df = self.__Armada_Lvl1.df.copy()
        self.__df.loc[:, 'OrderN'] = self.__df['OrderQ'].cumsum()
        self.__df = self.__df[self.__df['OrderN'] > 0].copy()
        self.__df.loc[:, 'OrderN'] = self.__df['OrderN'] * (1 - 2 *
                                                     self.__df['OrderQ'])
        # Group trades (sum qty, count of price levels traded)
        dfg = self.__df.copy().groupby(['DateTime', 'OrderN'], sort=False)
        datadfg = dfg.agg(
            {'OrderQ': all, 'bid_1_qty': sum, 'bid_1_price': sum,
             'trade_price': 'count', 'trade_qty': sum, 'ask_1_price': sum,
             'ask_1_qty': sum, 'lvl1': any, 'bid_traded': any,
             'ask_traded': any}).copy()
        datadfg = datadfg.reset_index()
        datadfg = datadfg.rename(columns={
            'trade_price': 'levels_traded', 'lvl1': 'Level1Q'})
        # Push trades on next order book state
        print('Pushing trades on next order book state')
        datadfg.loc[:, 'OrderId'] = np.abs(datadfg['OrderN']) +\
            (1 + np.sign(datadfg['OrderN']))/2
        dfagg2 = datadfg.groupby(['OrderId'])
        self.__dfg2 = dfagg2.agg({'DateTime': 'first', 'bid_1_qty': sum,
                               'bid_1_price': sum, 'levels_traded': sum,
                               'trade_qty': sum, 'ask_1_price': sum,
                               'ask_1_qty': sum, 'Level1Q': any,
                               'OrderQ': any, 'bid_traded': any,
                               'ask_traded': any})
        self.__dfg2 = self.__dfg2.reset_index()
        # Normalize amount by the minimum order size (MOS)
        self.__dfg2['bid_1_qty'] = self.__dfg2['bid_1_qty']/min_order_size
        self.__dfg2['ask_1_qty'] = self.__dfg2['ask_1_qty']/min_order_size
        self.__dfg2['trade_qty'] = self.__dfg2['trade_qty']/min_order_size
        # Spread, Midprice , Microprice and Imbalance
        self.__dfg2['Spread_Ticks'] = (self.__dfg2['ask_1_price'] -
                                    self.__dfg2['bid_1_price']) / tick_value
        self.__dfg2['Midprice'] = (
            self.__dfg2['ask_1_price'] + self.__dfg2['bid_1_price'])/2
        self.__dfg2['Microprice'] = (
            self.__dfg2['ask_1_price'] * self.__dfg2['bid_1_qty'] +
            self.__dfg2['bid_1_price'] * self.__dfg2['ask_1_qty']) / \
            (self.__dfg2['bid_1_qty'] + self.__dfg2['ask_1_qty'])
        self.__dfg2['Imbalance'] =\
            self.__dfg2['bid_1_qty'] / \
                (self.__dfg2['bid_1_qty'] + self.__dfg2['ask_1_qty']) - 1/2
        self.__dfg2['Imbal_Sign'] = pd.cut(self.__dfg2['Imbalance'],
                                        [-0.5, -0.2, +0.2, +0.5],
                                        labels=[-1, 0, 1])
        print('Armada_Collapsed finished')
        stop = timeit.default_timer()
        print('Time spent on Armada Data collapsing: ',
              round(stop - start), ' seconds')

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
        return (self.__file_entire_path)

    @property
    def exchange(self):
        return self.__exchange

    @property
    def df(self):
        return self.__dfg2
        
# %% Public Functions

    def get_processing_date(self):
        if self.__exchange == 'CME':
            return pd.to_datetime(self.file_name[0:8], format='%Y%m%d')
        if self.__exchange == 'BMF':
            return pd.to_datetime(self.file_name[6:-4], format='%Y%m%d')

    def get_exchange_starting_time(self):
        if self.__exchange == 'CME':
            return self.processing_date\
                                + pd.to_timedelta('00:00:00')
        if self.__exchange == 'BMF':
            return self.processing_date\
                                + pd.to_timedelta('09:00:00')

    def get_exchange_end_time(self):
        if self.__exchange == 'CME':
            return self.processing_date + pd.to_timedelta('16:00:00')
        if self.__exchange == 'BMF':
            return self.processing_date + pd.to_timedelta('18:15:00')


# %% Armada Level 1 collapsed class prepared for Hawkes

class Armada_Hawkes(Armada_Collapsed):
    def __init__(self, Armada_Collapsed, dt_shift, dt_cum_shift):
        start = timeit.default_timer()
        self.__Armada_Collapsed = Armada_Collapsed
        self.__exchange = Armada_Collapsed.exchange
        self.__file_name = Armada_Collapsed.file_name
        self.__file_name_long = Armada_Collapsed.file_name_long
        self.__file_entire_path = Armada_Collapsed.file_entire_path
        self.__processing_date = Armada_Collapsed.processing_date
        # Changes in top of book (diff)
        self.__df = self.__Armada_Collapsed.df.copy()
        self.__df['bid_1_qty_diff'] = self.__df['bid_1_qty'].diff()
        self.__df['bid_1_price_diff'] = self.__df['bid_1_price'].diff()
        self.__df['ask_1_price_diff'] = self.__df['ask_1_price'].diff()
        self.__df['ask_1_qty_diff'] = self.__df['ask_1_qty'].diff()
        # PriceQ column (was there a price change?)
        print('Calculating PriceQ column')
        self.__df['PriceQ'] = (self.__df['bid_1_price_diff'] != 0) |\
            (self.__df['ask_1_price_diff'] != 0)
        # ConsQ column (was there a comsumption of liquidity?)
        # Trades that take out levels but leave an unfilled balance: False
        print('Calculating ConsQ column')
        self.__df['ConsQ'] = np.where(
            self.__df['PriceQ'], ~(
                (self.__df['bid_1_price_diff'] > 0) |
                (self.__df['ask_1_price_diff'] < 0)),
            (self.__df['bid_1_qty_diff'] < 0) |
            (self.__df['ask_1_qty_diff'] < 0) |
            (~self.__df['Level1Q']))
        # AskQ column (was the event on the Ask side?)
        # Trades that take out levels but leave an unfilled balance: Cons sign
        print('Calculating AskQ column')
        self.__df['AskQ'] = np.where(
            self.__df['Level1Q'], ((self.__df['ask_1_price_diff'] != 0) |
                                   (self.__df['ask_1_qty_diff'] != 0)),
            self.__df['ask_traded'])
        self.__df.at[0, 'PriceQ'] = False
        self.__df.at[0, 'ConsQ'] = False
        self.__df.at[0, 'AskQ'] = False
        # Calculate event size
        print('Calculating Event Size')
        self.__df['Event_Size_order'] = np.where(
            self.__df['AskQ'],
            np.where(self.__df['ask_1_price_diff'] != 0,
                     self.__df['ask_1_qty'],
                     np.abs(self.__df['ask_1_qty_diff'])),
            np.where(self.__df['bid_1_price_diff'] != 0,
                     self.__df['bid_1_qty'],
                     np.abs(self.__df['bid_1_qty_diff'])))
        self.__df['Event_Size'] = np.where(
            self.__df['trade_qty'] > 0, self.__df['trade_qty'],
            self.__df['Event_Size_order'])
        self.__df['Event_Size'] = self.__df['Event_Size'].fillna(1)
        # Classify event
        print('Classifying events')
        self.__df['event_code'] =\
            self.__df['AskQ'] * 8 + self.__df['ConsQ'] * 4 +\
            self.__df['Level1Q'] * 2 + self.__df['PriceQ'] * 1
        event_dict = {
            0: 'Start', 1: 'PLb', 2: 'Lb', 3: 'Pb+', 4: 'Mb', 5: 'PbM-',
            6: 'Cb', 7: 'PbC-', 8: 'Start', 9: 'PLa', 10: 'La', 11: 'Pa-',
            12: 'Ma', 13: 'PaM+', 14: 'Ca', 15: 'PaC+'}
        self.__df['Event_detail'] = self.__df['event_code'].map(event_dict)
        self.__df['Event_detail_Prev'] = self.__df['Event_detail'].copy()\
            .shift().fillna('La')
        event_dict_14 = {
            0: 'L_B', 1: 'DmI_B', 2: 'L_B', 3: 'I_B', 4: 'M_B', 5: 'Dm_B',
            6: 'C_B', 7: 'Dc_B', 8: 'L_A', 9: 'DmI_A', 10: 'L_A', 11: 'I_A',
            12: 'M_A', 13: 'Dm_A', 14: 'C_A', 15: 'Dc_A'}
        self.__df['Event'] = self.__df['event_code'].map(event_dict_14)
        event_dict_consec = {
            'Ca': False, 'Cb': True, 'La': True, 'Lb': False, 'Ma': False,
            'Mb': True, 'PLa': False, 'PLb': True, 'Pa-': True, 'PaC+': False,
            'PaM+': False, 'Pb+': False, 'PbC-': True, 'PbM-': True}
        print('Calculating Reversion and Hawkes Timestamp')
        self.__df['Reversion'] =\
            self.__df['Event_detail'].map(event_dict_consec) ^\
                self.__df['Event_detail_Prev'].map(event_dict_consec)
        self.__df['dt0'] = \
            self.__df['DateTime'] == self.__df['DateTime'].shift()
        self.__df['ConsTS'] =\
            self.__df.groupby('DateTime')['dt0'].transform(pd.Series.cumsum)
        self.__df['TS_Hawkes'] = self.__df['DateTime'] + dt_shift +\
            dt_cum_shift * self.__df['ConsTS']
        self.__df['dt'] = self.__df['TS_Hawkes'].diff().dt.total_seconds()
        print(self.__df['dt'].value_counts().sort_index().head())
        cols_output1 =\
            ['DateTime', 'OrderId', 'bid_1_qty', 'bid_1_price', 'ask_1_price',
             'ask_1_qty', 'trade_qty', 'levels_traded', 'Event_Size',
             'AskQ', 'ConsQ', 'Level1Q', 'PriceQ', 'Event_detail', 'Event',
             'Reversion', 'TS_Hawkes', 'dt', 'Spread_Ticks', 'Midprice',
             'Microprice', 'Imbalance', 'Imbal_Sign']
        self.__df = self.__df[cols_output1]
        print('Armada_Hawkes finished')
        stop = timeit.default_timer()
        print('Time spent on Armada Data Hawkes preparation: ',
              round(stop - start), ' seconds')

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
        return (self.__file_entire_path)

    @property
    def exchange(self):
        return self.__exchange

    @property
    def df(self):
        return self.__df
        
# %% Public Functions

    def get_processing_date(self):
        if self.__exchange == 'CME':
            return pd.to_datetime(self.file_name[0:8], format='%Y%m%d')
        if self.__exchange == 'BMF':
            return pd.to_datetime(self.file_name[6:-4], format='%Y%m%d')

    def get_exchange_starting_time(self):
        if self.__exchange == 'CME':
            return self.processing_date\
                                + pd.to_timedelta('00:00:00')
        if self.__exchange == 'BMF':
            return self.processing_date\
                                + pd.to_timedelta('09:00:00')

    def get_exchange_end_time(self):
        if self.__exchange == 'CME':
            return self.processing_date + pd.to_timedelta('16:00:00')
        if self.__exchange == 'BMF':
            return self.processing_date + pd.to_timedelta('18:15:00')

    def event_size_pivot(self):
        def q10(array):
            return np.quantile(array, 0.1)
        def q30(array):
            return np.quantile(array, 0.3)
        def q70(array):
            return np.quantile(array, 0.7)
        def q90(array):
            return np.quantile(array, 0.9)
        return pd.pivot_table(self.df, 'Event_Size', index='Event',
                          aggfunc=[np.mean, q10, q30, np.median, q70, q90])
    
    def describe_DmI(self):
        mask_A = self.df['Event'] == 'DmI_A'
        piv_A = self.df[mask_A][['bid_1_qty', 'ask_1_qty', 'trade_qty']].describe()
        piv_A.index.rename('DmI_A', inplace=True)
        mask_B = self.df['Event'] == 'DmI_B'
        piv_B = self.df[mask_B][['bid_1_qty', 'ask_1_qty', 'trade_qty']].describe()
        piv_B.index.rename('DmI_B', inplace=True)
        return [piv_B, piv_A]

    def get_event_timestamps(self):
        data_framec = self.df.copy().iloc[1:][['TS_Hawkes', 'Event']]
        times = data_framec['TS_Hawkes'].copy()
        start = times.iloc[0]
        data_framec['Timestamp'] = (times - start).dt.total_seconds().values
        df_dummies = pd.get_dummies(
            data_framec.set_index('Timestamp')['Event'])
        df_dummies = df_dummies[EV_14_LBLS]
        labels = df_dummies.columns.values
        def get_timestamps_from_dummies(data_frame, col):
            data_framec = data_frame.copy()
            data_framec = data_framec[data_framec[col] == 1].copy()
            return data_framec.index.values
        list_values = [get_timestamps_from_dummies(df_dummies, col)
                    for col in labels]
        return [list_values, labels]

    def get_trading_window(self):
        times = self.df['DateTime'].copy()
        start = times.iloc[0]
        end = times.iloc[-1]
        return (end - start).total_seconds()

    def get_event_counts(self):
        return self.df['Event'].value_counts()[EV_14_LBLS]

    def pivot_events(self, piv_values='dt', aggfunc=np.mean, margins=False):
        # Options for values: 'dt'
        dfc = self.df[['Event', piv_values]].copy()
        dfc['Previous_Event'] = dfc['Event'].shift(+1).values
        return pd.pivot_table(dfc, values=piv_values, columns=['Previous_Event'],
                              index='Event', aggfunc=aggfunc, margins=margins)\
            [EV_14_LBLS].reindex(EV_14_LBLS)

    def pivot_prev_events(self, piv_values='Imbalance',
                          aggfunc=np.mean, margins=False):
        # Options for values: 'Imbalance', 'Event_Size', 'Spread_Ticks'
        dfc = self.df[['Event', piv_values]].copy()
        dfc['Previous_Event'] = dfc['Event'].shift(+1).values
        dfc['Previous_Values'] = dfc[piv_values].shift(+1).values
        return pd.pivot_table(dfc, values='Previous_Values',
                              columns=['Previous_Event'], index='Event',
                              aggfunc=aggfunc, margins=margins)\
            [EV_14_LBLS].reindex(EV_14_LBLS)

# %% Armada UZ Model Output Class
    
class Armada_UZModel_output:
    df_cont_alt_by_ticks = pd.DataFrame()
    df_uz_stats = pd.DataFrame()
    
    def __init__(self, df_cont_alt_by_ticks=pd.DataFrame(),
                 df_uz_stats= pd.DataFrame()):
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
        
        fig.add_trace(
            go.Scatter(x=self.df_uz_stats.Date,
                       y=self.df_uz_stats.rvp, mode='markers',
                       name='Realized Volatility - log price'), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=self.df_uz_stats.Date,
                       y=self.df_uz_stats.dt_avg, mode='markers',
                       name='Average Duration'), row=2, col=1)
        fig.add_trace(
            go.Scatter(x=self.df_uz_stats.Date,
                       y=self.df_uz_stats.chgavg, mode='markers',
                       name= 'Average Price Move'), row=3, col=1)
        fig.add_trace(
            go.Scatter(x=self.df_uz_stats.Date,
                       y=self.df_uz_stats.ndfpr, mode='markers',
                       name='Number of Price Change'), row=4, col=1)

        
        file = path_out+"uz_stats.html"
        print('saving html plot to ', file)
        fig.write_html(file)
        
        
    
# %% Armada UZ Model Class

class ArmadaData_UZModel():
    df = pd.DataFrame()
    df_trades = pd.DataFrame()
    df_trades_by_time = pd.DataFrame()
    df_trades_by_price = pd.DataFrame()
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
        self.df_trades = self.__get_trades()
        self.volume = float(self.df_trades.trade_qty.sum())
        self.__collapse_time()
        self.n_trades = float(len(self.df_trades_by_time))
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
                   self.df_trades.trade_price.shift(1))
        diff_no0 = diff[diff!=0]
        self.__trades_min_increment = np.nanmin(diff_no0)

    def __get_trades(self):
        '''function returns a trades matrix'''
        df_trades = self.df[~self.df['OT']].copy()
        return df_trades
        
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
        self.df_trades_by_time = df_grouped_at_first
        
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
        df_unique = data_frame.drop_duplicates(
            subset = ['DateTime', 'trade_price', 'trade_qty'])
        df_grouped = data_frame[['DateTime','trade_price']].groupby(
            'trade_qty').sum()
        
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
        data_frame = self.df_trades_by_time
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
        self.df_trades_by_price = df_grouped_at_first

    # %% Trades UZ fields
    
    def __adduz(self):
        '''adduz(data_frame,alpha) returns a data frame with
        the fields necessary for the UZ stats
        Inputs: Data Frame of trades collapsed by collapse_time and by
        collapse_price
        Outputs: Collapsed Data Frame with additional fields for the UZ model'''
        alpha = self.tick_value
        data_frame = self.df_trades_by_price
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
        
        fig.add_trace(
            go.Scatter(x=self.data_frame_trades.DateTime,
                       y=self.data_frame_trades.Al), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=self.data_frame_trades.DateTime,
                       y=self.data_frame_trades.Co), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=self.__df.DateTime,
                       y=self.__get_BAspread(self.__df)), row=2, col=1)
        fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
        
        file = path_out+self.file_name(True)+"_price.html"
        print('saving html plot to ', file)
        fig.write_html(file)
    
    def get_Armada_UZModel_output(self):
        return Armada_UZModel_output(self.df_cont_alt_by_ticks, self.df_uz_stats)
        
