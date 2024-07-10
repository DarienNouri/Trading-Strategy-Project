import pandas as pd
import ta
from data_utils.data_sourcing import Data_Sourcing

class Technical_Calculations(Data_Sourcing):

    def __init__(self, exchange, asset, interval, start_date, end_date, market = None):
        self.fast_length = 12
        self.slow_length = 26
        self.signal_smoothing = 9
        self.short_run = 20
        self.long_run = 50
        self.rsi_period = 14

        super().__init__()
        super(Technical_Calculations, self).exchange_data(exchange)
        super(Technical_Calculations, self).market_data(market)
        super(Technical_Calculations, self).intervals(interval, start_date, end_date)
        super(Technical_Calculations, self).apis(asset)

    def moving_average_convergence_divergence(self):
        ema1 = self.df['Adj Close'].ewm(span = self.fast_length, adjust = False).mean()
        ema2 = self.df['Adj Close'].ewm(span = self.slow_length, adjust = False).mean()

        self.df['MACD'] = ema1 - ema2
        self.df['MACDS'] = self.df['MACD'].ewm(span = self.signal_smoothing, adjust = False).mean()
        self.df['MACDH'] = self.df['MACD'] - self.df['MACDS']

    def relative_strength_index(self):
        change = self.df['Adj Close'].diff(1)
        gain = change.mask(change < 0, 0)
        loss = change.mask(change > 0, 0)
        average_gain = gain.ewm(com = self.rsi_period - 1, min_periods = self.rsi_period).mean()
        average_loss = loss.ewm(com = self.rsi_period - 1, min_periods = self.rsi_period).mean()
        rs = abs(average_gain / average_loss)

        self.df['RSI'] = 100 - (100 / (1 + rs))

    def slow_stochastic(self):
        low_stochastic = self.df['Low'].rolling(window = self.rsi_period).min()
        high_stochastic = self.df['High'].rolling(window = self.rsi_period).max()
        fast_k = 100 * ((self.df['Adj Close'] - low_stochastic) / (high_stochastic - low_stochastic))
        self.df['SR_K'] = fast_k.rolling(window = 3).mean()
        self.df['SR_D'] = self.df['SR_K'].rolling(window = 3).mean()

    def stochastic_rsi(self):
        low_rsi = self.df['RSI'].rolling(window = self.rsi_period).min()
        high_rsi = self.df['RSI'].rolling(window = self.rsi_period).max()
        fast_k_rsi = 100 * ((self.df['RSI'] - low_rsi) / (high_rsi - low_rsi))
        self.df['SR_RSI_K'] = fast_k_rsi.rolling(window = 3).mean()
        self.df['SR_RSI_D'] = self.df['SR_RSI_K'].rolling(window = 3).mean()

    def moving_averages(self):
        self.df['SMA'] = self.df['Adj Close'].rolling(self.short_run).mean()
        self.df['SEMA'] = self.df['Adj Close'].ewm(com = self.short_run - 1, min_periods = self.short_run).mean()
        self.df['LMA'] = self.df['Adj Close'].rolling(self.long_run).mean()
        self.df['LEMA'] = self.df['Adj Close'].ewm(com = self.long_run - 1, min_periods = self.long_run).mean()

    def pivot_point(self):
        self.df['P'] = (self.df['Adj Close'] + self.df['High'] + self.df['Low']) / 3
        self.df['R1'] = self.df['P'] + (0.382 * (self.df['High'] - self.df['Low']))
        self.df['R2'] = self.df['P'] + (0.618 * (self.df['High'] - self.df['Low']))
        self.df['R3'] = self.df['P'] + (1 * (self.df['High'] - self.df['Low']))
        self.df['S1'] = self.df['P'] - (0.382 * (self.df['High'] - self.df['Low']))
        self.df['S2'] = self.df['P'] - (0.618 * (self.df['High'] - self.df['Low']))
        self.df['S3'] = self.df['P'] - (1 * (self.df['High'] - self.df['Low']))

    def on_balance_volume(self):
        self.df.loc[((self.df['Adj Close']) > (self.df['Adj Close'].shift(1))), 'OBV_Volume'] = self.df['Volume']
        self.df.loc[((self.df['Adj Close']) < (self.df['Adj Close'].shift(1))), 'OBV_Volume'] = self.df['Volume'] * (-1)
        self.df.loc[((self.df['Adj Close']) == (self.df['Adj Close'].shift(1))), 'OBV_Volume'] = 0

        self.df['OBV'] = self.df['OBV_Volume'].cumsum()
        self.df = self.df.drop(['OBV_Volume'], axis = 1)

    def average_true_range(self):
        self.df['TR0'] = abs(self.df['High'] - self.df['Low'])
        self.df['TR1'] = abs(self.df['High'] - self.df['Adj Close'].shift())
        self.df['TR2'] = abs(self.df['Low'] - self.df['Adj Close'].shift())
        tr = self.df[['TR0', 'TR1', 'TR2']].max(axis = 1)
        self.df['ATR'] = tr.ewm(alpha = 1/self.rsi_period, adjust = False).mean()
        self.df = self.df.drop(['TR0', 'TR1', 'TR2'], axis = 1)

    def price_analysis(self):
        self.df['HL_PCT'] = (self.df['High'] - self.df['Low']) / self.df['Adj Close'] * 100.0
        self.df['PCT_CHG'] = (self.df['Adj Close'] - self.df['Open']) / self.df['Open'] * 100.0


    def MA(self, window: int):
        self.df[f'MA_{window}'] = self.df['Adj Close'].rolling(window=window).mean()

    def EMA(self, window: int):
        self.df[f'EMA_{window}'] = self.df['Adj Close'].ewm(span=window, adjust=False).mean()

    def TA_lib_all(self):
        return ta.add_all_ta_features(self.df.copy(),
                                      open="Open",
                                      high="High",
                                      low="Low",
                                      close="Adj Close",
                                      volume="Volume",
                                      fillna=True)


class Indications(Technical_Calculations):

    def __init__(self,
                 asset,
                 interval,
                 start_date=None,
                 end_date=None,
                 exchange='Refinitiv',
                 market=None):
        self.engulfing_period = -5
        self.sma = -15
        self.lma = -20

        super().__init__(exchange, asset, interval, start_date, end_date,
                         market)
        super(Indications, self).pivot_point()
        super(Indications, self).on_balance_volume()
        super(Indications, self).moving_average_convergence_divergence()
        super(Indications, self).moving_averages()
        super(Indications, self).relative_strength_index()
        super(Indications, self).slow_stochastic()
        super(Indications, self).stochastic_rsi()
        super(Indications, self).average_true_range()
        super(Indications, self).price_analysis()

    def engulfing_analysis(self):

        self.df.loc[(
            (self.df['Adj Close'] <
             self.df['Open']) |
            (self.df['Adj Close'] <=
             self.df['Open'].shift(self.engulfing_period)) &
            (self.df['Adj Close'].shift(self.
                                                           engulfing_period) >
             self.df['Open'].shift(self.engulfing_period))),
                                       'Engulfing_Indication'] = 2
        self.df.loc[(
            (self.df['Adj Close'] >
             self.df['Open']) |
            (self.df['Adj Close'] >=
             self.df['Open'].shift(self.engulfing_period)) &
            (self.df['Adj Close'].shift(self.
                                                           engulfing_period) <
             self.df['Open'].shift(self.engulfing_period))),
                                       'Engulfing_Indication'] = 0
        self.df[
            'Engulfing_Indication'] = self.df[
                'Engulfing_Indication'].fillna(1)

    def macd_analysis(self):
        self.df.loc[((self.df['MACD'] <
                                         self.df['MACDS'])),
                                       'MADC_Indication'] = 2
        self.df.loc[((self.df['MACD'] >
                                         self.df['MACDS'])),
                                       'MADC_Indication'] = 0
        self.df[
            'MADC_Indication'] = self.df[
                'MADC_Indication'].fillna(1)

    def rsi_divergence_convergence(self):
        self.df.loc[((
            self.df['RSI'] >= 70)),
                                       'RSI_Divagence_Convergence'] = 0
        self.df.loc[((
            self.df['RSI'] <= 30)),
                                       'RSI_Divagence_Convergence'] = 2
        self.df[
            'RSI_Divagence_Convergence'] = self.df[
                'RSI_Divagence_Convergence'].fillna(1)

    def stochastic_analysis(self):
        self.df.loc[(
            (self.df['SR_K'] >
             self.df['SR_D']) &
            (self.df['SR_K'] >= 80) &
            (self.df['RSI'] >= 70)), 'SR_Indication'] = 0
        self.df.loc[(
            (self.df['SR_K'] <
             self.df['SR_D']) &
            (self.df['SR_K'] <= 20) &
            (self.df['RSI'] <= 30)), 'SR_Indication'] = 2
        self.df[
            'SR_Indication'] = self.df[
                'SR_Indication'].fillna(1)

    def moving_average_analysis(self):
        self.df.loc[(
            (self.df['SMA'] <
             self.df['LMA']) &
            (self.df['SMA'].shift(self.sma) >
             self.df['LMA'].shift(self.lma))),
                                       'MA_Indication'] = 0
        self.df.loc[(
            (self.df['SMA'] >
             self.df['LMA']) &
            (self.df['SMA'].shift(self.sma) <
             self.df['LMA'].shift(self.lma))),
                                       'MA_Indication'] = 2
        self.df[
            'MA_Indication'] = self.df[
                'MA_Indication'].fillna(1)

    def support_resistance(self):
        self.df.loc[(
            (self.df['SMA'] <
             self.df['Adj Close']) &
            (self.df['SMA'].shift(self.sma) >
             self.df['Adj Close'].shift(self.sma))),
                                       'Support_Resistance_Indication'] = 0
        self.df.loc[(
            (self.df['SMA'] >
             self.df['Adj Close']) &
            (self.df['SMA'].shift(self.sma) <
             self.df['Adj Close'].shift(self.sma))),
                                       'Support_Resistance_Indication'] = 2
        self.df[
            'Support_Resistance_Indication'] = self.df[
                'Support_Resistance_Indication'].fillna(1)

    def price_action(self):
        self.indication_estimate = 3
        self.df[
            'Indication'] = self.df.loc[:,
                                                           'Engulfing_Indication':].ewm(
                                                               com=self.
                                                               indication_estimate
                                                               - 1,
                                                               min_periods=self
                                                               .
                                                               indication_estimate,
                                                               axis=1
                                                           ).mean(
                                                           ).iloc[:,
                                                                  -1].round(3)
        self.df.loc[(
            (self.df['Indication'] >= 1.25) &
            (self.df['Adj Close'] <=
             self.df['P'])), 'Distinct_Action'] = 'Buy'
        self.df.loc[(
            (self.df['Indication'] <= 0.75) &
            (self.df['Adj Close'] >=
             self.df['P'])), 'Distinct_Action'] = 'Sell'
        self.df[
            'Distinct_Action'] = self.df[
                'Distinct_Action'].fillna('Hold')
        self.df = self.df.drop(
            ['Indication'], axis=1)
        self.df = self.df.dropna()
