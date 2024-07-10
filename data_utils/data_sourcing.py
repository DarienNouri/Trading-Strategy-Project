import os
import sys
import json
import logging
import datetime as dt
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf
import refinitiv.data as rd
from dotenv import load_dotenv

from data_utils.update_market_data import update_market_data

pd.set_option("display.precision", 8)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load user-specific paths from env
load_dotenv()
REFINITIV_CONFIG_PATH = os.getenv('REFINITIV_CONFIG_PATH')
PROJECT_ROOT = os.getenv('PROJECT_ROOT')
MARKET_DATA_PATH = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, PROJECT_ROOT)

# Open Refinitiv session
# rd.open_session(config_name='/Users/darien/Configs/refinitiv-data.config.json').open()
rd.open_session(config_name=REFINITIV_CONFIG_PATH).open()


def date_utc(date_):
    """Convert date to UTC and remove timezone info"""
    return pd.to_datetime(date_, utc=True).dt.tz_localize(None)

def data_update():
    """Check if market data needs updating and update if necessary"""
    data_files = ['stocks.txt', 'indexes.txt', 'futures.txt', 'forex.txt']
    day_limit = 15

    for file in data_files:
        df = pd.read_csv(os.path.join(MARKET_DATA_PATH, file))
        if (dt.datetime.now() - pd.to_datetime(df['Last Update'][0])).days >= day_limit:
            update_market_data()
            break

class DataSourcing:
    """A class to handle data sourcing from various exchanges"""

    def __init__(self):
        self.df_stocks = pd.read_csv(os.path.join(MARKET_DATA_PATH, 'stocks.txt'))
        self.df_indexes = pd.read_csv(os.path.join(MARKET_DATA_PATH, 'indexes.txt'))
        self.df_futures = pd.read_csv(os.path.join(MARKET_DATA_PATH, 'futures.txt'))
        self.df_forex = pd.read_csv(os.path.join(MARKET_DATA_PATH, 'forex.txt'))

    def exchange_data(self, exchange: str):
        """Set exchange and prepare relevant data"""
        self.exchange = exchange
        if self.exchange != 'Binance':
            self.stock_indexes = np.sort(self.df_stocks['Index Fund'].unique())
            self.indexes = np.sort(self.df_indexes['Indexes'].unique())
            self.futures = np.sort(self.df_futures['Futures'].unique())
            self.forex = np.sort(self.df_forex['Currencies'].unique())

    def market_data(self, market: str):
        """Set market and prepare relevant data"""
        self.market = market
        if self.exchange == 'Yahoo! Finance':
            self.stocks = np.sort(
                self.df_stocks[self.df_stocks['Index Fund'] == self.market]['Company'].unique()
            )

    def intervals(self, selected_interval: Optional[str] = None, 
                  start_date: Optional[str] = None, 
                  end_date: Optional[str] = None):
        """Set interval and date range for data retrieval"""
        self.start_date = start_date
        self.end_date = end_date
        self.selected_interval = selected_interval
        self.period = None

        exchange_interval = {
            'Yahoo! Finance': {
                '5 Minute': '5m', '15 Minute': '15m', '30 Minute': '30m',
                '1 Hour': '60m', '1 Day': '1d', '1 Week': '1wk', '1 Month': '1mo'
            }
        }

        self.exchange_interval = exchange_interval.get(self.exchange, {}).get(self.selected_interval, self.selected_interval)

        if self.exchange == 'Yahoo! Finance':
            self.period = {
                '1 Minute': '7d',
                '5 Minute': '1mo', '15 Minute': '1mo', '30 Minute': '1mo',
                '1 Hour': '2y'
            }.get(self.selected_interval, 'max')

    def get_refinitiv(self, ticker: str, start_date: str, end_date: str, 
                      selected_interval: str, ticker_suffix: str, interval_prefix: str) -> Any:
        """Retrieve historical pricing data from Refinitiv Data Platform"""
        try:
            return rd.content.historical_pricing.summaries.Definition(
                ticker + ticker_suffix,
                start=start_date,
                end=end_date,
                interval=interval_prefix + selected_interval.upper()
            ).get_data()
        except Exception as e:
            logger.info(f"Error getting data for {ticker + ticker_suffix} with interval {interval_prefix + selected_interval.upper()} and suffix {ticker_suffix}: {e}")
        raise ValueError(f"No valid combination of interval and ticker suffix found for {ticker}")

    def apis(self, asset: str):
        """Retrieve data from the appropriate API based on the exchange"""
        self.asset = asset

        if self.exchange == 'Refinitiv':
            self._get_refinitiv_data()
        elif self.exchange == 'Yahoo! Finance':
            self._get_yahoo_finance_data()
        else:
            raise ValueError("Incorrect Exchange Selection")
        
        self._process_dataframe()

    def _get_refinitiv_data(self):
        """Retrieve data from Refinitiv"""
        logger.info(f'Querying Refinitiv for {self.asset}')
        self.ticker = self.asset
        rd.open_session(config_name=REFINITIV_CONFIG_PATH).open()
        
        if self.start_date and self.end_date:
            for interval_prefix in ["P", "PT"]:
                for ticker_suffix in [".O", "", ".I"]:
                    try:
                        rd_raw = self.get_refinitiv(self.ticker, self.start_date, self.end_date, 
                                                    self.selected_interval, ticker_suffix, interval_prefix)
                        self.df = rd_raw.data.df.rename(columns={
                            'TRDPRC_1': 'Close', 'HIGH_1': 'High', 'LOW_1': 'Low',
                            'ACVOL_UNS': 'Volume', 'OPEN_PRC': 'Open'
                        }).astype(float).reset_index()
                        self.df['Adj Close'] = self.df['Close']
                        return
                    except Exception:
                        continue
            raise ValueError(f"Unable to retrieve Refinitiv data for {self.asset}")

    def _get_yahoo_finance_data(self):
        """Retrieve data from Yahoo! Finance"""
        self.ticker = self._get_yahoo_ticker()
        
        download_params = {
            'tickers': self.ticker,
            'interval': self.selected_interval,
            'auto_adjust': True,
            'prepost': True,
            'threads': True,
            'proxy': None
        }

        if self.start_date and self.end_date:
            download_params.update({'start': self.start_date, 'end': self.end_date})
        elif self.period:
            download_params['period'] = self.period

        self.df = yf.download(**download_params).reset_index()
        self.df = self.df.rename(columns={'Datetime': 'Date', 'Close': 'Adj Close'})

    def _get_yahoo_ticker(self) -> str:
        """Determine the correct Yahoo! Finance ticker"""
        try:
            return self.df_stocks[
                (self.df_stocks['Company'] == self.asset) & 
                (self.df_stocks['Index Fund'] == self.market)
            ]['Ticker'].values[0]
        except IndexError:
            for df in [self.df_indexes, self.df_futures, self.df_forex]:
                try:
                    return df[df.iloc[:, 0] == self.asset]['Ticker'].values[0]
                except IndexError:
                    continue
            return self.asset

    def _process_dataframe(self):
        """Process the retrieved dataframe"""
        self.df['Date'] = date_utc(self.df['Date'])
        self.df = self.df.set_index('Date')
        self.df = self.df[['High', 'Low', 'Open', 'Volume', 'Adj Close']].apply(pd.to_numeric)
        self.price_history_df = self.df.copy()
        self.price_history_df['Close'] = self.price_history_df['Adj Close']

    def __repr__(self):
        return (f"DataSourcing(exchange={self.exchange}, market={self.market}, "
                f"asset={self.asset}, start_date={self.start_date}, "
                f"end_date={self.end_date}, interval={self.selected_interval})\n"
                "select df, ta_lib_indicators_df, price_history_df")