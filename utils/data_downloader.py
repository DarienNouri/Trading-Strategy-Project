# File: data_downloader.py

"""
This file contains a class DataUtil, which provides key functionalities to download stock price data from Yahoo
Finance for a given range of dates. It can handle either a single ticker or a list of tickers.

The module uses Yahoo Finance API via the yfinance library to fetch data for the given tickers. The downloaded data is
then cleaned up to include only the price type specified by the user. If the price type is not specified,
it defaults to the 'Close' price.

The main method get_data() provides an easy to use interface which requires only tickers and dates as parameters.

This class also includes a logging mechanism which logs the errors and warnings during the download process.
"""


import yfinance as yf
import datetime
from typing import List, Optional, Union
import pandas as pd
from rich import print


class DataUtils:
    example_usage = """
        A class which provides functionalities to fetch and clean stock price data.

        Example Usage:
        -------------

        ## Creating an Instance
        data_downloader = DataUtils(verbose=True)
        `verbose` argument can be set to `True` to print detailed output during download process.
        It defaults to `False`.

        ## Getting Data for Single Ticker (Default Price Type)
        data = data_downloader.get_data('AAPL', datetime.datetime(2020, 1, 1), datetime.datetime(2020, 12, 31))
        The above call downloads 'Close' price data for 'AAPL' stock for the year 2020.

        ## Getting Data for Multiple Tickers (Default Price Type)
        data = data_downloader.get_data(['AAPL', 'GOOG'], datetime.datetime(2020,1,1), datetime.datetime(2020,12,31))
        The above call downloads 'Close' price data for 'AAPL' and 'GOOG' stocks for the year 2020.

        ## Getting Data for Single Ticker (Specific Price Type)
        data = data_downloader.get_data('AAPL', datetime.datetime(2020,1,1), datetime.datetime(2020,12,31), price_type='Open')
        This call downloads 'Open' price data for 'AAPL' stock for the year 2020.

        ## Getting Data for Multiple Tickers (Multiple Price Types)
        data = data_downloader.get_data(['AAPL', 'GOOG'], datetime.datetime(2020,1,1), datetime.datetime(2020,12,31), price_type=['Open', 'Close'])
        The above call downloads 'Open' and 'Close' price data for 'AAPL' and 'GOOG' stocks for the year 2020.
    """

    def __init__(self, verbose: bool = True):
        """Constructs the necessary attributes for the DataDownloader object. Takes a boolean argument `verbose`"""

        self.verbose = verbose

        if self.verbose:
            print(DataUtils.example_usage)

    def validate_price_type(self, price_type: Union[str, List[str]]) -> Union[str, List[str]]:
        """Validates the input price type(s), checks if it/they is/are one of ["Open", "High", "Low", "Close", "Adj Close"]"""

        valid_price_types = ["Open", "High", "Low", "Close", "Adj Close"]

        if isinstance(price_type, str):
            if price_type not in valid_price_types:
                print(f"Invalid price type: {price_type}. Defaulting to 'Close'")
                return ['Close']
            return [price_type]
        elif isinstance(price_type, list):
            invalid_types = set(price_type) - set(valid_price_types)
            if invalid_types:
                print(f"Invalid price types: {invalid_types}. Ignoring these.")
                return list(set(price_type) - set(invalid_types))
            return price_type


    def get_data(self, tickers: Union[str, List[str]], start_date: datetime.datetime,
                    end_date: datetime.datetime, price_type: Union[str, List[str]] = 'Close',
                    *args, **kwargs
                    ) -> Union[pd.DataFrame, pd.Series]:
        """

        Method that takes either a string or list of tickers, start and end dates as required params,
        and returns the price data along with *args and **kwargs that can be used for additional
        parameters in yfinance download method.

        Parameters:
        tickers (str or list[str]) : Single ticker as a string or list of tickers as strings
        start_date (datetime.datetime) : Start date for which data has to be downloaded
        end_date (datetime.datetime) : End date until which data has to be downloaded
        price_type (str or list[str]) : The type of price data to be fetched. Defaults to 'Close'
        *args, **kwargs : Additional parameters to be passed to the yfinance download method

        Returns
        .........
        pd.DataFrame/ pd.Series (if tickers is a single str)
        """
        price_type = self.validate_price_type(price_type)
        print(f"Pulled {price_type} for {tickers} from {start_date} to {end_date}")
        try:
            data = yf.download(tickers, start=start_date, end=end_date, progress=self.verbose, *args, **kwargs)

            if data.empty:
                print(f"No data for {tickers}")
                return pd.DataFrame()

            price_data = data[price_type]

            # Check if tickers is a string, (i.e., single ticker provided), then return Series.
            if isinstance(tickers, str) and len(price_type) == 1:
                # set second level column name as the ticker
                # price_data.columns = pd.MultiIndex.from_product([price_data.columns, [tickers]])
                price_data = price_data.squeeze()
                price_data.name = tickers
                return price_data


            if len(price_type) == 1:
                price_data.columns = price_data.columns.droplevel(0)

            return price_data

        except Exception as e:
            print(f"Error downloading data for {tickers}: {e}")
            return pd.DataFrame()
