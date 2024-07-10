import json
import requests
import datetime as dt
import pandas as pd

def fetch_wiki_data(url, table_idx, columns_mapping, index_fund, currency, currency_name, ticker_suffix=None, ticker_split_pos=None):
    try:
        df = pd.read_html(url)[table_idx]
        df = df[list(columns_mapping.keys())]
        df.columns = list(columns_mapping.values())
        if ticker_split_pos is not None:
            df['Ticker'] = df['Ticker'].apply(lambda x: x.split(' ')[ticker_split_pos] + ticker_suffix)
        elif ticker_suffix is not None:
            df['Ticker'] = df['Ticker'].apply(lambda x: x + ticker_suffix)
        df['Index Fund'] = index_fund
        df['Currency'] = currency
        df['Currency_Name'] = currency_name
    except:
        df = pd.DataFrame(columns=['Ticker', 'Company', 'Index Fund', 'Currency', 'Currency_Name'])
    return df

def update_market_data():
    dow_columns = {'Symbol': 'Ticker', 'Company': 'Company'}
    nasdaq_columns = {'Ticker': 'Ticker', 'Company': 'Company'}
    russell_columns = {'Ticker': 'Ticker', 'Company': 'Company'}
    snp_columns = {'Symbol': 'Ticker', 'Security': 'Company'}
    sse_columns = {'Ticker symbol': 'Ticker', 'Name': 'Company'}
    csi_columns = {'Index': 'Index', 'Company': 'Company'}
    ftse_columns = {'Ticker': 'Ticker', 'Company': 'Company'}
    dax_columns = {'Ticker': 'Ticker', 'Company': 'Company'}
    cac_columns = {'Ticker': 'Ticker', 'Company': 'Company'}
    bse_sensex_columns = {'Symbol': 'Ticker', 'Companies': 'Company'}
    nifty_columns = {'Symbol': 'Ticker', 'Company Name': 'Company'}
    asx_columns = {'Code': 'Ticker', 'Company': 'Company'}

    WIKI_BASE = "https://en.wikipedia.org/wiki/"

    index_data = [
        (f"{WIKI_BASE}Dow_Jones_Industrial_Average", 1, dow_columns, 'US Dow Jones', 'USD', 'US Dollar'),
        (f"{WIKI_BASE}Nasdaq-100", 4, nasdaq_columns, 'US NASDAQ 100', 'USD', 'US Dollar'),
        (f"{WIKI_BASE}Russell_1000_Index", 2, russell_columns, 'US Russell 1000', 'USD', 'US Dollar'),
        (f"{WIKI_BASE}List_of_S%26P_500_companies", 0, snp_columns, 'US S&P 500', 'USD', 'US Dollar'),
        (f"{WIKI_BASE}SSE_50_Index", 1, sse_columns, 'Chinese SSE 50', 'CNY', 'Chinese Yuan', '.SS', 1),
        (f"{WIKI_BASE}CSI_300_Index", 3, csi_columns, 'Chinese CSI 300', 'CNY', 'Chinese Yuan'),
        (f"{WIKI_BASE}FTSE_100_Index", 4, ftse_columns, 'British FTSE 100', 'GBP', 'British Pound', '.L'),
        (f"{WIKI_BASE}DAX", 4, dax_columns, 'German DAX', 'EUR', 'Euro'),
        (f"{WIKI_BASE}CAC_40", 4, cac_columns, 'French CAC 40', 'EUR', 'Euro'),
        (f"{WIKI_BASE}BSE_SENSEX", 1, bse_sensex_columns, 'Indian S&P BSE SENSEX', 'INR', 'Indian Rupee'),
        (f"{WIKI_BASE}NIFTY_50", 2, nifty_columns, 'Indian Nifty 50', 'INR', 'Indian Rupee', '.NS'),
        (f"{WIKI_BASE}S%26P/ASX_200", 1, asx_columns, 'Australian S&P ASX 200', 'AUD', 'Australian Dollar', '.AX')
    ]


    df_list = [fetch_wiki_data(*data) for data in index_data]
    df_stocks = pd.concat(df_list, ignore_index=True)
    df_stocks.loc[0, 'Last Update'] = dt.date.today()
    df_stocks.to_csv('ticker_associations/stocks.txt', index=False)

    try:
        df_forex = pd.read_html('https://finance.yahoo.com/currencies')[0]
        df_forex = df_forex[['Symbol', 'Name']].iloc[:-1]
        df_forex.columns = ['Ticker', 'Currencies']
        df_forex['Currency'] = df_forex['Currencies'].apply(lambda x: x.split('/')[0])
        df_forex['Market'] = df_forex['Currencies'].apply(lambda x: x.split('/')[1])
        df_forex['Currencies'] = df_forex['Currencies'].str.replace('/', ' to ')
        df_forex.loc[0, 'Last Update'] = dt.date.today()
        df_forex.to_csv('ticker_associations/forex.txt', index=False)
    except:
        pass

    try:
        df_futures = pd.read_html('https://finance.yahoo.com/commodities')[0]
        df_futures = df_futures[['Symbol', 'Name']]
        df_futures.columns = ['Ticker', 'Futures']
        additional_futures = [['BTC=F', 'Bitcoin Futures'], ['ETH=F', 'Ether Futures'],  ['DX=F', 'US Dollar Index']]
        df_futures = pd.concat([df_futures, pd.DataFrame(additional_futures, columns=['Ticker', 'Futures'])]).drop_duplicates(subset=['Ticker', 'Futures'])
        df_futures.loc[0, 'Last Update'] = dt.date.today()
        df_futures.to_csv('ticker_associations/futures.txt', index=False)
    except:
        pass

    try:
        df_indexes = pd.read_html('https://finance.yahoo.com/world-indices/')[0]
        df_indexes = df_indexes[['Symbol', 'Name']]
        df_indexes.columns = ['Ticker', 'Indexes']
        df_indexes.loc[0, 'Last Update'] = dt.date.today()
        df_indexes.to_csv('ticker_associations/indexes.txt', index=False)
    except:
        pass