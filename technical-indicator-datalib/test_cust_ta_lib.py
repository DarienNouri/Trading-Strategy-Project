# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

%pip install refinitiv-data

#%%

%load_ext autoreload
%autoreload 2
import os 
import sys
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
%matplotlib widget


import warnings
warnings.filterwarnings("ignore")


# %%
sys.path.insert(0, '/Users/darien/Library/Mobile Documents/com~apple~CloudDocs/Code/QuantTrading/TradingProject/auto-technical-analysis')
# %%

# sys.path.insert(0, '/Users/darien/Library/Mobile Documents/com~apple~CloudDocs/Code/QuantTrading/TradingProject/auto-technical-analysis/src')


# %%
exchange = 'Yahoo! Finance'
exchange = 'Refinitiv'
equity = 'AAPL'
market = 'US S&P 500'
start_date = '2021-01-01'
end_date = '2024-01-01'
interval = '1d'

# %%
from src.technical_indicators import Indications, Technical_Calculations
from src.custom_indicators import CustomIndicators

data = CustomIndicators(equity, interval, start_date, end_date, exchange=exchange, market=market)

data.df


#%% # Pull Sentiment Data
# 
sentiment_data_path = "/Users/darien/Library/Mobile Documents/com~apple~CloudDocs/Code/QuantTrading/TradingProject/TradingLibADS/data/bloomberg/cleaning/bloomberg_sentiment_cols_renamed.csv"

sentiment_data = pd.read_csv(sentiment_data_path)

sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date'])
sentiment_data = sentiment_data.set_index('Date')
sentiment_data

# regex for column name which has equity name in it

cols_with_equity = sentiment_data.columns[sentiment_data.columns.str.contains(equity, case=False)]

cols_with_equity

company_sentiment = sentiment_data[cols_with_equity]

# Normalizing data
sentiment_data_normalized = (company_sentiment - company_sentiment.mean()) / company_sentiment.std()
sentiment_data_normalized = sentiment_data_normalized.rename(columns={sentiment_data_normalized.columns[0]: 'Close'})

sentiment_data_normalized.rolling(12).mean().plot()

# sentiment_data_normalized.rolling()

import mplfinance as mpf


data_plot = data.price_history_df.copy()    
# rename index to Date, remove multi-index if present
data_plot.index = data_plot.index.get_level_values(0)
data_plot.index = pd.to_datetime(data_plot.index)
data_plot.index.name = 'Date'
data_plot.columns = list(data_plot.columns)
data_plot = data_plot.drop(columns='Adj Close')

# Select rows where index is in data_plot and the first column only
sentiment_data_plot = sentiment_data_normalized.loc[sentiment_data_normalized.index.isin(data_plot.index), sentiment_data_normalized.columns]


import plotly.express as px
#set default template
px.defaults.template = "darien_dark"
sent_roll = company_sentiment.diff().dropna()
sent_roll.iloc[:, 0] = sent_roll.iloc[:, 0].rolling(10).mean()
sent_roll.iloc[:, 1] = sent_roll.iloc[:, 1].diff().rolling(60).mean()
sent_roll.iloc[:, 2] = sent_roll.iloc[:, 2].diff().rolling(60).mean()
sent_roll.iloc[:, 3] = sent_roll.iloc[:, 3].diff().rolling(60).mean()


# sent_roll['sentiment_20ma'] = sentiment_data_plot.iloc[:, 1].diff().rolling(20).mean()

fig = px.line(sent_roll, x=sent_roll.index, y=sent_roll.columns, title='Sentiment Data')
fig.show()





# %%

# %%
