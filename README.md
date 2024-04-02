# Initial implementation of the project

***See [file](https://github.com/DarienNouri/Trading-Strategy-Prjoect/blob/77504032eafc3947fea0f0414bb1bed2926ee10d/resources/Notes.md) for notes and links to articles/code resources**

### Quick Notes
- dev/Notebooks/* contains some scratch notebooks that includes the ticker selection set.
- data/news_scraping contains the code for scraping the news data from lexis nyu database with some results in data/results. I will work on obtaining the results for the remaining tickers over the upcoming days. Currently rate limitted.


```python
selected_stock_basket = [
    'GE',
    'HON',
    'DE',
    'UNP',
    'FDX',
    'JPM',
    'BAC',
    'WFC',
    'C',
    'GS',
    'MS',
    'AXP',
    'BLK',
    'AAPL',
    'GOOGL',
    'INTC',
    'CRM'
 ]
```







<br/>
<br/>



### Overview of Utils Module
----
I created a utils module that packages functions and classes for the project.

See the [utils_usage_examples.ipynb](https://github.com/DarienNouri/Trading-Strategy-Prjoect/blob/c6327f94ccef0929b11cef79f4a6d02e51100c98/examples/utils_usage_examples.ipynb) notebook for usage examples.

 This includes:

- data_downloader.py
incldues DataUtils class to download stock data using yfinance

- stock_visualizer.py
has some functions for visualizing statistical tests related to time series and financial data, some for preprocessing and some general

- analyser.py
Bulk of the utils. Inlcudes many functions for time series and financial data analysis. Raning from preprocessing to signal generation.

Note: I also included some resources, including articles book and python libs, I came across that may be useful for the project in [notes.md](https://github.com/DarienNouri/Trading-Strategy-Prjoect/blob/c6327f94ccef0929b11cef79f4a6d02e51100c98/resources/Notes.md)


Utils overview:

## TradingLib [Utils](https://github.com/DarienNouri/Trading-Strategy-Prjoect/tree/c6327f94ccef0929b11cef79f4a6d02e51100c98/utils) Module directory

- [utils.data_downloader](https://github.com/DarienNouri/Trading-Strategy-Prjoect/blob/c6327f94ccef0929b11cef79f4a6d02e51100c98/utils/data_downloader.py)
  - DataUtils
    - get_data
- [utils.analyser](https://github.com/DarienNouri/Trading-Strategy-Prjoect/blob/c6327f94ccef0929b11cef79f4a6d02e51100c98/utils/analyser.py)
  - check_stationarity
  - plot_stationary_analysis
  - transform_to_stationary
  - plot_seasonal_decomposition
  - test_granger_causality
  - calculate_cointegration_matrix
  - identify_cointegrated_pairs
  - calculate_moving_average
  - calculate_exponential_moving_average
  - apply_rolling_ols
  - rolling_logret_zscore_nb
  - calculate_zscore
  - calculate_volatility
  - ols_spread_nb
  - calculate_bollinger_bands
  - calculate_relative_strength_index
  - calculate_pearson_correlation
- [utils.stock_visualizer](https://github.com/DarienNouri/Trading-Strategy-Prjoect/blob/c6327f94ccef0929b11cef79f4a6d02e51100c98/utils/stock_visualizer.py)
  - apply_rolling_average
  - plot_spread
  - plot_ratio
  - plot_scaled_ratio_and_spread
  - plot_zscore_ratio
