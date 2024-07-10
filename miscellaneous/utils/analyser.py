import os
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import statsmodels.api as sm

# File: analyser.py

"""
This file contains some function that may be useful for preprocessing, transformations and analysis
of time series equity data. Feel free to add more functions to this file as needed.

Functions:
    calculate_zscore
    check_stationarity
    plot_stationary_analysis
    plot_seasonal_decomposition
    transform_to_stationary
    calculate_moving_average
    calculate_exponential_moving_average
    calculate_cointegration_matrix
    identify_cointegrated_pairs
    calculate_bollinger_bands
    calculate_relative_strength_index
    test_granger_causality
    calculate_pearson_correlation
    apply_rolling_ols
    calculate_volatility
    rolling_logret_zscore_nb
    ols_spread_nb
    rolling_ols_zscore_nb
"""


def calculate_zscore(series: pd.Series) -> pd.Series:
    """Calculates the Z-score of a given series. Takes a pandas Series and returns a Series of Z-scores."""

    return (series - series.mean()) / np.std(series)


def check_stationarity(time_series: pd.Series, verbose: bool = False) -> float:
    """Tests the stationarity of a given time series data using the Dickey-Fuller Test. Takes a pandas Series and returns the p-value from the test."""

    stationary_test_result = adfuller(time_series.dropna(), autolag="AIC")
    p_value = stationary_test_result[1]

    if verbose:
        print(f"Results of Dickey-Fuller Test (p-value): {p_value:.4f}")

    return p_value


def plot_stationary_analysis(time_series: pd.Series):
    """Generates visualizations for stationary test analysis. Takes a pandas Series."""

    # original time series plot
    plt.figure(figsize=(14, 6))
    plt.subplot(311)
    plt.plot(time_series, color="blue", label="Original")
    plt.legend(loc="best")

    # Time series difference plot
    plt.subplot(312)
    plt.plot(time_series.diff().dropna(), color="red", label="Difference (1 lag)")
    plt.legend(loc="best")

    # plot of time series rolling mean and standard deviation
    plt.subplot(313)
    rolling_mean = time_series.rolling(window=12).mean()
    rolling_std = time_series.rolling(window=12).std()
    plt.plot(rolling_mean, color="green", label="Rolling Mean")
    plt.plot(rolling_std, color="black", label="Rolling Std")
    plt.legend(loc="best")
    plt.title("Rolling Mean & STD for Stationarity Check")

    plt.tight_layout()
    plt.show()

    check_stationarity(time_series, verbose=True)


def plot_seasonal_decomposition(time_series, model="additive", period=252):
    """
    Plots seasonal decomposition of the time series.

    Parameters:
    time_series: Time series data.
    model: 'additive' or 'multiplicative' model to use for decomposition.

    """
    # Perform seasonal decomposition
    decomp = seasonal_decompose(time_series, model=model, period=period)

    # Plot decomposition
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 6))
    decomp.observed.plot(ax=ax1)
    decomp.trend.plot(ax=ax2)
    decomp.seasonal.plot(ax=ax3)
    decomp.resid.plot(ax=ax4)
    ax1.title.set_text("Observed")
    ax2.title.set_text("Trend")
    ax3.title.set_text("Seasonal")
    ax4.title.set_text("Residual")

    plt.tight_layout()
    plt.show()


def transform_to_stationary(time_series: pd.Series, verbose: bool = False) -> pd.Series:
    """Converts a non-stationary time series to stationary. Takes a pandas Series and returns a stationary version of the input series."""

    p_value = check_stationarity(time_series)

    while p_value > 0.05:
        time_series = time_series.diff().dropna()
        p_value = check_stationarity(time_series)

    return time_series


def calculate_moving_average(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Applies moving average to a DataFrame. Takes a DataFrame and window size, returns a DataFrame of moving averages."""

    return df.rolling(window=window).mean()


def calculate_exponential_moving_average(
    df: pd.DataFrame, window: int = 12
) -> pd.DataFrame:
    """Calculates the exponential moving average (EMA) for each column in the DataFrame. Returns a DataFrame of EMAs."""
    return df.ewm(span=window, adjust=False).mean()


def calculate_cointegration_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates cointegration of each pair in a dataframe. Takes a DataFrame and returns a DataFrame of p-values."""

    # Initialize an empty DataFrame to store the p-values
    cointegration_df = pd.DataFrame(np.nan, index=df.columns, columns=df.columns)

    # Calculate the cointegration for each pair
    for i in df.columns:
        for j in df.columns:
            if i != j:
                _, pvalue, _ = coint(df[i], df[j])
                cointegration_df.loc[i, j] = pvalue

    return cointegration_df


def identify_cointegrated_pairs(df: pd.DataFrame, alpha: float = 1.0) -> tuple:
    """Performs a co-integration test on each pair of stocks in the given dataset. Takes a DataFrame and alpha level, returns a tuple of score matrix, p-value matrix, and significantly co-integrated pairs."""

    number_of_columns = df.shape[1]
    score_matrix = np.zeros((number_of_columns, number_of_columns))
    p_value_matrix = np.ones((number_of_columns, number_of_columns))
    column_names = df.keys()
    significant_pairs = []

    for i in range(number_of_columns):
        for j in range(i + 1, number_of_columns):
            series1 = df[column_names[i]]
            series2 = df[column_names[j]]
            test_result = coint(series1, series2)
            score_matrix[i, j] = test_result[0]
            p_value_matrix[i, j] = test_result[1]

            if test_result[1] < alpha:
                significant_pairs.append((column_names[i], column_names[j]))

    return score_matrix, p_value_matrix, significant_pairs

def _sort_pairs(dataframe, ascending=True):
    """
    Function to sort pairs in a dataframe.
    """
    # Set the diagonal to NaN
    np.fill_diagonal(dataframe.values, np.nan)

    # Create a mask for the upper triangle
    upper_triangle_mask = np.triu(np.ones_like(dataframe, dtype=bool))

    # Unstack and sort the pairs
    unstacked_pairs = dataframe.where(upper_triangle_mask).unstack()
    sorted_pairs = unstacked_pairs.sort_values(ascending=ascending)

    return sorted_pairs

def calculate_bollinger_bands(
    df: pd.Series, window: int = 20, num_of_std: int = 2
) -> pd.DataFrame:
    """Calculates Bollinger Bands for a given series. Returns a DataFrame with columns for Upper Band, Lower Band, and SMA."""
    rolling_mean = df.rolling(window=window).mean()
    rolling_std = df.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return pd.concat(
            [upper_band, lower_band, rolling_mean],
            axis=1
        ).rename(columns={0: "Upper Band", 1: "Lower Band", 2: "SMA"})


def calculate_relative_strength_index(df: pd.Series, window: int = 14) -> pd.Series:
    """Calculates the Relative Strength Index (RSI) for a given series. Returns a Series with RSI values."""
    delta = df.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def test_granger_causality(x: pd.Series, y: pd.Series, maxlag: int = 12) -> bool:
    """Tests for Granger causality between two time series. Returns True if causality is found, False otherwise."""
    test_result = grangercausalitytests(
        np.column_stack([x, y]), maxlag=maxlag, verbose=False
    )
    p_values = [
        round(test_result[i + 1][0]["ssr_chi2test"][1], 4) for i in range(maxlag)
    ]
    return any(p_value < 0.05 for p_value in p_values)


def calculate_pearson_correlation(x: pd.Series, y: pd.Series) -> float:
    """Calculates Pearson correlation coefficient between two series. Returns the coefficient."""
    correlation, _ = pearsonr(x, y)
    return correlation


def apply_rolling_ols(y: pd.Series, x: pd.Series, window: int):
    """Applies a rolling Ordinary Least Squares (OLS) regression and returns the coefficient over time."""
    model = RollingOLS(endog=y, exog=sm.add_constant(x), window=window)
    results = model.fit()
    return results.params


def calculate_volatility(df: pd.Series, window: int = 252) -> pd.Series:
    """Calculates the annualized volatility for a given series. Returns a Series with volatility values."""
    daily_ret = df.pct_change()
    volatility = daily_ret.rolling(window=window).std() * np.sqrt(window)
    return volatility


def rolling_logret_zscore_nb(x, period):
    """Calculate the log return spread."""
    a = x.values[period:]
    b = x.shift(period).dropna().values
    a = a[~np.isnan(a if a.shape[0] > b.shape[0] else b)]
    b = b[~np.isnan(b if a.shape[0] > b.shape[0] else b)]

    spread = np.full_like(a, np.nan, dtype=np.float_)
    spread[1:] = np.log(a[1:] / a[:-1]) - np.log(b[1:] / b[:-1])
    zscore = np.full_like(a, np.nan, dtype=np.float_)
    for i in range(a.shape[0]):
        from_i = max(0, i + 1 - period)
        to_i = i + 1
        if i < period - 1:
            continue
        spread_mean = np.mean(spread[from_i:to_i])
        spread_std = np.std(spread[from_i:to_i])
        zscore[i] = (spread[i] - spread_mean) / spread_std
    return spread, zscore


def ols_spread_nb(x):
    """Calculate the OLS spread. Takes price ts returns the spread."""
    a = x.values[1:]
    b = x.shift().dropna().values
    a = a[~np.isnan(a if a.shape[0] > b.shape[0] else b)]
    b = b[~np.isnan(b if a.shape[0] > b.shape[0] else b)]

    a = np.log(a)
    b = np.log(b)
    _b = np.vstack((b, np.ones(len(b)))).T
    slope, intercept = np.dot(np.linalg.inv(np.dot(_b.T, _b)), np.dot(_b.T, a))
    spread = a - (slope * b + intercept)
    return spread[-1]

def _ols_spread_nb(a, b):
    """Calculate the OLS spread. Takes price ts returns the spread."""
    a = a[~np.isnan(a if a.shape[0] > b.shape[0] else b)]
    b = b[~np.isnan(b if a.shape[0] > b.shape[0] else b)]

    a = np.log(a)
    b = np.log(b)
    _b = np.vstack((b, np.ones(len(b)))).T
    slope, intercept = np.dot(np.linalg.inv(np.dot(_b.T, _b)), np.dot(_b.T, a))
    spread = a - (slope * b + intercept)
    return spread[-1]


def rolling_ols_zscore_nb(x, period):
    """Calculate the z-score of the rolling OLS spread. Takes two arrays and returns the spread and z-score."""
    a = x.values[period:]
    b = x.shift(period).dropna().values
    a = a[~np.isnan(a if a.shape[0] > b.shape[0] else b)]
    b = b[~np.isnan(b if a.shape[0] > b.shape[0] else b)]

    spread = np.full_like(a, np.nan, dtype=np.float_)
    zscore = np.full_like(a, np.nan, dtype=np.float_)
    for i in range(a.shape[0]):
        from_i = max(0, i + 1 - period)
        to_i = i + 1
        if i < period - 1:
            continue
        spread[i] = _ols_spread_nb(a[from_i:to_i], b[from_i:to_i])
        spread_mean = np.mean(spread[from_i:to_i])
        spread_std = np.std(spread[from_i:to_i])
        zscore[i] = (spread[i] - spread_mean) / spread_std
    return spread, zscore


def subplot_lines(original_data, transformed_data, title):
    """
    Plots the original and transformed data on two subplots.
    """
    fig, axs = plt.subplots(2, figsize=(10, 6), sharex=True)

    axs[0].plot(original_data)
    axs[0].set_title('Original Data')

    axs[1].plot(transformed_data)
    axs[1].set_title('Transformed Data')

    fig.suptitle(title)
    plt.xticks(rotation=25)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
