"""
CustomFunctions.py contains the following functions for financial data analysis:

1. apply_rolling_average: Implements a moving average on a DataFrame
2. plot_spread: Plots spread time series between two series
3. plot_ratio: Plots ratio time series between two series
4. plot_scaled_ratio_and_spread: Plots scaled spread and ratio time series on the same graph
5. plot_zscore_ratio: Plots z-score of ratio time series
"""

import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from sklearn.preprocessing import StandardScaler
import pandas as pd
import statsmodels.api as sm
import numpy as np

import warnings

warnings.filterwarnings("ignore")


def standardize_input(input1, input2=None):
    """Standardizes inputs to ensure they are in DataFrame format with two columns."""

    if isinstance(input1, pd.DataFrame) and input2 is None:
        assert input1.shape[1] == 2, "DataFrame must have exactly two columns."
        return input1
    elif isinstance(input1, pd.Series) and isinstance(input2, pd.Series):
        if input1.name is None or input2.name is None:
            return pd.DataFrame({f"series1": input1, f"series2": input2})
        else:
            return pd.DataFrame({input1.name: input1, input2.name: input2})

    elif isinstance(input1, np.ndarray) and isinstance(input2, np.ndarray):

        df = pd.DataFrame({f"series1": input1, f"series2": input2})
        return df
    else:
        raise ValueError(
            "Invalid input: please provide two Series, two numpy arrays, or a single DataFrame with two columns."
        )


def apply_rolling_average(df, window_size=3):
    """Applies moving average to DataFrame."""

    df = df.rolling(window=window_size).mean()
    return df


def plot_spread(series1, series2, tickers):
    """Plots the spread between two series."""

    X = sm.add_constant(series1)
    results = sm.OLS(series2, X).fit()
    spread = series2 - results.params[tickers[0]] * series1
    spread.plot(figsize=(12, 6))
    plt.axhline(spread.mean(), color="orange")
    plt.xlabel("Time")
    plt.ylabel("Spread")
    plt.grid(True)
    plt.legend(["Spread", "Mean Spread"])
    plt.title(f"Spread Time Series for {tickers[0]} and {tickers[1]}")
    plt.show()


def plot_spread(input1, input2=None, tickers=None):
    """Plots the spread between two series."""

    df = standardize_input(input1, input2)
    if tickers is None:
        tickers = df.columns.tolist()

    tickers = df.columns.tolist()
    series1, series2 = df[tickers[0]], df[tickers[1]]

    X = sm.add_constant(series1)
    results = sm.OLS(series2, X).fit()
    spread = series2 - results.params[tickers[0]] * series1
    spread.plot(figsize=(12, 6))
    plt.axhline(spread.mean(), color="orange")
    plt.xlabel("Time")
    plt.ylabel("Spread")
    plt.grid(True)
    plt.legend(["Spread", "Mean Spread"])
    plt.title(f"Spread Time Series for {tickers[0]} and {tickers[1]}")
    plt.show()


def plot_ratio(series1, series2=None, tickers=None):
    """Plots ratio time series."""

    df = standardize_input(series1, series2)
    if tickers is None:
        tickers = df.columns.tolist()

    tickers = df.columns.tolist()
    series1, series2 = df[tickers[0]], df[tickers[1]]
    ratio = series1 / series2
    ratio.plot(figsize=(12, 6))
    plt.axhline(ratio.mean(), color="black", label="Mean")
    plt.xlabel("Time")
    plt.ylabel("Value Ratio")
    plt.grid(True)
    plt.legend(["Value Ratio", "Mean Ratio"])
    plt.title(f"Value Ratio Time Series for {tickers[0]} and {tickers[1]}")
    plt.show()


def plot_scaled_ratio_and_spread(series1, series2=None, tickers=None):
    """Plots scaled spread and ratio time series."""

    df = standardize_input(series1, series2)
    if tickers is None:
        tickers = df.columns.tolist()

    series1, series2 = df[tickers[0]], df[tickers[1]]

    X = sm.add_constant(series1)
    results = sm.OLS(series2, X).fit()

    spread = series2 - results.params[tickers[0]] * series1
    ratio = series1 / series2

    scaler = StandardScaler()
    scaled_spread = pd.DataFrame(
        scaler.fit_transform(spread.values.reshape(-1, 1)),
        index=spread.index,
        columns=["spread"],
    )
    scaled_ratio = pd.DataFrame(
        scaler.fit_transform(ratio.values.reshape(-1, 1)),
        index=ratio.index,
        columns=["ratio"],
    )

    plt.figure(figsize=(12, 6))
    scaled_spread["spread"].plot()
    scaled_ratio["ratio"].plot()

    plt.axhline(scaled_spread["spread"].mean(), color="black")
    plt.grid(True)
    plt.legend(["Price Ratio", "Spread"])
    plt.ylabel("Scaled Values")
    plt.title(
        f"Price Ratio and Spread Time Series {tickers[0]} and {tickers[1]} (Level Space)"
    )
    plt.show()


def plot_zscore_ratio(series1, series2=None, tickers=None):
    """Plots z-score of ratio time series."""
    
    # Get the ticker names from the DataFrame's columns
    df = standardize_input(series1, series2)
    if tickers is None:
        tickers = df.columns.tolist()

    # Calculate the ratio
    ratio = df[tickers[0]] / df[tickers[1]]

    # Calculate z-score
    zscore = lambda x: (x - x.mean()) / np.std(x)
    zscore_ratio = ratio.pipe(zscore)

    # Plot
    zscore_ratio.plot(figsize=(12, 6))

    plt.axhline(zscore_ratio.mean(), color="black")
    plt.axhline(1.0, color="red", linestyle="--")
    plt.axhline(-1.0, color="green", linestyle="--")

    plt.xlabel("Time")
    plt.ylabel("Z-Score")
    plt.grid(True)

    plt.legend(["Price Ratio Z-Score", "Mean", "+1", "-1"])
    plt.title(f"Z-Score Price Ratio over Time for {tickers[0]} and {tickers[1]}")
    plt.show()
