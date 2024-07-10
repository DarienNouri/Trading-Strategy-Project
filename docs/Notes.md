# Project Resources



This markdown page serves as a reference for a project focused on evaluating Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM) networks, Autoregressive Integrated Moving Average (ARIMA), and Ordinary Least Squares (OLS) models. The goal is to investigate their efficacy in generating trading signals for U.S. equities, with an emphasis on linking model predictive accuracy to actionable trading strategies for improved risk management and portfolio optimization.


## Concepts and Ideas


[Entire Pairs Strategy (check it out)](https://daehkim.github.io/pair-trading/)


- **LSTM for Momentum Trading**: An exploration of using LSTM networks to enhance momentum trading strategies by incorporating mean reversion and changepoint detection, providing a comprehensive method for trend estimation and position sizing. [Link](https://medium.datadriveninvestor.com/create-superior-momentum-trading-with-tensorflow-5de203f8334f)

- **Cointegration and Pairs Trading**: A detailed guide on identifying cointegrated stock pairs using statistical tests, estimating error-correction models, and backtesting pairs trading strategies in Python. It offers a comprehensive exploration of pairs trading mechanics, including the use of statistical tests for cointegration and the practical application of these concepts through Python code examples. [Blog Post](https://letianzj.github.io/cointegration-pairs-trading.html)

- **Sparse mean reversion**: Detailed guide on identifying and weighting basket of stocks for sparse mean reversion strategy. [Blog Post](https://hudsonthames.org/sparse-mean-reverting-portfolio-selection/)

[Time Series Models For Volatility FOrecasts and Statistical Arbitrage](https://ml4trading.io/chapter/8)

## Examples

1. **Trading Probability and PnL Calculation**: A basic guide on calculating trading probabilities and profit/loss, useful for foundational trading strategies. [GitHub](https://github.com/ThomasAFink/trading-profit-loss-diagram-and-simple-trading-probabilities/blob/main/README.md)

2. **Simple Reversion Strategy**: Implementation of a straightforward mean reversion strategy, offering insights into its application in trading. [GitHub](https://github.com/Laurier-Fintech/OpenFintech/blob/main/README.md)

3. **Weekly Stock Pair Spread Prediction Using LSTM**: An LSTM model aimed at predicting the weekly spread between stock pairs, demonstrating a basic approach to autoregressive modeling. [GitHub](https://github.com/fplon/trading_strategies/blob/master/structural_first_busines_day_strategy.ipynb)

4. **Pair Trading for Cointegrating Currencies with LSTM**: Utilizes LSTM networks for statistical arbitrage in currency pairs, focusing on the Canadian and Australian dollars through cointegration and mean reversion strategies. [GitHub](https://github.com/shimonanarang/pair-trading)




## Publications

- **Deep Learning for Trading Strategy**: This paper introduces a trading strategy that employs deep learning for changepoint detection, aiming to combine slow momentum with fast reversion for enhanced trading performance. [arXiv](https://arxiv.org/pdf/2105.13727v3.pdf) | [ResearchGate](https://www.researchgate.net/publication/356936311_Slow_Momentum_with_Fast_Reversion_A_Trading_Strategy_Using_Deep_Learning_and_Changepoint_Detection)
- **Sparse Mean Reversion Strategy using LSTM and VAR**: [link](https://intapi.sciendo.com/pdf/10.2478/ausi-2021-0013)

[an Autoregressive approach to pairs trading](https://cs229.stanford.edu/proj2017/final-reports/5244154.pdf)

## Tools and Libraries

- **deepdow**: A very comprehensive Python package for portfolio optimization using deep learning techniques. It connects portfolio optimization with deep learning by allowing networks to perform weight allocation in a single forward pass. This framework facilitates the merger of market forecasting and optimization problem design, providing a unique approach to portfolio management with deep learning. [GitHub](https://github.com/jankrepl/deepdow)


- **VectorBT**: A powerful Python library designed for backtesting, optimizing, and analyzing trading strategies at scale. It offers a flexible framework to rapidly prototype and evaluate models with an extensive array of data analytics and visualization tools. Ideal for comparing performance metrics across different machine learning models in trading. [GitHub](https://github.com/polakowo/vectorbt)