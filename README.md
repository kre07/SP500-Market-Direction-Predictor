# S&P 500 Market Direction Predictor

A machine-learning project that predicts whether the S&P 500 is likely to move higher on the next trading day.

The project uses historical market data from Yahoo Finance, engineers technical and macro-market features, trains a Random Forest classifier, and evaluates the model with walk-forward backtesting. Instead of trying to predict the exact future price, the model focuses on a simpler classification problem: **will the next trading day close higher than the current day?**

## Overview

The predictor is built in Python inside a Jupyter Notebook. Historical S&P 500 data is downloaded using `yfinance`, transformed into a set of market indicators, and passed into a `RandomForestClassifier`.

The final configuration uses a **65% confidence threshold**. A BUY signal is only produced when the model estimates at least a 65% probability that the market will move higher the next day. This reduces the number of trades in exchange for focusing on higher-confidence predictions.

In the saved notebook run, the final configuration achieved:

- **Precision:** 66.13%
- **Predicted BUY signals during backtesting:** 62
- **BUY threshold:** 65%

> The 66.13% result is a precision score, not overall classification accuracy. Results can change when the notebook is rerun because Yahoo Finance provides updated market data.

## Features

The model uses several groups of market information:

### S&P 500 Historical Data

The notebook downloads historical data for the S&P 500 index using the Yahoo Finance ticker:

```python
^GSPC
```

If the index data cannot be downloaded, the code falls back to the SPDR S&P 500 ETF:

```python
SPY
```

The model uses data beginning in 1990.

### Next-Day Prediction Target

The target is created by comparing the next trading day's closing price with the current day's closing price:

```python
sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
```

The target is:

- `1` if the next day's closing price is higher
- `0` if the next day's closing price is lower or equal

### Multi-Timeframe Price Features

The project compares the current S&P 500 closing price against rolling averages over several time horizons:

```text
2 days
5 days
60 days
250 days
1000 days
```

For each horizon, the notebook creates:

- a closing-price ratio
- a market trend feature showing how many previous days were positive

These features give the model both short-term and long-term market context.

### RSI

The notebook calculates the **Relative Strength Index (RSI)** using a 14-day window.

RSI gives the model information about recent price momentum and whether recent price movements have been unusually strong in either direction.

### VIX

The project downloads the CBOE Volatility Index:

```python
^VIX
```

The VIX is used as an additional market-risk feature and provides information about current market volatility.

### Interest Rates

The notebook also downloads the U.S. 10-Year Treasury Yield:

```python
^TNX
```

For the Treasury yield, the project creates rolling ratios and trend features over the same 2, 5, 60, 250, and 1000-day horizons.

This gives the model additional macroeconomic context alongside stock-market price data.

## Machine Learning Model

The final model is a Scikit-learn `RandomForestClassifier`:

```python
RandomForestClassifier(
    n_estimators=200,
    min_samples_split=100,
    random_state=1
)
```

### Why Random Forest?

Random Forest works well for this project because it can model nonlinear relationships between multiple indicators without requiring a fixed mathematical relationship between the inputs and the output.

The larger `min_samples_split` value also helps prevent individual trees from learning extremely small patterns that may not generalize well to unseen market data.

## Confidence Filter

Instead of using the default 50% classification threshold, the model uses predicted probabilities:

```python
preds = model.predict_proba(test[predictors])[:, 1]
```

A BUY prediction is only made when:

```python
probability >= 0.65
```

Otherwise, the model returns a DO NOT BUY signal.

The purpose of the higher threshold is to trade less frequently while focusing on predictions where the model has stronger confidence.

## Walk-Forward Backtesting

The project uses walk-forward backtesting rather than training and testing on randomly mixed market dates.

The backtest begins after approximately 2,500 trading days of historical data and then moves forward in blocks of 250 trading days.

Conceptually:

```text
Train on earlier market history
        ↓
Predict the next period
        ↓
Add more historical data
        ↓
Train again
        ↓
Predict the next period
        ↓
Repeat
```

This better represents how a model would have been used historically because future data is not intentionally included in the training data for earlier predictions.

## Evaluation Metric

The main evaluation metric is **precision**:

```python
precision_score(predictions["Target"], predictions["Predictions"])
```

Precision answers the question:

> Out of all the days when the model predicted that the S&P 500 would go up, how often did the market actually go up?

The saved final run produced:

```text
FINAL MODEL (66% CONFIGURATION)
Precision Score: 0.6613 (66.13%)
Total Trades: 62
```

## Next-Day Signal

After backtesting, the notebook takes the most recent available market features and calculates a probability for the next market move.

Example output:

```text
PREDICTION FOR TOMORROW
Model Confidence: 53.9% (Requires 65% to Buy)
SIGNAL: DO NOT BUY
```

If the predicted probability is at least 65%, the program outputs a BUY signal. Otherwise, it outputs DO NOT BUY.

Because market data changes over time, the exact probability and signal will change whenever the notebook is rerun with updated Yahoo Finance data.

## Project Workflow

```text
Yahoo Finance
     |
     v
S&P 500 Historical Data
     |
     +----------------------+
     |                      |
     v                      v
Price / Trend Features    RSI
     |                      |
     +----------+-----------+
                |
        +-------+-------+
        |               |
        v               v
       VIX          10-Year Yield
        |               |
        +-------+-------+
                |
                v
       Feature Engineering
                |
                v
       Random Forest Model
                |
                v
    Walk-Forward Backtesting
                |
                v
        Precision Analysis
                |
                v
     Next-Day Probability
                |
          +-----+-----+
          |           |
       >= 65%       < 65%
          |           |
         BUY      DO NOT BUY
```

## Technologies

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- yfinance
- Yahoo Finance market data
- Random Forest classification

## Project Structure

```text
SP500-Market-Direction-Predictor/
│
├── S&P 500 Predictor.ipynb
├── README.md
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/SP500-Market-Direction-Predictor.git
cd SP500-Market-Direction-Predictor
```

Install the required Python packages:

```bash
python -m pip install -r requirements.txt
```

Open the notebook in either JupyterLab, Jupyter Notebook, or Visual Studio Code with the Microsoft Jupyter extension.

Then run the notebook from the first cell to the last cell.

## Requirements

The project requires the following Python packages:

```text
yfinance
pandas
numpy
scikit-learn
```

## What This Project Demonstrates

This project demonstrates practical experience with:

- collecting financial data from an external data source
- cleaning and preparing time-series data
- creating custom technical and macroeconomic features
- binary classification with Scikit-learn
- Random Forest model configuration
- probability-based decision thresholds
- walk-forward backtesting
- evaluating model precision
- working with Pandas time-series data
- building an end-to-end machine-learning pipeline in a Jupyter Notebook

## Interview Summary

A concise way to explain the project in an interview:

> I built a machine-learning model in Python that predicts whether the S&P 500 will move higher on the next trading day. I used Yahoo Finance to collect historical market data and engineered features based on price trends over multiple time horizons, RSI, the VIX, and the 10-year Treasury yield. I trained a Random Forest classifier and used walk-forward backtesting so the model was always evaluated on future periods it had not intentionally trained on. I also used a 65% probability threshold so the model only generated BUY signals when confidence was relatively high. In one saved run, the final configuration achieved about 66% precision on its BUY predictions.

## Notes

This project is intended for educational and portfolio purposes. It is not financial advice, and historical backtesting performance does not guarantee future market performance.
