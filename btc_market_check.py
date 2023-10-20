#this script checks the daily returns of btc against major market indicies for correlation and cointegration
#Copyright (c)2023 btconometrics 
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
import numpy as np
# Fetch data for Bitcoin
bitcoin_ticker = 'BTC-USD'
bitcoin_data = yf.download(bitcoin_ticker, start="2020-01-01", end="2023-10-20")

# Define the tickers for the major market indices
market_indices = ['^GSPC', '^DJI', '^IXIC', '^FTSE', '^N225', '000001.SS', '^STOXX50E']

# Create a DataFrame to store returns
returns_df = pd.DataFrame()
returns_df['Bitcoin'] = bitcoin_data['Close'].pct_change().dropna()

for market_index_ticker in market_indices:
    market_index_data = yf.download(market_index_ticker, start="2020-01-01", end="2023-10-20")
    returns_df[market_index_ticker] = market_index_data['Close'].pct_change()

# Drop rows with any NaN values
returns_df = returns_df.dropna()

# Create the scatterplot matrix
g = sns.pairplot(returns_df)

# Annotate with correlation and cointegration results
for i, j in zip(*np.triu_indices_from(g.axes, 1)):
    ax = g.axes[i, j]
    correlation = returns_df[g.x_vars[j]].corr(returns_df[g.y_vars[i]])
    
    # Cointegration test
    score, p_value, _ = coint(returns_df[g.x_vars[j]].add(1).cumprod(), returns_df[g.y_vars[i]].add(1).cumprod())
    cointegration_result =  "Cointegrated" if p_value < 0.05 else "Not Cointegrated"
    
    ax.annotate(f'Correlation = {correlation:.2f}\n{cointegration_result}', xy=(0.1, 0.8), xycoords=ax.transAxes)

plt.tight_layout()
plt.subplots_adjust(top=0.85)  # Adjust this value as needed

plt.suptitle("Scatterplot Matrix of Daily Returns\nBitcoin vs. Major Market Indices since January 2020", y=0.95, fontsize=16)


plt.show()
