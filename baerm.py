#(c)2020-2023 Nick Phraudsta (btconometrics/codeorange)
import pandas as pd
import numpy as np
import statsmodels.api as sm
import datetime as dt
from scipy.stats import norm
from scipy import stats
import seaborn as sns
import matplotlib.widgets as widgets
from matplotlib.widgets import Button
import matplotlib
matplotlib.use('TkAgg')
from math import exp, cos, log

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
from scipy import stats
# Import data
url = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
df = pd.read_csv(url)

# Fix date formats and tsset the data
df['date'] = pd.to_datetime(df['time'], format='%Y-%m-%d')
df.drop('time', axis=1, inplace=True)

# Extend the range
df = df.append(pd.DataFrame({'date': pd.date_range(df['date'].max() + pd.Timedelta(days=1), periods=2000, freq='D')})).reset_index(drop=True)

# Replace missing blkcnt values
df.loc[df['BlkCnt'].isnull(), 'BlkCnt'] = 6 * 24

# Generate sum_blocks and hp_blocks
df['sum_blocks'] = df['BlkCnt'].cumsum()
df['hp_blocks'] = df['sum_blocks'] % 210001

# Generate hindicator and epoch
df['hindicator'] = (df['hp_blocks'] < 200) & (df['hp_blocks'].shift(1) > 209000)
df['epoch'] = df['hindicator'].cumsum() 

# Generate reward and daily_reward
df['reward'] = 50 / (2 ** df['epoch'].astype(float))
df.loc[df['epoch'] >= 33, 'reward'] = 0
df['daily_reward'] = df['BlkCnt'] * df['reward']

# Generate tsupply
df['tsupply'] = df['daily_reward'].cumsum()

# Generate logprice
df['logprice'] = np.log(df['PriceUSD'])

# Calculate phaseplus variable
start_date = pd.to_datetime('2008-10-31')
df['days_since_start'] = (df['date'] - start_date).dt.days
df['phaseplus'] = df['reward'] - (df['epoch'] + 1) ** 2

# Define function to calculate Kelly criterion bet size
def kelly_criterion(prob, odds):
    return max(0, (prob * (odds + 1) - 1) / odds)

# Drop rows with date < 2010-07-18
df = df[df['date'] >= dt.datetime.strptime('2010-07-18', '%Y-%m-%d')].reset_index(drop=True)

# Run the regression
mask = df['epoch'] < 2
reg_X = df.loc[mask, ['logprice', 'phaseplus']].shift(1).iloc[1:]
reg_y = df.loc[mask, 'logprice'].iloc[1:]
reg_X = sm.add_constant(reg_X)
ols = sm.OLS(reg_y, reg_X).fit()
coefs = ols.params.values


# Step 1: Calculate AR + phase OLS
start_date = pd.to_datetime('2010-07-30')
mask = df['date'] > start_date
indices = df[mask].index

# Initialize YHAT with 0
df['YHAT'] = df['logprice']

# Calculate YHAT one row at a time
for i in range(indices[0] + 1, indices[-1] + 1):
    df.at[i, 'YHAT'] = coefs[0] + coefs[1] * df.at[i - 1, 'YHAT'] + coefs[2] * df.at[i, 'phaseplus']

# Step 2: Calculate decayfunc
n = df.index.to_numpy()
df['decayfunc'] = 3 * np.exp(-0.0004 * n) * np.cos(0.005 * n - 1)

# Step 3: Calculate prediction from Step 1 (already in YHAT column)

# Step 4: Add decay to Step 3
df['YHAT'] = df['YHAT'] + df['decayfunc']

# Step 5: Exponentiate
df['eYHAT'] = np.exp(df['YHAT'])


def format_dollars(value, pos, is_minor=False):
    if value >= 1e9:
        s = '${:,.1f}B'.format(value * 1e-9)
    elif value >= 1e6:
        s = '${:,.1f}M'.format(value * 1e-6)
    elif value >= 1e3:
        s = '${:,.0f}K'.format(value * 1e-3)
    else:
        s = '${:,.0f}'.format(value)
    return s
    
# Plot the results
date_20190101 = dt.datetime.strptime('2011-01-01', '%Y-%m-%d')
date_20240101 = dt.datetime.strptime('2028-01-01', '%Y-%m-%d')
plot_df = df[(df['date'] > date_20190101) & (df['date'] < date_20240101)]

# Calculate residuals
residuals = plot_df['logprice'] - plot_df['YHAT']

# Calculate the distribution function using the normal distribution
cdf = norm.cdf(residuals, loc=residuals.mean(), scale=residuals.std())
plot_df = plot_df.assign(cdf=cdf)



# Create a 'signal' column to mark buy and sell points
plot_df['signal'] = np.where(plot_df['cdf'] <= 0.4, 'buy', np.where(plot_df['cdf'] > 0.6, 'sell', 'hold'))

# Filter buy and sell points from the second epoch onward
buy_signal = plot_df[(plot_df['signal'] == 'buy') & (plot_df['epoch'] >= 2)]
sell_signal = plot_df[(plot_df['signal'] == 'sell') & (plot_df['epoch'] >= 2)]


# Prepare the figure with an additional subplot for the CDF
#fig = plt.figure(figsize=(16, 10))

# Main grid
#gs_main = plt.GridSpec(3, 1, height_ratios=[3, 1, 1])  # Change the number of rows and height_ratios

# Side grid
#gs_side = plt.GridSpec(2, 2, width_ratios=[1, 1])
fig, (ax1, ax5, ax6) = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
# Subplots
#ax1 = plt.subplot(gs_main[0])
#ax5 = plt.subplot(gs_main[1])
#ax4 = plt.subplot(gs_main[2])  
#ax2 = plt.subplot(gs_side[0])
#ax3 = plt.subplot(gs_side[1])


line1, = ax1.plot(plot_df['date'], plot_df['PriceUSD'], label='PriceUSD')
line2, = ax1.plot(plot_df['date'], plot_df['eYHAT'], label='BAERM')
ax1.set_yscale('log')
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: format_dollars(y, pos)[0]))
ax1.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda y, pos: format_dollars(y, pos, is_minor=True)[0]))
ax1.tick_params(axis='y', which='minor', labelsize=8)
ax1.set_xlabel('Date')
ax1.set_ylabel('Exchange Rate')
ax1.tick_params(axis='both', which='both', labelsize=12)

# Add buy and sell signals (triangles) to the plot
ax1.scatter(buy_signal['date'], buy_signal['PriceUSD'],
            marker='^', color='g', label='Buy Signal')

ax1.scatter(sell_signal['date'], sell_signal['PriceUSD'],
            marker='v', color='r', label='Sell Signal')




# Add dashed vertical lines at the change of each epoch
epoch_changes = plot_df[plot_df['hindicator'] == True]
for date in epoch_changes['date']:
    ax1.axvline(x=date, color='blue', linestyle='-', linewidth=2)
    
minor_locator = ticker.LogLocator(subs=(0.2,0.4,0.6,0.8))
ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
ax1.tick_params(axis='y', which='minor', labelsize=10)

# Add grid lines to the plot
ax1.grid(True)

# Define function to update the y-axis scale
def update_scale():
    global is_log_scale
    if is_log_scale:
        ax1.set_yscale('log')
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_dollars))
        button_scale.label.set_text('Switch to Linear Scale')
    else:
        ax1.set_yscale('linear')
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'${y:,.0f}'))
        button_scale.label.set_text('Switch to Log Scale')
    fig.canvas.draw()

# Define function to toggle between linear and log scales
def toggle_scale(event):
    global is_log_scale
    is_log_scale = not is_log_scale
    update_scale()

# Add a button to toggle between linear and log scales
ax_button_scale = plt.axes([0.8, 0.025, 0.1, 0.04])
button_scale = Button(ax_button_scale, 'Switch to Log Scale')
is_log_scale = True
button_scale.on_clicked(toggle_scale)
update_scale()

# Plot the residuals vs fitted values
#ax2.scatter(plot_df['eYHAT'], residuals, s=2)
#ax2.axhline(y=0, color='black', linestyle='--')
#ax2.set_xlabel('Fitted Values')
#ax2.set_ylabel('Residuals')

# Plot a histogram of the residuals with overlaid standard normal distribution
#sns.histplot(residuals, kde=True, ax=ax3, stat='density', color='skyblue')
#x = np.linspace(-4, 4, 100)
#ax3.plot(x, stats.norm.pdf(x, loc=residuals.mean(), scale=residuals.std()), color='red')
#ax3.set_xlabel('Residuals')
#ax3.set_ylabel('Density')

#ax3.set_xlim([-4, 4])
#ax3.spines['top'].set_visible(False)
#ax3.spines['right'].set_visible(False)
#ax3.grid(axis='y')

# Plot the probability density function as a time series
#ax4.plot(plot_df['date'], plot_df['cdf'], label='Cumulative Distribution')
#ax4.set_xlabel('Date')
#ax4.set_ylabel('Cumulative Distribution')



# Backtesting
initial_capital = 1000
capital = initial_capital
position = 0

backtest = pd.DataFrame(columns=['date', 'capital', 'position', 'signal'])
# Filter plot_df to only include buy and sell signals
trading_df = plot_df[plot_df['epoch'] >= 2]

n_rows = len(plot_df)
progress_step = n_rows // 50  # Set the progress step to print a hash mark every 2% of progress

print("Backtest progress: ", end="")
for index, row in trading_df.iterrows():
    signal = row['signal']
    price = row['PriceUSD']
    cdf_value = row['cdf']
    
    if signal == 'buy' and capital > 0:
        position_size = 1-cdf_value
        position += capital * position_size / price
        capital -= capital * position_size

    elif signal == 'sell' and position > 0:
        position_size = cdf_value
        sell_amount = position * position_size
        capital += sell_amount * price
        position -= sell_amount

    backtest = pd.concat([backtest, pd.DataFrame({
        'date': [row['date']],
        'capital': [capital],
        'position': [position],
        'signal': [signal],
        'PriceUSD': [price]
    })], ignore_index=True)
    # Print a hash mark for every progress_step
    if index % progress_step == 0:
        print("#", end="", flush=True)

print(" Done.")

# Calculate portfolio value
backtest['value'] = backtest['capital'] + backtest['position'] * backtest['PriceUSD']

# Plot backtest results
# Plot backtest results below ax1
ax5.plot(backtest['date'], backtest['capital'], label='Capital')
ax5.set_xlabel('Date')
ax5.set_ylabel('USD Owned')
ax5.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: format_dollars(y, pos)))
ax5.grid(True)
ax5.legend()

ax6.plot(backtest['date'], backtest['position'], label='BTC Value')
ax6.set_xlabel('Date')
ax6.set_ylabel('BTC Owned')
ax6.grid(True)
ax6.legend()

plt.tight_layout(pad=2.0, h_pad=2.0)
plt.show()

