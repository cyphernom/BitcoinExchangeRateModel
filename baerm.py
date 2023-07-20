#(c)2020-2023 Nick Btconometrics
import pandas as pd
import numpy as np
import statsmodels.api as sm
import datetime as dt
from scipy.stats import norm
from scipy import stats
from matplotlib.widgets import Button
from math import exp, cos, log
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter

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


def plot_charts(plot_df):
    global is_log_scale
    gs_main = plt.GridSpec(1, 1)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Calculate residuals
    residuals = plot_df['logprice'] - plot_df['YHATs2']
    residuals_std = np.std(residuals)

    # Calculate likelihood thresholds
    likelihoods = np.arange(0.3, .7, 0.01)
    thresholds = norm.ppf(likelihoods, loc=0, scale=residuals_std)


    # Calculate upper and lower bounds for each likelihood threshold
    ub_lines = plot_df['YHATs2'].to_numpy()[:, np.newaxis] + thresholds
    lb_lines = plot_df['YHATs2'].to_numpy()[:, np.newaxis] - thresholds

    line1, = ax.plot(plot_df['date'], plot_df['PriceUSD'], label='PriceUSD')
    line3, = ax.plot(plot_df['date'], plot_df['eYHATs2'], label='Phased BAERM')
    
    # Define a colormap for the filled areas
    colormap = plt.cm.get_cmap('rainbow')
    num_colors = len(likelihoods)
    color_values = np.linspace(0, 1, num_colors)
    
    # Plot the lines representing different likelihoods and fill the gaps with colours
    for i, likelihood in enumerate(likelihoods):
        color = colormap(color_values[i])

        if i < num_colors - 1:
            next_color = colormap(color_values[i + 1])
            ax.fill_between(
                plot_df['date'],
                np.exp(ub_lines[:, i]),
                np.exp(ub_lines[:, i + 1]),
                where=(np.exp(ub_lines[:, i]) >= np.exp(lb_lines[:, i])) & (np.exp(ub_lines[:, i + 1]) >= np.exp(lb_lines[:, i + 1])),
                interpolate=True,
                color=color,
                alpha=0.3,
            )
            ax.fill_between(
                plot_df['date'],
                np.exp(lb_lines[:, i]),
                np.exp(lb_lines[:, i + 1]),
                where=(np.exp(ub_lines[:, i]) >= np.exp(lb_lines[:, i])) & (np.exp(ub_lines[:, i + 1]) >= np.exp(lb_lines[:, i + 1])),
                interpolate=True,
                color=color,
                alpha=0.3,
            )
        
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: format_dollars(y, pos)[0]))
    ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda y, pos: format_dollars(y, pos, is_minor=True)[0]))
    ax.tick_params(axis='y', which='minor', labelsize=8)
    ax.set_xlabel('Date')
    ax.set_ylabel('Exchange Rate')
    ax.tick_params(axis='both', which='both', labelsize=12)

    # Add dashed vertical lines at the change of each epoch
    epoch_changes = plot_df[plot_df['hindicator'] == True]
    for date in epoch_changes['date']:
        ax.axvline(x=date, color='blue', linestyle='-', linewidth=2)
        
    minor_locator = ticker.LogLocator(subs=(0.2, 0.4, 0.6, 0.8))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.tick_params(axis='y', which='minor', labelsize=10)

    # Add grid lines to the plot
    ax.grid(True)


    
    # Define function to update the y-axis scale
    def update_scale():

        if is_log_scale:
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_dollars))
            button_scale.label.set_text('Switch to Linear Scale')
        else:
            ax.set_yscale('linear')
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'${y:,.0f}'))
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

    plt.show()

def calculate_oos_r_squared(actual, predicted):
    residuals = actual - predicted
    SSR = np.sum(residuals ** 2)
    SST = np.sum((actual - actual.mean()) ** 2)
    return 1 - SSR / SST

def run_regression(df):
    mask = df['epoch'] < 2
    reg_X = df.loc[mask, ['logprice', 'phaseplus']].shift(1).iloc[1:]
    reg_y = df.loc[mask, 'logprice'].iloc[1:]
    reg_X = sm.add_constant(reg_X)
    ols = sm.OLS(reg_y, reg_X).fit()
    coefs = ols.params.values
    return coefs, ols.summary()

# Function to calculate YHAT one row at a time
def calculate_YHAT(df, coefs):
	df['YHAT'] = df['logprice']
	for i in range(df.index[0] + 1, df.index[-1] + 1):
		df.at[i, 'YHAT'] = coefs[0] + coefs[1] * df.at[i - 1, 'YHAT'] + coefs[2] * df.at[i, 'phaseplus']



def main():
	#important dates:
	phaseplus_start_date = pd.to_datetime('2008-10-31')
	ar_start_date = pd.to_datetime('2010-07-30')
	plot_start_date = dt.datetime.strptime('2011-01-01', '%Y-%m-%d')
	plot_end_date = dt.datetime.strptime('2028-01-01', '%Y-%m-%d')  
	
	# Import data
	url = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
	df = pd.read_csv(url, low_memory=False)
	
	#data pre-processing
	df['date'] = pd.to_datetime(df['time'], format='%Y-%m-%d')
	df.drop('time', axis=1, inplace=True)
	df = pd.concat([df, pd.DataFrame({'date': pd.date_range(df['date'].max() + pd.Timedelta(days=1), periods=15000, freq='D')})], ignore_index=True)
	df.loc[df['BlkCnt'].isnull(), 'BlkCnt'] = 6 * 24
	df['sum_blocks'] = df['BlkCnt'].cumsum()
	df['hp_blocks'] = df['sum_blocks'] % 210001
	df['hindicator'] = (df['hp_blocks'] < 200) & (df['hp_blocks'].shift(1) > 209000)
	df['epoch'] = df['hindicator'].cumsum() 
	df['reward'] = 50 / (2 ** df['epoch'].astype(float))
	df.loc[df['epoch'] >= 33, 'reward'] = 0
	df['daily_reward'] = df['BlkCnt'] * df['reward']
	df['tsupply'] = df['daily_reward'].cumsum()
	df['logprice'] = np.log(df['PriceUSD'])
	
	df['days_since_start'] = (df['date'] - phaseplus_start_date).dt.days
	df['phaseplus'] = df['reward'] - (df['epoch'] + 1) ** 2
	df = df[df['date'] >= dt.datetime.strptime('2010-07-18', '%Y-%m-%d')].reset_index(drop=True)

	coefs, ols_summary = run_regression(df)

	print(coefs)
	print(ols_summary)

	#AR method:
	# Step 1: Calculate AR + phase OLS	
	mask = df['date'] > ar_start_date
	indices = df[mask].index

	calculate_YHAT(df, coefs)
	
	# Step 2: Calculate decayfunc
	n = df.index.to_numpy()
	df['decayfunc'] = 3 * np.exp(-0.0004 * n) * np.cos(0.005 * n - 1)
	
	# Step 3: Calculate prediction from Step 1 (already in YHAT column)
	# Step 4: Add decay to Step 3# Calculate the stock-to-flow ratio for each row

	df['YHATs2'] = df['YHAT'] +df['decayfunc']

	# Sep 5: Exponentiate
	df['eYHAT'] = np.exp(df['YHAT'])
	df['eYHATs2'] = np.exp(df['YHATs2'])

	# Calculate the out-of-sample R-squared
	oos_mask = df['epoch'] >= 2
	oos_actual = df.loc[oos_mask, 'logprice']
	oos_predicted = df.loc[oos_mask, 'YHAT']
	oos_predicteds2 = df.loc[oos_mask, 'YHATs2']

	R2_oos = calculate_oos_r_squared(oos_actual, oos_predicted)
	R2_oos2 = calculate_oos_r_squared(oos_actual, oos_predicteds2)


	print(f"Out-of-sample R-squared\n Base Model: {R2_oos} \n Phased Model: {R2_oos2}")

  
	plot_df= df[(df['date'] > plot_start_date) & (df['date'] < plot_end_date)]

	plot_charts(plot_df)

if __name__ == "__main__":
    main()
