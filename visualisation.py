#(c)2020-2023 Nick Btconometrics
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.widgets import Button
from scipy.stats import norm
import numpy as np
import datetime as dt

class Visualisation:
    def __init__(self, df):
        self.df = df
        self.is_log_scale = True

    @staticmethod
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

    def plot_charts(self, plot_df):
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
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: self.format_dollars(y, pos)))
                ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda y, pos: self.format_dollars(y, pos, is_minor=True)))
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
        # Define function to update the y-axis scale
        def update_scale():
            if self.is_log_scale:
                ax.set_yscale('log')
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(self.format_dollars))
                button_scale.label.set_text('Switch to Linear Scale')
            else:
                ax.set_yscale('linear')
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'${y:,.0f}'))
                button_scale.label.set_text('Switch to Log Scale')
            fig.canvas.draw()

        # Define function to toggle between linear and log scales
        def toggle_scale(event):
            self.is_log_scale = not self.is_log_scale
            update_scale()

        # Add a button to toggle between linear and log scales
        ax_button_scale = plt.axes([0.8, 0.025, 0.1, 0.04])
        button_scale = Button(ax_button_scale, 'Switch to Log Scale')
        button_scale.on_clicked(toggle_scale)
        update_scale()  # Call the update_scale function to initialize the scale

    
        plt.show()
                        
    def plot_rolling_r_squared(self, df):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['date'], df['rolling_r_squared'], label='Rolling Out-of-sample R-squared', color='blue')
        ax.set_xlabel('Date')
        ax.set_ylabel('Rolling Out-of-sample R-squared')
        ax.set_title('Rolling Out-of-sample R-squared')
        ax.set_ylim(0, 1)  # Force the y-scale to be exactly 0 to 1
        ax.legend()
        ax.grid(True)

        # Add dashed vertical lines at the change of each epoch for the relevant epochs
        epoch_changes = df[(df['hindicator'] == True) & (df['epoch'] >= 2) & (df['date'] <= dt.datetime.now())]

        for date in epoch_changes['date']:
            ax.axvline(x=date, color='red', linestyle='--', linewidth=1)

        plt.show()

    def plot_coefficient_evolution(self, coefficients, dates):
        plt.plot(dates, coefficients)
        plt.title('Evolution of the Autocorrelation Coefficient Over Time')
        plt.xlabel('Date')
        plt.ylabel('Coefficient Value')
        plt.xticks(rotation=45)  # Optional: rotate the x-axis labels for better visibility
        plt.show()
    
    def plot_all_yhats_with_rainbow(self, all_yhats, dates):
        gs_main = plt.GridSpec(1, 1)
        fig, ax = plt.subplots(figsize=(8, 6))

        plot_df = self.df

        # Calculate residuals
        residuals = plot_df['logprice'] - plot_df['YHATs2']
        residuals_std = np.std(residuals)

        # Calculate likelihood thresholds
        likelihoods = np.arange(0.3, .7, 0.01)
        thresholds = norm.ppf(likelihoods, loc=0, scale=residuals_std)

        # Calculate upper and lower bounds for each likelihood threshold
        ub_lines = plot_df['YHATs2'].to_numpy()[:, np.newaxis] + thresholds
        lb_lines = plot_df['YHATs2'].to_numpy()[:, np.newaxis] - thresholds

        ax.plot(plot_df['date'], plot_df['PriceUSD'], label='PriceUSD')
        ax.plot(plot_df['date'], plot_df['eYHATs2'], label='Phased BAERM')

        # Plotting all the new y-hats
        for i, yhat in enumerate(all_yhats):
            color = plt.cm.rainbow(i / len(all_yhats))  # Generating colors using a colormap
            current_dates = dates[:len(yhat)]
            ax.plot(current_dates, yhat, label=f'Regression {i+1}' if i < 3 else "", alpha=0.5, color=color)

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
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: self.format_dollars(y, pos)))
        ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda y, pos: self.format_dollars(y, pos, is_minor=True)))
        ax.tick_params(axis='y', which='minor', labelsize=8)
        ax.set_xlabel('Date')
        ax.set_ylabel('Exchange Rate')
        ax.tick_params(axis='both', which='both', labelsize=12)
    
        # Add dashed vertical lines at the change of each epoch
        epoch_changes = plot_df[plot_df['hindicator'] == True]
        for date in epoch_changes['date']:
            ax.axvline(x=date, color='blue', linestyle='-', linewidth=2)
    
        # Add grid lines to the plot
        ax.grid(True)
    
        # Define function to update the y-axis scale
        def update_scale():
            if self.is_log_scale:
                ax.set_yscale('log')
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(self.format_dollars))
                button_scale.label.set_text('Switch to Linear Scale')
            else:
                ax.set_yscale('linear')
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'${y:,.0f}'))
                button_scale.label.set_text('Switch to Log Scale')
            fig.canvas.draw()
    
        # Define function to toggle between linear and log scales
        def toggle_scale(event):
            self.is_log_scale = not self.is_log_scale
            update_scale()
    
        # Add a button to toggle between linear and log scales
        ax_button_scale = plt.axes([0.8, 0.025, 0.1, 0.04])
        button_scale = Button(ax_button_scale, 'Switch to Log Scale')
        button_scale.on_clicked(toggle_scale)
        update_scale()  # Call the update_scale function to initialize the scale
    
        plt.show()




        
