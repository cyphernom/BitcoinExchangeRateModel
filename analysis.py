#(c)2020-2023 Nick Btconometrics
import numpy as np
import datetime as dt

class Analysis:
    def __init__(self, df):
        self.df = df

    def calculate_oos_r_squared(self, actual, predicted):
        residuals = actual - predicted
        SSR = np.sum(residuals ** 2)
        SST = np.sum((actual - actual.mean()) ** 2)
        return 1 - SSR / SST

    def calculate_oos_r_squared_values(self):
        oos_mask = self.df['epoch'] >= 2
        oos_actual = self.df.loc[oos_mask, 'logprice']
        oos_predicted = self.df.loc[oos_mask, 'YHAT']
        oos_predicteds2 = self.df.loc[oos_mask, 'YHATs2']

        R2_oos = self.calculate_oos_r_squared(oos_actual, oos_predicted)
        R2_oos2 = self.calculate_oos_r_squared(oos_actual, oos_predicteds2)

        return R2_oos, R2_oos2

    def calculate_rolling_r_squared(self):
        self.df['rolling_r_squared'] = np.nan  # Create the 'rolling_r_squared' column and fill with NaN
        oos_mask = (self.df['epoch'] >= 2) & (self.df['date'] <= dt.datetime.now())

        for i in self.df.index:
            if oos_mask[i]:  # Update the rolling R-squared only where epoch >= 2
                oos_actual = self.df.loc[oos_mask & (self.df.index <= i), 'logprice']
                oos_predicted = self.df.loc[oos_mask & (self.df.index <= i), 'YHATs2']
                R2_oos = self.calculate_oos_r_squared(oos_actual, oos_predicted)
                self.df.at[i, 'rolling_r_squared'] = R2_oos

        return self.df['rolling_r_squared'].tolist()