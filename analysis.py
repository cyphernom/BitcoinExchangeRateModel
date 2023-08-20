#(c)2020-2023 Nick Btconometrics
import numpy as np
import datetime as dt
from scipy.stats import binom

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

    def percentile(self, x, p):
        if p >= 1:
            return np.max(x)
        if p <= 0:
            return np.min(x)
        else:
            n = len(x)
            r = (1 + p * (n - 1)) - 1
            rfrac = r % 1
            rint = int(r // 1)
            x.sort()
            xp = (1 - rfrac) * x[rint] + rfrac * x[rint + 1]
            return xp

    def binominv(self, n, p, a):
        low = 0
        high = 1000000000000  # big enough for most purposes!

        while low <= high:
            mid = (low + high) // 2
            result = binom.cdf(mid, n, p)
            if result > a:
                high = mid - 1
            else:
                low = mid + 1

        return low

    def upperCI(self, alpha, p, x):
        n = len(x)
        return percentile(x, binominv(n, p, 1 - alpha / 2) / n)

    def lowerCI(self, alpha, p, x):
        n = len(x)
        return percentile(x, binominv(n, p, alpha / 2) / n)
