#(c)2020-2023 Nick Btconometrics
import numpy as np

class Model:
    def __init__(self, df):
        self.df = df
        self.coefs = None
        
    #PRE: Y is a 1D numpy array and X is a 2D numpy array.
    def mOLS(self, Y, X):
    # Check for NaN values
        if np.isnan(X).any() or np.isnan(Y).any():
            raise ValueError("NaN values detected.")

    # Check for constant columns
        if np.any(np.ptp(X, axis=0) == 0):
            raise ValueError("Constant columns detected.")

    # Check for enough observations
        if X.shape[0] <= X.shape[1]:
            raise ValueError("Insufficient observations.")

        n = len(X)
        constant = np.ones(n)
        x = np.column_stack((constant, X))
        xtx = x.T @ x

        try:
            xtxinv = np.linalg.inv(xtx)
        except np.linalg.LinAlgError:
            raise ValueError("Singular matrix.")

     # Using np.linalg.lstsq to directly solve the least squares problem
        betahat, residuals, _, _ = np.linalg.lstsq(x, Y, rcond=None)

        yhat = x @ betahat
        residuals = Y - yhat
        SSE = residuals @ residuals
        hatmatrix = x @ np.linalg.lstsq(x.T @ x, x.T, rcond=None)[0]
        leverage = np.diag(hatmatrix)
        ybar = np.mean(Y)
        SST = np.sum((Y - ybar) ** 2)
        R2 = 1 - SSE / SST
        SSR = SST - SSE
        AIC = n * np.log(SSE / n) + (n + X.shape[1]) / (1 - (X.shape[1] + 2) / n)
       
        #xtxinvxt = xtxinv @ x.T
        #betahat = xtxinvxt @ Y
        #hatmatrix = x @ xtxinvxt
        #yhat = hatmatrix @ Y
        #leverage = np.diag(hatmatrix)
        #residuals = Y - yhat
        #SSE = np.sum(residuals**2)
        #ybar = np.mean(Y)
        #SST = np.sum((ybar - Y)**2)
        #R2 = 1 - SSE / SST
        #AIC = n * np.log(SSE / n) + (n + X.shape[1]) / (1 - (X.shape[1] + 2) / n)
        #SSR = SST - SSE

        lm = {
            'betahat': betahat,
            'yhat': yhat,
            'fitted': yhat,
            'hat_matrix': hatmatrix,
            'leverage': leverage,
            'residuals': residuals,
            'SSE': SSE,
            'SSR': SSR,
            'AIC': AIC,
            'x': x,
            'y': Y,
            'SST': SST,
            'R2': R2
        }

        return lm


    def run_regression(self):
        mask = self.df['epoch'] < 2
        reg_X = self.df.loc[mask, ['logprice', 'phaseplus']].shift(1).iloc[1:].values
        reg_y = self.df.loc[mask, 'logprice'].iloc[1:].values
        result = self.mOLS(reg_y, reg_X)
        self.coefs = result['betahat']
        return self.coefs, result



    def calculate_YHAT(self):
        self.df['YHAT'] = self.df['logprice']
        for i in range(self.df.index[0] + 1, self.df.index[-1] + 1):
            self.df.at[i, 'YHAT'] = self.coefs[0] + self.coefs[1] * self.df.at[i - 1, 'YHAT'] + self.coefs[2] * self.df.at[i, 'phaseplus']

    def calculate_YHATs2(self):
        n = self.df.index.to_numpy()
        self.df['decayfunc'] = 3 * np.exp(-0.0004 * n) * np.cos(0.005 * n - 1)
        self.df['YHATs2'] = self.df['YHAT'] + self.df['decayfunc']
        self.df['eYHAT'] = np.exp(self.df['YHAT'])
        self.df['eYHATs2'] = np.exp(self.df['YHATs2'])

    #lets study the autocorrelation coefficient
    def study_coefficient_evolution(self):
        coefficients = []
        dates = []
        
        for i in range(self.df.index[0], self.df.index[-1] + 1):
            mask = self.df.index <= i
            reg_X = self.df.loc[mask, ['logprice', 'phaseplus']].shift(1).iloc[1:].values
            reg_y = self.df.loc[mask, 'logprice'].iloc[1:].values

            try:
                result = self.mOLS(reg_y, reg_X)
                coefficients.append(result['betahat'][1])
                dates.append(self.df.at[i, 'date'])  
            except ValueError as e:
                print(f"Skipping iteration {i} due to {e}")
                continue

        return coefficients, dates

    #lets calculate YHATs for reg's on all dates. 
    #this function should: run the regression mOLS for each datapoint in the self.df (as per the study_coefficient_evolution method above) 
    #and then calculate the YHAT (that is the predicted values) for each regression, with a view to plotting the results
    def study_yhats(self):
        all_yhats = []
        dates = []
        n = self.df.index.to_numpy()

        for i in range(self.df.index[0], self.df.index[-1] + 1):
            mask = self.df.index <= i
            reg_X = self.df.loc[mask, ['logprice', 'phaseplus']].shift(1).iloc[1:].values
            reg_y = self.df.loc[mask, 'logprice'].iloc[1:].values

            try:
                result = self.mOLS(reg_y, reg_X)
                coefs = result['betahat']
            
                # Calculate YHAT using the recursive formula
                yhat = np.zeros_like(reg_y)
                yhat[0] = reg_y[0]
                for j in range(1, len(yhat)):
                    yhat[j] = coefs[0] + coefs[1] * yhat[j - 1] + coefs[2] * reg_X[j - 1, 1]

                # Apply decay function as in calculate_YHATs2
                decayfunc = 3 * np.exp(-0.0004 * n[:len(yhat)]) * np.cos(0.005 * n[:len(yhat)] - 1)
                yhat_s2 = yhat + decayfunc
                all_yhats.append(yhat_s2) # Adding all the adjusted predicted values for this regression

                # Add the corresponding dates for this yhat, making sure the lengths match
                current_dates = [self.df.at[idx, 'date'] for idx in range(self.df.index[0], self.df.index[0] + len(yhat))]
                dates.append(current_dates)
            except ValueError as e:
                print(f"Skipping iteration {i} due to {e}")
                continue

        return all_yhats, dates
       






        
