#(c)2020-2023 Nick Btconometrics
import datetime as dt
import numpy as np
import pandas as pd

class DataPreprocessing:
    def __init__(self, url):
        self.url = url
        self.df = None

    def load_and_preprocess_data(self):
        self.df = pd.read_csv(self.url, low_memory=False)
        self.df['date'] = pd.to_datetime(self.df['time'], format='%Y-%m-%d')
        self.df.drop('time', axis=1, inplace=True)
        self.df = pd.concat([self.df, pd.DataFrame({'date': pd.date_range(self.df['date'].max() + pd.Timedelta(days=1), periods=15000, freq='D')})], ignore_index=True)
        self.df.loc[self.df['BlkCnt'].isnull(), 'BlkCnt'] = 6 * 24
        self.df['sum_blocks'] = self.df['BlkCnt'].cumsum()
        self.df['hp_blocks'] = self.df['sum_blocks'] % 210001
        self.df['hindicator'] = (self.df['hp_blocks'] < 200) & (self.df['hp_blocks'].shift(1) > 209000)
        self.df['epoch'] = self.df['hindicator'].cumsum()
        self.df['reward'] = 50 / (2 ** self.df['epoch'].astype(float))
        self.df.loc[self.df['epoch'] >= 33, 'reward'] = 0
        self.df['daily_reward'] = self.df['BlkCnt'] * self.df['reward']
        self.df['tsupply'] = self.df['daily_reward'].cumsum()
        self.df['logprice'] = np.log(self.df['PriceUSD'])
        self.df['days_since_start'] = (self.df['date'] - pd.to_datetime('2008-10-31')).dt.days
        self.df['phaseplus'] = self.df['reward'] - (self.df['epoch'] + 1) ** 2
        self.df = self.df[self.df['date'] >= dt.datetime.strptime('2010-07-18', '%Y-%m-%d')].reset_index(drop=True)