# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 18:36:37 2020

@author: user
"""

import pandas as pd
import numpy as np
from talib import RSI, BBANDS
import matplotlib.pyplot as plt
import datetime

import pandas_datareader as pdr 
df = pdr.get_data_yahoo('AAPL')
df1 = df[df['Volume']>0]
data_source = r'C:\Users\user\Desktop\Grad\PY Modules\AAPL.xlsx'
df1.to_excel(data_source)
import pandas as pd 
import numpy as np
df = pd.read_excel(r'C:\Users\user\Desktop\Grad\PY Modules\AAPL.xlsx')
df.head()

start = '2015-06-29'
end = '2018-12-31'
symbol = 'AAPL'
max_holding = 100

price = df.iloc[::-1]
price = price.dropna()
close = price['Adj Close'].values

up, mid, low = BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
rsi = RSI(close, timeperiod=14)
print("RSI (first 10 elements)\n", rsi[14:24])

def bbp(price):
    up, mid, low = BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    bbp = (price['Adj Close'] - low) / (up - low)
    return bbp
price

isHoldingFull = False
holdings = pd.DataFrame(index=price.index, data={'Holdings': np.array([np.nan] * index.shape[0])})
holdings.loc[((price['RSI'] < 30) & (price['BBP'] < 0)), 'Holdings'] = max_holding
holdings.loc[((price['RSI'] > 70) & (price['BBP'] > 1)), 'Holdings'] = 0
holdings.ffill(inplace=True)
holdings.fillna(0, inplace=True)
