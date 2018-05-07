#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import statsmodels.tsa.stattools.coint as coint

tickers = ['AAPL', 'MSFT']

def get_price_data(tickers, start):
    data = web.DataReader(tickers, 'yahoo', start)
    return data['Close']

def get_correlation_matrix(price_data, lookback = 252): 
    returns = price_data.apply(np.log).diff()
    corr_matrix = returns.iloc[-lookback:].corr().values
    return corr_matrix

def get_hist_returns(price_data):
    lookback = 30
    log_prices = price_data.apply(np.log)
    return log_prices.diff(lookback).iloc[-1].values

def get_vols(price_data):
    lookback = 30
    log_prices = price_data.apply(np.log)
    returns = log_prices.diff()
    return np.sqrt(252)*returns.rolling(window=lookback).std().iloc[-1].values

start = '2015-01-01'
p = get_price_data(tickers, start)
r = coint(p)

print("bonjour")