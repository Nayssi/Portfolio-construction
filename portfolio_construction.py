#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import utilities
from scipy.optimize import minimize
import datetime

def optimize_portfolio(tickers, total_allocation, start_date = datetime.datetime(2010,1,1), previous_weights = None):
    """
    Core portfolio construction function.
    tickers is a list of tickers.
    total_allocation is the amount (in the base currency) allocated to
    previous_weights is a dictionary of weights allocated to a subset of tickers provided in "tickers", whose sum is less than 1.
    
    """
    n = len(tickers) #Number of securities
    
    #Define the hyperparameters
    gamma = 1.2 # Volatility search tolerance
    lbda = 1. #Tolerance to transaction costs
    
    #Get the data from Yahoo! Finance and preprocess it.
    price_data = utilities.get_price_data(tickers, start_date)
    hist_returns = utilities.get_hist_returns(price_data)
    hist_vols = utilities.get_vols(price_data)
    corr_matrix = utilities.get_correlation_matrix(price_data)
    
    #Build the initial objective function
    portfolio_variance = lambda w : w @ (hist_vols.T * corr_matrix * hist_vols) @ w.T
    
    #Define the search space
    sum_constraint = {'type':'ineq','fun': lambda w: 1-np.sum(w)}
    bounds = [(0, 0.15) for k in range(n)]
    
    #Dummy initialization
    w0 = 1./n * np.ones(n)
    
    #Find the initial minimum variance portfolio.
    min_variance_portfolio = minimize(portfolio_variance,w0,constraints=(sum_constraint), bounds=bounds).x
    min_variance = min_variance_portfolio @ (hist_vols.T * corr_matrix * hist_vols) @ min_variance_portfolio
    
    #Build the variance and turnover constraints.
    variance_constraint = {'type':'ineq','fun': lambda w: gamma*min_variance - portfolio_variance(w)} #Limit the overall variance of the portfolio.
    
    if previous_weights == None:
        previous_weights = np.zeros(n)
    else:
        previous_weights = utilities.dict_to_weight_vector(tickers, previous_weights)
    
    #Define the proper objective function
    objective_function = lambda w : - np.dot(hist_returns, w.T) + lbda * utilities.transaction_cost(previous_weights, w,  total_allocation)
    constraints = (variance_constraint, sum_constraint)
    
    optimal_portfolio = minimize(objective_function,w0,constraints=constraints, bounds=bounds).x
    return utilities.weight_vector_to_dict(tickers, optimal_portfolio)

#Define the portfolio
tickers = ['BABA', 'JD', 'TTWO']
previous = {'BABA': .5, 'JD': .2}
total_allocation = 5e4 #€50k to allocate
portfolio1 = optimize_portfolio(tickers, total_allocation)
portfolio2 = optimize_portfolio(tickers, total_allocation, previous_weights=previous)
print(portfolio1)
print(portfolio2)