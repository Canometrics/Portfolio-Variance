import yfinance as yf # to fetch up to date stock history
import pandas as pd # for dataframes
import matplotlib.pyplot as plt # for graphing
import numpy as np # for mathmatical operations
import datetime as dt # for time info

# calculate sd for each ticker
# maybe some graphs or smth

def get_portfolio(file): # read csv file in given format
    portfolio = pd.DataFrame(pd.read_csv(file))
    return portfolio

def get_date(): # get today's date and date going back 365 days
    end = dt.date.today()
    start = dt.date.today() - dt.timedelta(days = 365)
    return end, start

def get_lnrets(ticks): # get natural logged returns
    end, start = get_date() # get timeframe
    lnrets = pd.DataFrame() # store lnrets here

    for i in ticks:
        if i == "FCASH": # since return of cash is zero, don't attempt to fetch it from yfinance
            lnrets[i] = 0
        else: 
            hist = pd.DataFrame(yf.download(i, start, end, progress = False)["Adj Close"]) # get history for each stock
            lnrets[i] = np.log(hist["Adj Close"]/hist["Adj Close"].shift(1)) # calculate ln return (np.log has base e by default)
    lnrets = pd.DataFrame(lnrets.dropna()) # return calculation will result in NaN cells by its nature, drop them
    return lnrets

def get_indsd(ticks, lnrets): # get standard deviation of each individual stock
    sd = pd.DataFrame(columns = ["Ticker", "Volatility"]) # store sd's here
    
    for i in ticks:
        sd.loc[len(sd.index)] = [i, lnrets[i].std() * np.sqrt(252)] # get sd (volatility) of each stock
    return sd
    

def get_covmatrix(lnrets): # get covariance matrix showing covarience of every stock with every other stock
    covmatrix = pd.DataFrame(lnrets.cov())
    return covmatrix

def get_portsd(port, lnrets): # get portfolio standard deviation (volatility)
    cov = get_covmatrix(lnrets) # get covariance matrix

    weight_T = pd.DataFrame(port["Weight"]).T # transpose weights once for the formula
    weight = pd.DataFrame(port["Weight"]) # untransposed weights
    
    portsd = np.sqrt((weight_T @ cov.values @ weight).values.item() * 252) # apply portfolio variance formula, 252 trading days in a year
    return portsd

def main():

    portfolio = get_portfolio(input("Portfolio File Name: ")) # input user for portfolio csv file destination
    lnrets = get_lnrets(portfolio["Ticker"])

    print(get_indsd(portfolio["Ticker"], lnrets))

    print('Portfolio Volatility:', get_portsd(portfolio, lnrets)) # print portfolio variation

if __name__ == "__main__":
    main()