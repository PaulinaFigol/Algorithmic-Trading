import pandas as pd
import matplotlib.pyplot as plt
import warnings
import pandas as pd
from sklearn import preprocessing
import seaborn as sns
from matplotlib import pyplot
import numpy as np
import plotly.express as px
import re as re


def macd(price, span_long, span_short):
    """
    Moving average convergence divergence (MACD) is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price.
    
    :param price: price series of a security's price
    :param span_long: number of periods for the long moving average
    :param span_short: number of periods for the short moving average
    :return: difference between two moving averages of a security's price
    """
    # Get the long EMA of the closing price
    s = price.ewm(span=span_short, adjust=False, min_periods=span_short).mean()
    # Get the short EMA of the closing price
    l = price.ewm(span=span_long, adjust=False, min_periods=span_long).mean()
    # Subtract the long EMA from the short EMA to get the MACD
    macd_calc = s - l
    return macd_calc


def RSI(price, period, ema = True):
    """
    The Relative Strength Index is a technical indicator that shows when a financial product is overbought or oversold to determine entry or exit position.
    
    :param price: price series of a security's price
    :param period: number of periods for the rolling window
    :return: a pd.Series with the relative strength index.
    """
    close_delta = price.diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    if ema == True:
    # Use exponential moving average
        ma_up = up.ewm(com = period - 1, adjust=True, min_periods = period).mean()
        ma_down = down.ewm(com = period - 1, adjust=True, min_periods = period).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window = period, adjust=False).mean()
        ma_down = down.rolling(window = period, adjust=False).mean()
        
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi


#Plot RSI against the stock closing price
def RSI_visualise(start, end, data, stock, period):
    """
    Plot RSI against the stock's closing price
    
    :param start: start date
    :param end: end date
    :param data: dataframe including stock's closing price
    :param stock: stock's closing price
    :param period: number of periods/rolling window to calculate RSI
    :return: drawn an overlap between stock's price against its RSI
    """  
    #Calculate RSI
    RSI_test = pd.DataFrame(RSI(data[stock], period))
    RSI_test.columns = ['RSI']
       
    #Join with the current closing price  
    both = RSI_test.merge(data[stock], left_on =RSI_test.index ,right_on = data.index)
    both['date'] = both['key_0']
    
    #Include only data between start and end date
    start_date = both["date"] >= start
    end_date = both["date"] <= end
    between_two_dates = start_date & end_date
    df_filtered = both.loc[between_two_dates].reset_index()
    df_filtered['date'] = pd.to_datetime(df_filtered['date'])
    #Plot
    fig, ax1 = plt.subplots(figsize=(15,5))
    ax2 = ax1.twinx()
    ax1.plot(df_filtered['date'], df_filtered['RSI'], 'g-')
    ax2.plot(df_filtered['date'], df_filtered[stock], 'b-')
    ax1.set_ylabel('RSI', color='g')
    ax2.set_ylabel('stock', color='b')
    
    plt.show()
    
    
def average_trade_returns(data):
    """
    Calculate average trade returns & append dataframe with trade returns and days when the returns appeared
    :param data: dataframe
    :return: return datadrame including trade returns and when they occured (number of trading days after purchase)
    """    
    #Create empty dictionary and columns in the data frame
    price_ch_days = {}
    data['trade returns'] = np.nan
    data['returns_day'] = np.nan
    # Loop over 10 days returns and calculate the price change
    for n in range(1,11):
        price_ch_days['price_change_rate_'+str(n)] = -(data.groupby(by="stock", dropna=False)[['price']].diff(-n))
        data['price_change_rate_'+str(n)] = price_ch_days['price_change_rate_'+str(n)]
        #Calculations done separately as it takes much less time when the column is already in the data frame
        data['price_change_rate_'+str(n)] = data['price_change_rate_'+str(n)]/data['price']*100

    price_change_days = price_ch_days.keys()
    
    #only include stocks with 'buy' signal based on the prediction
    df = data[data['y_pred']==1]
    #look for days when price increased by minimum of 5% (1-10 days post purchase)
    p = df[price_change_days].apply(lambda row: row[row >=5].index, axis=1) 
   
    idx = [p[i].empty for i in p.index]
    #only include the first date post purchase when return reached >=5% 
    df.loc[~np.array(idx),'returns_day'] = [p[rows][0] for rows in df[~np.array(idx)].index]
    #if the above doesn't exist sell on day 10
    df.loc[np.array(idx),'returns_day'] = 'price_change_rate_10'
    #map trade return with the return day
    df['trade returns'] = [df.loc[rows,df.loc[rows,'returns_day']] for rows in df.index] 
    #replace nan with the trade return and return date in the data frame
    data.loc[df.index,['trade returns', 'returns_day']] = df.loc[df.index,['trade returns', 'returns_day']]
    #return a median and mean percentage return
    print("Median percentage return: %.2f%%" % (df['trade returns'].median()))
    print("Mean percentage return: %.2f%%" % (df['trade returns'].mean()))
    