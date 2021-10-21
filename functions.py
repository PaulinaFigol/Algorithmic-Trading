import pandas as pd
import matplotlib.pyplot as plt
import warnings
import pandas as pd
from sklearn import preprocessing
import seaborn as sns
from matplotlib import pyplot
import numpy as np
import plotly.express as px



#Develop MACD function
def macd(price, span_long, span_short):
    # Get the long EMA of the closing price
    s = price.ewm(span=span_short, adjust=False, min_periods=span_short).mean()
    # Get the short EMA of the closing price
    l = price.ewm(span=span_long, adjust=False, min_periods=span_long).mean()
    # Subtract the long EMA from the short EMA to get the MACD
    macd_calc = s - l
    return macd_calc


#Develop RSI function
def RSI(series, period, ema = True):
    """
    Returns a pd.Series with the relative strength index.
    """
    close_delta = series.diff()

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
def RSI_visualise(before_end_date, after_start_date, data, stock, period):
    
    RSI_test = pd.DataFrame(RSI(data[stock], period))
    RSI_test.columns = ['RSI']
        
    both = RSI_test.merge(data[stock], left_on =RSI_test.index ,right_on = data.index)
    both['date'] = both['key_0']

    after = both["date"] >= after_start_date
    before = both["date"] <= before_end_date
    between_two_dates = after & before
    df_filtered = both.loc[between_two_dates].reset_index()
    
    fig, ax1 = plt.subplots(figsize=(15,5))

    ax2 = ax1.twinx()
    ax1.plot(df_filtered['date'], df_filtered['RSI'], 'g-')
    ax2.plot(df_filtered['date'], df_filtered[stock], 'b-')
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('RSI', color='g')
    ax2.set_ylabel('stock', color='b')
    
    plt.show()
    
    
# Calculate avrage trade returns    
def average_trade_returns(indicex_of_positive_predictions, data):
    #Iterate over the positive predictions to calculate return
    for i in indicex_of_positive_predictions:
        index_n = i
        #Find cumulative return by row where return >= 5%
        bool_hitting_5Perc = data.loc[index_n+1:index_n+10,'price_change_rate_1d'].cumsum()>= 5
        # create list of indices that hit that target
        select_indices = list(data.loc[index_n+1:index_n+10,:][bool_hitting_5Perc].index)
        #If list is empty (no profits higher than 5%) take the return from day 10 after the stock was bought
        if not select_indices:
            #Index from day 10
            index_max = index_n+10
            #Realised return on day 10
            if data.loc[index_max, 'stock'] == data.loc[index_n, 'stock']:
                data.loc[index_n, 'trade returns'] = data.loc[index_max, 'price_change_rate_1d']
        #If list is not empty realise the trade as soon as the cummulative price change >= 5%
        else:
            idx_of_first_True = data.loc[index_n+1:index_n+10,'price_change_rate_1d'].cumsum()[bool_hitting_5Perc].index[0]
            if data.loc[idx_of_first_True, 'stock'] == data.loc[index_n, 'stock']:
                data.loc[index_n, 'trade returns'] = list(data.loc[index_n+1:index_n+10,'price_change_rate_1d'].cumsum()[bool_hitting_5Perc])[0]
    print("Median percentage return: %.2f%%" % (data[data['y_pred']==1]['trade returns'].median()))
    print("Mean percentage return: %.2f%%" % (data[data['y_pred']==1]['trade returns'].mean()))