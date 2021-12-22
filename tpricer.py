from scipy import stats
import pandas_market_calendars as mcal
import datetime as dt
import pandas as pd
import numpy as np
import yahoo_fin.stock_info as si
from yahoo_fin.options import get_options_chain
import timeit


def truncated_paths(S,df,loc,scale,steps,N, bounds):
    lb = bounds[0]
    ub = bounds[1]
    r = stats.t.rvs(df = df, loc = loc, scale = scale, size = int(steps*N))
    r = [lr for lr in r if lr < ub and lr > lb]
    while len(r) < steps*N:
        size = int(np.rint((steps*N-len(r))*1.2))
        t = stats.t.rvs(df = df, loc = loc, scale = scale, size = size)
        r = np.concatenate((r,t))
    r = r[:steps*N]
    r = np.reshape(r, (steps,N))   
     
    ST = S*np.exp(np.cumsum(r, axis =0))
    P = np.full(shape=N,fill_value=S)
    ST = np.concatenate(([P],ST), axis = 0)
    return ST



def get_dte(exp):
    format_data = "%Y-%m-%d"
    exp_date = dt.datetime.date(dt.datetime.strptime(exp,format_data))
    exp_date = dt.datetime(exp_date.year, exp_date.month, exp_date.day).replace(hour = 16)
    
    
    now = dt.datetime.now()
    now_hour = now.strftime('%H')
    if int(now_hour) > 14:
        start = dt.date.today() + dt.timedelta(days=1)
    else:
        start = dt.date.today()
        
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date = start.strftime('%Y-%m-%d'), end_date = exp)
    dte = schedule.shape[0]
    return dte
        
        
    
def price(ticker, flag, K ,expiration):
    df = si.get_data(ticker)
    arr = df.to_numpy()
    closes = arr[:,3]
    log_returns = np.array([np.log(closes[i]/closes[i-1]) for i in range(1,len(closes))])
    log_returns = log_returns[~np.isnan(log_returns)]
    fit = stats.t.fit(log_returns)
    v,loc,scale = fit[0],fit[1],fit[2]
    dte = get_dte(expiration)
    bounds = (-2.5/v,2/v)
    S = si.get_live_price(ticker)
    ST = truncated_paths(S,v,loc,scale,dte,100000,bounds)
    S_t = ST[-1]
    if flag == 'c':
        vals = [np.max([S-K,0]) for S in S_t]
    else:
        vals = [np.max([K-S,0]) for S in S_t]
        
    return np.mean(vals) 

def price_chain(ticker, expiration):
    #gets price history
    df = si.get_data(ticker)
    arr = df.to_numpy()
    closes = arr[:,3]
    
    #calculates log-returns
    log_returns = np.array([np.log(closes[i]/closes[i-1]) for i in range(1,len(closes))])
    log_returns = log_returns[~np.isnan(log_returns)]
    
    #fits a t distribution
    fit = stats.t.fit(log_returns)
    v,loc,scale = fit[0],fit[1],fit[2]
    dte = get_dte(expiration)
    bounds = (-2.5/v,2/v)
    
    #calculates price paths
    S = si.get_live_price(ticker)
    ST = truncated_paths(S,v,loc,scale,dte,100000,bounds)
    S_t = ST[-1]
    
    #gets option chain a throws away all but the strike and last trade price
    chain = get_options_chain(ticker,expiration)
    calls = chain['calls'].to_numpy()
    puts = chain['puts'].to_numpy()
    calls = calls[:,2:4]
    calls = np.array([row for row in calls if row[0] < 1.25*S and row[0] > S ])
    puts = puts[:,2:4]
    puts = np.array([row for row in puts if row[0] < S and row[0] > .8*S ])
    
    #calculates the price of each option in the chain and the SE of the estimate
    call_strikes = calls[:,0]
    put_strikes = puts[:,0]
    put_prices = []
    call_prices = []
    call_CI = []
    put_CI = []

    for K in call_strikes:
        call_vals = [np.max([S-K,0]) for S in S_t]
        #sem_c = np.around(np.sqrt(np.var(call_vals)/100000), decimals = 3)
        x_c = np.mean(call_vals)
        #CI_c = '(' + str(np.around(x_c - 2.58*sem_c,2)) + ' , '  + str(np.around(x_c + 2.58*sem_c,2)) + ')'
        #call_CI.append(CI_c)
        call_prices.append(np.around(x_c, decimals = 2))
        
    for K in put_strikes:
        put_vals = [np.max([K-S,0]) for S in S_t]
        #sem_p = np.around(np.sqrt(np.var(put_vals)/100000), decimals = 3)
        x_p = np.mean(put_vals)
        #CI_p = '(' + str(np.around(x_p - 2.58*sem_p,2)) + ' , '  + str(np.around(x_p + 2.58*sem_p,2)) + ')'
        #put_CI.append(CI_p)
        put_prices.append(np.around(x_p, decimals = 2))
        
    #appends the new price to the end of the array    
    calls = np.append(calls, np.transpose([call_prices]), axis = 1)
    puts = np.append(puts, np.transpose([put_prices]), axis = 1)
    
    #Estimating the kelly fraction
    #call_p_hats is estimated P_itm, calls[1] is Last Price, calls[2] is estimated prices
    b = np.divide(call_prices,calls[:,1])

    
    b1 = np.divide(put_prices,puts[:,1])
  
    
    calls = np.append(calls, np.transpose([b]), axis =1)
    puts = np.append(puts, np.transpose([b1]), axis = 1)
    
    #appends CI's to the end of the array
    #calls = np.append(calls, np.transpose([call_CI]), axis = 1)
    #puts = np.append(puts, np.transpose([put_CI]), axis = 1)
    

    
    #adds a header for easy reading
    header = ['Strike', 'Last Price', 'Estimate', 'Ratio' ]
    calls = np.insert(calls, 0, header, axis = 0)
    puts = np.insert(puts,0,header,axis = 0)
    
    #print the chains
    print(ticker.upper())
    print(expiration)
    print('CALLS:')
    print(calls)
    print('PUTS:')
    print(puts)
        
        
        
    
#start = timeit.default_timer()
#price_chain('low', '2021-12-23')
#stop = timeit.default_timer()
#print('Time: ', stop - start)     
    
    
    
    