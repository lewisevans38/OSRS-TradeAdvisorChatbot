import xgboost
import pandas as pd 
import numpy as np
from Grabber import TSGrabber##This has the Grabber.Frame function, that spits out a time series (hr increment) of avg high price, avg low price, avg high price volume and avg low price volume,
from LatestPriceDatabaseBuilder import Data ##The Dataframe is called Data.Sheet
from statistics import mean 

def IDFinder(item_name):
    item_name =   item_name 
    df = Data.Sheet
    i = 0
    for x in df['ItemName']:
        if x == item_name: 
            ItemID=(df.iloc[i])['ID']
            break
        else:    
            i += 1
    return ItemID  


def forecast(ItemName, hrs):
    ''' A function that takes a ItemName and an amount of hours as an input. The function calls the time series API for the specific item, and then fits an XGBRegressor 
    the past 24 hours. After this is the model forecasts the price in the desired (inputted) hrs'''
    #As we are using the XGBoost regressor no matter what, it is okay to ignore the validation step. Best practice would be to use
    #Walk forward validation and ensure that the model performs well on the data.

    ID = IDFinder(ItemName)
    df = TSGrabber.Frame(ID)
    
    #setup the data for the avg_high_price
    df1 = TSGrabber.Measure(df,'AvgHighPrice')['AvgHighPrice']
 
    X = df1
    for x in range(len(df1)):
        if df1[x]==0:
            df1[x]=df1[x-1]
    
    y = X.shift(-1).dropna() #The target is the next time step
    

    model = xgboost.XGBRegressor(objective ='reg:squarederror', n_estimators=150,  )
    model.fit(X[:-1],y)
    X = np.array(X)
    for i in range(hrs):
        X = np.append(X, model.predict(X[:-1])[-1])
    
  
    avghighpred = int(X[-1])


    M = TSGrabber.Measure(df,'AvgLowPrice')['AvgLowPrice']

    for x in range(len(M)):
        if M[x]==0:
            M[x]=M[x-1]
    
    m = M.shift(-1).dropna() #The target is the next time step

    model = xgboost.XGBRegressor(objective ='reg:squarederror', n_estimators=150,  )
    model.fit(M[:-1],m)
    M = np.array(M)
    for i in range(hrs):
        M = np.append(M, model.predict(M[:-1])[-1])
  
    avglowpred = int(M[-1])
    
    return avghighpred , avglowpred
    


#print(forecast("Dragon scimitar", 1))