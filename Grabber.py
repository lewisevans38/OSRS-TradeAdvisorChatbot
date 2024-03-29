import requests
from collections import namedtuple
import pandas as pd
import re
from LatestPriceDatabaseBuilder import Data

class TSGrabber():
    '''
    This class creates a dataframe of a inputted items time-series' of the variables AvgHighPrice,AvgLowPrice, HighPriceVol, LowPriceVol from the time series API.
    Caller is a helper function that uses requests to pull the entire time-series from the API. It returns the entire thing as a string.
    Frame manipulates this string using regex and returns a pandas dataframe of each column. 
    Finally Measure, takes this dataframe and creates a sub dataframe of the time increment and the column that you want to look at. 
    '''
    def __init__(self):
        super(TSGrabber,self).__init__()
    def Caller(ID):
        headers = {
        'User-Agent': 'TimeSeriesGrabber-@Stinjy',
        'From': 'lew.evans27@gmail.com'  # This is another valid field
    }
        URL = 'https://prices.runescape.wiki/api/v1/osrs/timeseries?'
        page = requests.get(URL, headers=headers, params = {"id":ID, "timestep":"1h"})
        return page.text
    
    def Frame(ID):
        '''
        Function is fed text from API, sorts through with regex to find values and returns a dataframe of Time, avghighPrice, avgLowPrice, high_price_vol and low_price_vol
        '''
        text = TSGrabber.Caller(ID)
        ## Doing some prework to set everything up.
        time_series = namedtuple('time_series', 'Time_hrs, AvgHighPrice, AvgLowPrice, HighPriceVol, LowPriceVol')
        lines = text.split('},')
        #Finding each time increment.
        timefinder = re.compile(r'["timestamp":]\d+')
        Times = [0]*len(lines)


        ## This is the meat of the regex stuff. We are just searching through and getting each value for the respective variable. I split the compilers up because I was finding regex very fiddly.
        ## To tidy things up I could put them in the same compiler, but for now I find this worked so I will stick with it.
        Datafinder = re.compile(r'"avgHighPrice":(\d+)')
        Datafinder1 = re.compile(r',"avgLowPrice":(\d+)')
        Datafinder2 = re.compile(r',"highPriceVolume":(\d+)')#"avgLowPrice":(\d+),"highPriceVolume":(\d+),"lowPriceVolume":(\d+)
        Datafinder3 = re.compile(r'"lowPriceVolume":(\d+)')
        avg_high_price= [0]*len(lines)
        avg_low_price = [0]*len(lines)
        high_price_vol= [0]*len(lines)
        low_price_vol = [0]*len(lines)
        
      
        x=0
        
        line_items = []
        ##This for loop is just doing the regex stuff of finding the correct values. The try except clauses are catching when we have a null variable and for now returning 0. Later I plan to fill these values by 
        ## filling them with the mean, but just to get a dataframe this suffices.
        for line in lines:
               
            Times[x] = x
            try:
                avg_high_price[x] = int(Datafinder.search(line).group(1).replace(":",""))
            except AttributeError:
                avg_high_price[x] = 0
            try:
                avg_low_price[x] = int(Datafinder1.search(line).group(1))
            except AttributeError:
                avg_low_price[x] = 0
            try:
                high_price_vol[x]= int(Datafinder2.search(line).group(1).replace(":",""))
            except AttributeError:
                high_price_vol[x]= 0
            try:

                low_price_vol[x] = int(Datafinder3.search(line).group(1).replace(":",""))
            except AttributeError:
                low_price_vol[x] = 0   
            line_items.append(time_series(Times[x], avg_high_price[x], avg_low_price[x], high_price_vol[x], low_price_vol[x]))
            x=x+1
        ##finally we create the dataframe
        dataframe = pd.DataFrame(line_items)
        return dataframe
    
    
    
    def Measure(dataframe,X):
        '''
        Creates a sub dataframe of x and time.
        
        X is the column of the dataframe we want to measure against Time_hrs
        
        '''
        datafrem = dataframe.loc[:,[X,'Time_hrs']]
        return datafrem