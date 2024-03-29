import requests
import re
import pandas as pd
from collections import namedtuple

class Data:
    #This class is used to build the fundamental database of each item. Giving an overall dataframe with columns of item id, item naem, current buy price, sell price, margin between the two
    #alltimehigh and alltime low.
    
    
    ##This code makes the reference database including item name, and item ID
    URL= 'https://prices.runescape.wiki/api/v1/osrs/mapping'
    PAGE= requests.get(URL)
    text =str(PAGE.text)
    lines=text.split('},')
    ref_sheet= namedtuple('ref_sheet','ID ItemName' )
    line_items=[]
    for x in range(1,len(lines)):
        lines[x] = lines[x].split(':')
        #id
        lines[x][2] = re.sub(",\"members\"","",lines[x][2])
        ID = lines[x][2]  
        #name
        #conditionals are to remove the problem of some items not having limits
        #which throws the index out.
        if len(lines[x])==10:
            names = lines[x][9]
        elif len(lines[x])==9:
            names=lines[x][8]
        elif len(lines[x])==8:
            names=lines[x][7]
        else:
            names=lines[x][6]
        #Now I just want to build a reference sheet database for the names and ID
        line_items.append(ref_sheet(ID,names))
    refsheet = pd.DataFrame(line_items)
    
    
    ##This code creates the LatestAPI Database
    url = 'https://prices.runescape.wiki/api/v1/osrs/latest'
    page = requests.get(url)
    text = str(page.text)
    Sheet = namedtuple('Sheet', 'ID BuyPrice AllTimeHigh SellPrice AllTimeLow,Margin')
    rows = text.split("},")
    row_items= []
    for x in range(1,len(rows)-1):
        #This splits up the text again
        rows[x]=rows[x].split(":")

        #This cleans and names the Item ID list. 
        rows[x][0] = re.sub("\"","",rows[x][0])
        ID = rows[x][0]
        #This finds the High price
        rows[x][2] = re.sub(",\"highTime\"","",rows[x][2])
        High= rows[x][2]
        #This finds the allTimeHigh for the day
        rows[x][3] = re.sub(",\"low\"","",rows[x][3])
        allTimeHigh = rows[x][3]
        #Low Price
        rows[x][4]=re.sub(",\"lowTime\"","",rows[x][4])
        Low= rows[x][4]
        #This finds the allTimeLowest for the day
        allTimeLow= rows[x][5]

        try:
            High=int(High)
            Low =int(Low)
            allTimeHigh=int(allTimeHigh)
            allTimeLow=int(allTimeLow)
        except ValueError:
              continue
        #Margin Calculator
        #Margin= High - Low
        Margin = High - Low
        row_items.append(Sheet(ID,High,allTimeHigh,Low,allTimeLow,Margin))

    df = pd.DataFrame(row_items)
    
    #And Finally this code merges the two functions to create the main Database that I need. 
   
    Sheet = pd.merge(df,refsheet, on='ID' , how = 'inner')
    
    
    def IDFinder(item_name):
        df = Data.Sheet
        i = 0
        ItemID=None
        for x in df['ItemName']:
            #print(x)
            #print(x)
            if x == 'item_name': 
                ItemID=(df.iloc[i])['ID']
                break
            else:    
                i += 1
        return ItemID
