# OSRS-TradeAdvisorChatbot
OldSchool Runescape (OSRS) is a MMORPG that contains around 2.5million daily players. In the game, players can trade items between themselves through a centralised trading post called the Grand Exchange. 

This project uses the APIs on  https://oldschool.runescape.wiki/w/RuneScape:Real-time_Prices to construct a database of every tradable item, keeping track of different price characteristics like AvgHighPrice, AvgLowPrice etc.
Further, using the APIs the project constructs a time series for each item, tracking the price of the item at each of the last 24 hours. With these datums, the project fits an XGBoost Regressor, and creates a function that forecasts the value of a item at the requested amount of hours.

To interface with these functions, the project employs a Chatbot. The Chatbot trains and employs a DeBERTa model from huggingface to use as an intent classifier and can currently output 4 main category of responses. The tags are the following, Greeting, Goodbye, forecast, Margin.
If the model classifies the intent as Forecast, the project detects two key words, Number of Hours, and the Item Name and then uses the above function to predict the value. If the model classifies the intent as Margin, the program detects two key words, the threshold price, and the number of items the user wishes to see. After detecting these key words, the program prints a dataframe.head of all the items in the game, sorted by the difference between the BuyPrice and SellPrice, that are below the threshold price.  


![image](https://github.com/lewisevans38/OSRS-TradeAdvisorChatbot/assets/143433180/31fd07a4-b187-480a-9ed1-83c22f1e72be)
