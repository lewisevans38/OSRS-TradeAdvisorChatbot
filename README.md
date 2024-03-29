# OSRS-TradeAdvisorChatbot
OldSchool Runescape (OSRS) is a MMORPG that contains around 2.5million daily players. In the game, players can trade items between themselves through a centralised trading post called the Grand Exchange. 

This project uses the APIs on  https://oldschool.runescape.wiki/w/RuneScape:Real-time_Prices to construct a database of every tradable item, keeping track of different price characteristics like AvgHighPrice, AvgLowPrice etc.
Further, using the APIs the project constructs a time series for each item, tracking the price of the item at each of the last 24 hours. With these datums, the project fits an XGBoost Regressor, and creates a function that forecasts the price of a item after an inputted amount of hours.

To interface with these functions 
