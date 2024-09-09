import time
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split



items = [["Open","Close","Low","High"],
         ["Open","Close","Low","High","Volume"],
         ["Open","Close","Low","High","Volume","Benefit"],
         ["Open","Close","Low","High","Volume","Benefit","SMA14"],
         ["Open","Close","Low","High","Volume","Benefit","SMA14","SMA21"]]


result_list = []

for year in ["2015-01-01","2017-01-01","2019-01-01","2021-01-01","2023-01-01"]:

    btc = yf.download("BTC-USD", start=year,end="2024-07-30")

    btc["Benefit"] = btc["Close"]-btc["Open"]
    btc["Tomorrow"] = btc["Close"].shift(-1)
    btc["SMA14"] = btc["Close"].rolling(14).mean()
    btc["SMA21"] = btc["Close"].rolling(21).mean()
    btc.dropna(inplace=True)

    for item in items:
        X = btc[item]
        y = btc["Tomorrow"]

        x_train,x_test,y_train,y_test=train_test_split(X,y, test_size=0.2, shuffle=False)

        for model in [LinearRegression(), Lasso(), Ridge()]:
            start_time = time.time()
            model.fit(x_train,y_train)

            all_pred = model.predict(X)
            y_test_pred = model.predict(x_test)
            y_train_pred = model.predict(x_train)

            all_rmse = root_mean_squared_error(y, all_pred)
            test_rmse = root_mean_squared_error(y_test, y_test_pred)
            train_rmse = root_mean_squared_error(y_train, y_train_pred)
            end_time = time.time()

            result = {
                "Items" : item,
                "Model": f"{str(model).replace("(", "").replace(")", "")}",
                "Year" :year,
                "ALL" : all_rmse,
                "Test" : test_rmse,
                "Train" : train_rmse,
                "Time" : end_time-start_time,
            }
            result_list.append(result)

result_df = pd.DataFrame(result_list)


plt.subplot(2,1,1)
plt.plot(result_df.index,result_df["ALL"], label = "All")
plt.plot(result_df.index,result_df["Test"], label = "test")
plt.plot(result_df.index,result_df["Train"], label = "train")
plt.legend()

plt.subplot(2,1,2)
plt.plot(result_df.index,result_df["Time"],label = "Time")
plt.legend()
plt.show()