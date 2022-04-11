
# Change directory to the directory above "data"

# LinearRegression is a machine learning library for linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# pandas and numpy are used for data manipulation
import pandas as pd
import numpy as np
from math import sqrt
from numpy import log
from pandas import Series


from flask import *
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller, arma_order_select_ic
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import statsmodels as sm

# matplotlib and seaborn are used for plotting graphs
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import seaborn as sns
from datetime import datetime
import subprocess


ds_gold = 'Indian rupee'
ds_etf = 'Close'
date_format = '%Y-%m-%d'
df = pd.read_csv("data.csv")

df = df[['Name', ds_gold]]
df['Name'] = [datetime.strptime(i, date_format) for i in df['Name']]
df.set_index('Name')

dd =df

"""*  Drop rows with missing values"""

df = df.dropna()

df[ds_gold].hist()
log_transform = log(df[ds_gold])

sns.set()

# Can be used to show non stationary

# Define exploratory variables
# Finding moving average of past 3 days and 9 days
df['S_1'] = df[ds_gold].shift(1).rolling(window=3).mean()
df['S_2'] = df[ds_gold].shift(1).rolling(window=12).mean()
df = df.dropna()
X = df[['S_1', 'S_2']]
X.head()

# dependent variable
y = df[ds_gold]
y.head()

# Split into train and test
t = 0.2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=t, shuffle=False)

# Performing linear regression
linear = LinearRegression().fit(X_train, y_train)
predicted_price = linear.predict(X_test)

predicted_price = pd.DataFrame(
    predicted_price, index=y_test.index, columns=['price'])
predicted_price.plot(figsize=(10, 5))
y_test.plot()


# Calculate R square and rmse to check goodness of fit
r2_score = linear.score(X_test, y_test)*100
sqrt(mean_squared_error(y_test,predicted_price))



# Check stationarity
X = df[ds_gold]
split = len(X) // 2
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()

result_of_adfuller = adfuller(df[ds_gold])


# Now taking log transform
log_transform = log(df[ds_gold])
result_of_adfuller = adfuller(log_transform)

# To remove trends, differencing of order 1
k = df[ds_gold].diff()

# print(k.head())
k = k.dropna()

# check stationarity after differencing
result_of_adfuller = adfuller(k)

# So now we can say with 1 % confidence level that its stationary


# Again regression
df[ds_gold] = k
# Finding moving average of past 3 days and 9 days
df['S_1'] = df[ds_gold].shift(1).rolling(window=3).mean()
df['S_2'] = df[ds_gold].shift(1).rolling(window=12).mean()
df = df.dropna()
X = df[['S_1', 'S_2']]
X.head()
df['S_1'] = df[ds_gold].shift(1).rolling(window=3).mean()
df['S_2'] = df[ds_gold].shift(1).rolling(window=12).mean()

# dependent variable
y = df[ds_gold]
y.head()
# print(y.head())

# Split into train and test
t = 0.2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=t, shuffle=False)

# Performing linear regression
linear = LinearRegression().fit(X_train, y_train)
# Predict prices

predicted_price = linear.predict(X_test)

predicted_price = pd.DataFrame(
    predicted_price, index=y_test.index, columns=['price'])

app = Flask(__name__)  
  
def fun(a,b):
    global linear
    l=[]
    ll=[]
    l.append(a)
    ll.append(b)
    d={'S_1':l,'S_2':ll}
    l=pd.DataFrame(d)
    print(l)
    a=linear.predict(l)
    print(a)
    return a
@app.route('/login',methods = ['GET'])  
def login():  
      d3=request.args.get('d3')  
      d9=request.args.get('d9')
      ans=fun(d3,d9)
      return render_template('page3.html',name=ans)
app.run()