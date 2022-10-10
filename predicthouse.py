import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
houses = pd.read_csv("kc_house_data.csv")
houses.head()
#check for nulls in the data
houses.isnull().sum()
# create x and y
feature_cols = 'sqft_living'
x = houses[feature_cols] # predictor
y = houses.price # response
# split data into train and test
x_train, x_test, y_train, y_test = train_test_split(
x, y, test_size=0.2)
# the test set will be 20% of the whole data set
# instantiate, fit
linreg = LinearRegression()
linreg.fit(x_train, y_train)
print linreg.intercept_
print linreg.coef_
-46773.65
[282.29] # for an increase of 1 square meter in house size,
# the house price will go up by ~$282, on average
# manually
price = -46773.65 + 1000*282.29
# using the model
linreg.predict(1000)
array([ 238175.93])
mse = mean_squared_error(y_test, linreg.predict(x_test))
np.sqrt(mse)
259163.48
linreg.score(x_test,y_test)
0.5543