import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from sklearn.datasets import fetch_california_housing
dataset_house = fetch_california_housing(as_frame=True)

ds_x = dataset_house.data
ds_y = dataset_house.target

tts = train_test_split()

# Fit regression model
regr_1 = DecisionTreeRegressor()
regr_1.fit(ds_x, ds_y)


# Predict

y_1 = regr_1.predict(X_test)


# Plot the results
plt.figure()
plt.scatter(dataset_house.data, dataset_house.target, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()