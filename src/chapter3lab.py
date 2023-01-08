import pandas as pd
from sklearn.linear_model import LinearRegression

# 3.6 Lab: Linear Regression

boston = pd.read_csv('../data/Boston.csv')
boston.info()
boston.head(5)
y = boston.medv
# lstat = scale(boston.lstat, with_mean=True, with_std=False).reshape(-1, 1)
print(boston.lstat.array[0:10])
X = boston.lstat.array.reshape(-1, 1)
print(X[0:10])
singleRegression = LinearRegression().fit(X, y)
print(singleRegression.coef_)
print(singleRegression.intercept_)

X = boston.drop('medv', axis=1)
