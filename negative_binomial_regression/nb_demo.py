import pandas as pd
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


df = pd.read_csv('06 June 2017 Cyclist Numbers for Web.csv', header=0, infer_datetime_format=True, parse_dates=[0], index_col=[0])
ds = df.index.to_series()
df['MONTH'] = ds.dt.month
df['DAY_OF_WEEK'] = ds.dt.dayofweek
df['DAY'] = ds.dt.day


mask = np.random.rand(len(df)) < 0.8
df_train = df[mask]
df_test = df[~mask]
print('Training data set length='+str(len(df_train)))
print('Testing data set length='+str(len(df_test)))


expr = """Brooklyn_Bridge ~ DAY  + DAY_OF_WEEK + MONTH + HighTemp + LowTemp + Precipitation"""

y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')

poisson_training_results = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
print(poisson_training_results.summary())

print(poisson_training_results.mu)
print(len(poisson_training_results.mu))