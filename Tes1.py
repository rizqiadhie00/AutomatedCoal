import pandas as pan
import numpy as np
from sklearn.linear_model import LinearRegression

df = pan.read_csv('sample_data.csv')
df.head()

y = df['Gross Caloric Value']
x = df[['Mouisture','Ash','Volatile Matter']]

linreg = LinearRegression()

linreg.fit(x,y)

ypred = linreg.predict(x)
