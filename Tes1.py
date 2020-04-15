import pandas as pan
import numpy as np
from sklearn.linear_model import LinearRegression

#Bagian ini berfungsi membaca sampel data
df = pan.read_csv('sample_data.csv')
df.head()

#Variabel Dependent dan Variabel Independent di inisialisasi
y = df['Gross Caloric Value']
x = df[['Mouisture','Ash','Volatile Matter']]

#Inisialisasi fungsi
linreg = LinearRegression()

#Fitting data
linreg.fit(x,y)

#Prediksi data
ypred = linreg.predict(x)
print(ypred)
