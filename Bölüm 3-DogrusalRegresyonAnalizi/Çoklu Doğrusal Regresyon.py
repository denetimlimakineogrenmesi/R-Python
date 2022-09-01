# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

veriler = pd.read_csv('C:\Dosyalar\X2019.csv')
print(veriler.head(10))
v10 = veriler.head(10)

veri = veriler[['Score','GDP per capita','Social support','Healthy life expectancy','Freedom to make life choices','Generosity','Perceptions of corruption']]
print(veri.head(10))


print(veri.isnull().values.any())
veri.info()

veri.describe()
print(veri.describe())
betist = veri.describe()

veri.corr()
print(veri.corr())
vericorr = veri.corr()

fig = plt.figure(figsize=(7,5))
sns.set()
sns.distplot(veri['Score'],bins=12);

veri.hist(edgecolor = 'white', linewidth = 0.5, figsize = (16,8),grid=False,color='#0B486B')
plt.show()

X = veri[['GDP per capita','Social support','Healthy life expectancy','Freedom to make life choices','Generosity','Perceptions of corruption']]
Y = veri[['Score']]

X_egitim, X_test, Y_egitim, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

LinReg = LinearRegression().fit(X_egitim,Y_egitim)

print("Sabit terim: ",LinReg.intercept_)
print("Parametreler:  ",LinReg.coef_)

egitim_tahmin = pd.DataFrame({'Gerçek': Y_egitim.values.flatten(), 'Tahmin': LinReg.predict(X_egitim).flatten()})
print(egitim_tahmin)


plt.title('Eğitim Verisi Tahmin Grafiği')
plt.scatter(Y_egitim,LinReg.predict(X_egitim), label="Gerçek Değerler")
plt.plot([2, 8], [2, 8], linestyle='--', lw=2, color='r', label="Tahmini Değerler")
plt.grid()
plt.legend()
plt.show()

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error

print('Mean Absolute Error:', mean_absolute_error(Y_egitim, LinReg.predict(X_egitim)))
print('Mean Absolute Percentage Error:', mean_absolute_percentage_error(Y_egitim, LinReg.predict(X_egitim))) 
print('Mean Squared Error:', mean_squared_error(Y_egitim, LinReg.predict(X_egitim)))  
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(Y_egitim, LinReg.predict(X_egitim))))
print('R^2:',LinReg.score(X_egitim,Y_egitim))


test_tahmin = pd.DataFrame({'Gerçek': Y_test.values.flatten(), 'Tahmin': LinReg.predict(X_test).flatten()})
print(test_tahmin.head(10))

plt.title('Test Verisi Tahmin Grafiği')
plt.scatter(Y_test,LinReg.predict(X_test), label="Gerçek Değerler")
plt.plot([2, 8], [2, 8], linestyle='--', lw=2, color='r', label="Tahmini Değerler")
plt.grid()
plt.legend()
plt.show()


print('Mean Absolute Error:', mean_absolute_error(Y_test, LinReg.predict(X_test)))
print('Mean Absolute Percentage Error:', mean_absolute_percentage_error(Y_test, LinReg.predict(X_test))) 
print('Mean Squared Error:', mean_squared_error(Y_test, LinReg.predict(X_test)))  
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(Y_test, LinReg.predict(X_test))))
print('R^2:',LinReg.score(X_test,Y_test))















