# -*- coding: utf-8 -*-
"""
Created on Tue May 31 11:48:06 2022

@author: HP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report


veri= veri.drop(['customerID', 'gender','SeniorCitizen','Partner','Dependents','PhoneService',
           'MultipleLines','StreamingTV','StreamingMovies','MonthlyCharges',
           'PaperlessBilling'], axis=1)

le = preprocessing.LabelEncoder()
categ = ['InternetService','OnlineSecurity','OnlineBackup',
         'DeviceProtection','TechSupport','PaymentMethod',
         'Contract','Churn']
veri[categ] = veri[categ].apply(le.fit_transform)

veri['TotalCharges'] = pd.to_numeric(veri['TotalCharges'], errors='coerce')
eksik_veriler=veri.isnull().sum()

veri['TotalCharges'] = veri['TotalCharges'].fillna(veri['TotalCharges'].mean())
veri.info()  

oznitelikler = veri.iloc[:,0: 9]
hedef = veri.iloc[:, 9]


egitim_x,test_x,egitim_y,test_y= train_test_split(oznitelikler,hedef,test_size=0.20,random_state=42)

rf=RandomForestClassifier(n_estimators=1000,criterion='entropy',random_state=2)
rf.fit(egitim_x,egitim_y)
print("Accuracy on training set: {:.3f}".format(rf.score(egitim_x, egitim_y)))

y_tahmin_e = rf.predict(egitim_x)

cm = confusion_matrix(egitim_y,y_tahmin_e)
display = ConfusionMatrixDisplay(cm).plot()
plt.title('Karmaşıklık Matrisi-Eğitim')
plt.show()

result = permutation_importance(rf, egitim_x, egitim_y, n_repeats=5, random_state=42)
perm_sorted_idx = result.importances_mean.argsort()

tree_importance_sorted_idx = np.argsort(rf.feature_importances_)
tree_indices = np.arange(0, len(rf.feature_importances_)) + 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
ax1.barh(tree_indices, rf.feature_importances_[tree_importance_sorted_idx], height=0.7)
ax1.set_yticks(tree_indices)

ax1.set_yticklabels(veri.columns[tree_importance_sorted_idx])

ax1.set_ylim((0, len(rf.feature_importances_)))
ax2.boxplot(
    result.importances[perm_sorted_idx].T,
    vert=False,
    labels=veri.columns[perm_sorted_idx],

)
fig.tight_layout()
plt.show()

y_tahmin_t = rf.predict(test_x)

cm = confusion_matrix(test_y,y_tahmin_t)
display = ConfusionMatrixDisplay(cm).plot()
plt.title('Karmaşıklık Matrisi-Test')
plt.show()

print("Accuracy on test set: {:.3f}".format(rf.score(test_x, test_y)))
cr= classification_report(test_y,y_tahmin_t)
print("Sınıflandırma Raporu") 
print(cr)


