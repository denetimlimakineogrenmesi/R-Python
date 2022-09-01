# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn import metrics

veri= veri.drop(['Own', 'Region'], axis=1)

le = preprocessing.LabelEncoder()
categ = ['Student', 'Married']
veri[categ] = veri[categ].apply(le.fit_transform)
veri.info()

oznitelikler = pd.DataFrame(veri.iloc[:,0: 8]) 
hedef = pd.DataFrame(veri.iloc[:, 8])

egitim_x,test_x,egitim_y,test_y=train_test_split(oznitelikler,hedef,test_size=0.20,random_state=42)

rf=RandomForestRegressor (n_estimators=1000, oob_score=True,random_state=2)
rf.fit(egitim_x,egitim_y)
print("Accuracy on training set: {:.3f}".format(rf.score(egitim_x, egitim_y)))

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

y_tahmin = rf.predict(test_x)
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(test_y, y_tahmin))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(test_y, y_tahmin))
print('Root Mean Squared Error (RMSE):', metrics.mean_squared_error(test_y, y_tahmin, squared=False))
print('Mean Absolute Percentage Error(MAPE):', metrics.mean_absolute_percentage_error(test_y, y_tahmin))
print('Explained Variance Score:', metrics.explained_variance_score(test_y, y_tahmin))
print('Max Error:', metrics.max_error(test_y, y_tahmin))
print('Mean Squared Log Error:', metrics.mean_squared_log_error(test_y, y_tahmin))
print('Median Absolute Error:', metrics.median_absolute_error(test_y, y_tahmin))
print('R^2:', metrics.r2_score(test_y, y_tahmin))
