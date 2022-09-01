# Linear Discriminant Analysis (LDA)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import statsmodels.formula.api as smf
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
# There are two features that we do not need "User ID" and "Gender",
# so, we will drop them.
dataset.drop(['User ID','Gender'], axis=1, inplace=True)
# Summary
print('Summary of Dataset: \n', dataset.describe())
# Mean
dataset_mean = dataset.groupby('Purchased').mean()
print('Dataset Mean: \n',dataset_mean)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from statistics import mean
# Split the dataset
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 0,
    )
# Summary of training
print('Summary of Training Dataset: \n', pd.DataFrame(X_train).describe())


# Applying LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
X_transform= lda.transform(X_train)

y_pred = lda.predict(X_test)
y_pob = lda.predict_proba(X_test)

# Weight of training dataset
df11=pd.DataFrame(lda.coef_[0].reshape(-1,1),['Age', 'EstimatedSalary'],columns=["Weight"])
df12=pd.DataFrame(lda.intercept_[0].reshape(-1,1),["Bias"],columns=["Weight"])
params = pd.concat([df12, df11], axis=0)
print(params)

# Printing the results
from sklearn.metrics import confusion_matrix, accuracy_score
print("")
print('Training Dataset Mean: \n',pd.DataFrame(X_train).mean())
print('')
print('Coefficient: ',lda.coef_)
print('')
print('Confusion Matrix: \n',confusion_matrix(y_test, y_pred))
print('')
print('Accuracy score: ', accuracy_score(y_test, y_pred))
print('Score: ',lda.score(X_train, y_train))
print("")

# Visualising the Training set
plt.figure(figsize=(15, 8))
# plotting the graph
plt.scatter(X_train[:,0],X_train[:,1],  c=y_train)
plt.show()
