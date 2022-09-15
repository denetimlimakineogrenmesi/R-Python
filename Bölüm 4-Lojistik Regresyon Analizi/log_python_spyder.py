# -*- coding: utf-8 -*-






import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt


import seaborn as sns

veri.drop('Unnamed: 0', axis = 1, inplace = True)
veri.drop('id', axis = 1, inplace = True)
veri.drop('Inflight wifi service', axis = 1, inplace = True)
veri.drop('Departure/Arrival time convenient', axis = 1, inplace = True)
veri.drop('Ease of Online booking', axis = 1, inplace = True)   
veri.drop('Gate location', axis = 1, inplace = True)
veri.drop('Food and drink', axis = 1, inplace = True)
veri.drop('Online boarding', axis = 1, inplace = True)
veri.drop('Seat comfort', axis = 1, inplace = True)
veri.drop('Inflight entertainment', axis = 1, inplace = True)
veri.drop('On-board service', axis = 1, inplace = True)
veri.drop('Leg room service', axis = 1, inplace = True)
veri.drop('Baggage handling', axis = 1, inplace = True)
veri.drop('Checkin service', axis = 1, inplace = True)
veri.drop('Inflight service', axis = 1, inplace = True)
veri.drop('Cleanliness', axis = 1, inplace = True)
veri.drop('Arrival Delay in Minutes', axis = 1, inplace = True)


veri.info()
veri.isna().sum()

veri.columns = [c.replace(' ', '_') for c in veri.columns]

veri.describe()

sns.countplot(x='satisfaction',data=veri)
veri.satisfaction.value_counts()

lencoders = {}
for col in veri.select_dtypes(include=['object']).columns:
    lencoders[col] = LabelEncoder()
    veri[col] = lencoders[col].fit_transform(veri[col])


X = veri.iloc[:, 0:7]   #öznitelikler
y = veri.iloc[:, -1]     #hedef değişken


X_eğitim, X_test, y_eğitim, y_test = train_test_split(X, y, test_size = 0.20, random_state=0)

print("Öznitelikler için eğitim veri seti:        ",np.shape(X_eğitim))
print("Hedef değişken için eğitim veri seti:  ",np.shape(y_eğitim))
print("Öznitelikler için test veri seti:             ",np.shape(X_test))
print("Hedef değişken için test veri seti:       ",np.shape(y_test))


sc = StandardScaler()

X_eğitim = sc.fit_transform(X_eğitim)
X_test = sc.transform (X_test)


loj_reg_model = LogisticRegression().fit(X_eğitim, y_eğitim)

print("sabit terim:  ",loj_reg_model.intercept_)
print("katsayılar:   ",loj_reg_model.coef_)

Eğitim_tahmin = loj_reg_model.predict(X_eğitim)
print("Eğitim verisi tahminleri:  ",Eğitim_tahmin)


Eğitim_KM = metrics.confusion_matrix(y_eğitim, Eğitim_tahmin)

Eğitim_KM_Label = pd.DataFrame(Eğitim_KM)
Eğitim_KM_Label.columns = ['Tahmin Memnun Değil', 'Tahmin Memnun']
Eğitim_KM_Label = Eğitim_KM_Label.rename(index ={0: 'Gerçek Memnun Değil', 1: 'Gerçek Memnun'})

print("Eğitim Verisine Ait Karmaşıklık Matrisi")
Eğitim_KM_Label

Eğitim_KM_Prc = sns.heatmap(Eğitim_KM/np.sum(Eğitim_KM), annot=True, fmt='.2%', cmap='Blues')

Eğitim_KM_Prc.set_title('Eğitim Verisine Ait Karmaşıklık Matrisinin Yüzdesel Gösterimi\n\n');
Eğitim_KM_Prc.set_xlabel('\nTahmin Değerler')
Eğitim_KM_Prc.set_ylabel('Gerçek Değerler ');

Eğitim_KM_Prc.xaxis.set_ticklabels(['Memnun Değil','Memnun'])
Eğitim_KM_Prc.yaxis.set_ticklabels(['Memnun Değil','Memnun'])

plt.show()

print("Eğitim için doğruluk oranı:  ",loj_reg_model.score(X_eğitim, y_eğitim))


Test_tahmin = loj_reg_model.predict(X_test)
print("Test verisi tahminleri:  ",Test_tahmin)


# Karmaşıklık matrisinin oluşturulması
Test_KM = metrics.confusion_matrix(y_test, Test_tahmin)

# Matrisinin tahmin etiketlerinin yeniden isimlendirilmesi
Test_KM_Label = pd.DataFrame(Test_KM)
Test_KM_Label.columns = ['Tahmin Memnun Değil', 'Tahmin Memnun']
Test_KM_Label = Test_KM_Label.rename(index ={0: 'Gerçek Memnun Değil', 1: 'Gerçek Memnun'})

print("Test Verisine Ait Karmaşıklık Matrisi")
Test_KM_Label


Test_KM_Prc = sns.heatmap(Test_KM/np.sum(Test_KM), annot=True, fmt='.2%', cmap='Blues')

Test_KM_Prc.set_title('Test Verisine Ait Karmaşıklık Matrisinin Yüzdesel Gösterimi \n\n');
Test_KM_Prc.set_xlabel('\nTahmin Değerler')
Test_KM_Prc.set_ylabel('Gerçek Değerler ');

Test_KM_Prc.xaxis.set_ticklabels(['Memnun Değil','Memnun'])
Test_KM_Prc.yaxis.set_ticklabels(['Memnun Değil','Memnun'])

plt.show()


print("Test için doğruluk oranı:",loj_reg_model.score(X_test, y_test))


# Hedef değişkene ait sınıfların etiketlendirilmesi için dahil edilmiştir.
Sınıflar = ['Memnun Değil', 'Memnun']

print(classification_report(y_test, Test_tahmin, target_names=Sınıflar))




#ROC
YPO,DPO,Eşikdeğer = metrics.roc_curve(y_test, loj_reg_model.predict_proba(X_test)[:,1]) 
AUC = metrics.auc(YPO, DPO)

plt.plot(YPO, DPO, label='Lojistic Regresyon (AUC = %.2f)' %AUC)
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r')
plt.title('ROC Eğirisi')
plt.xlabel('Yanlış Pozitif Oranı')
plt.ylabel('Doğru Pozitif Oranı')
plt.grid()
plt.legend()
plt.show()



