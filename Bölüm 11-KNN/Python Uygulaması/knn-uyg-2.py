
# coding: utf-8

# In[1]:


import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Veri okuma işlemini yapıyoruz.
data = pd.read_csv("heart_failure.csv")
data.head(5)


# In[3]:


#Veri kümesindeki öznitelik isimlerini Türkçe'ye çeviriyoruz.
data = data.rename(columns={'age':'Yaş','anaemia':'anemi','creatinine_phosphokinase':'kreatinin fosfokinaz','diabetes':'diyabet','ejection_fraction':'ejeksiyon fraksiyonu','high_blood_pressure':'yüksek kan basıncı','platelets':'trombositler','serum_creatinine':'Serum kreatinin','serum_sodium':'serum sodyum','sex':'cinsiyet','smoking':'sigara','time':'zaman', 'DEATH_EVENT':'Durumu'})


# In[6]:


#Veri kümemizde bulunan bağımlı(sınıf) değişkeni ve test kümesi boyutunu %30 olacak şekilde belirtiyoruz.y = data.obezite.values
x = data.drop(['Durumu'],axis=True)
y = data['Durumu'].values
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=1)


# In[7]:


#Sınfılandırma algoritmasının kurulumu yapıyoruz. k=5 ve uzaklık ölçüsü öklid seçilmiştir.
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("Sınıflandırma işleminin ilk örnek sonucu: {}".format(prediction[0]))
print("{} komşuluk derecesine göre sınıflandırma işleminin doğruluk skoru: {}, \nprecision skoru: {}, recall skoru: {}, f1 skoru: {}"
      .format(knn.n_neighbors, knn.score(x_test, y_test), precision_score(y_test, prediction, average='weighted'),recall_score(y_test, prediction, average='weighted'), f1_score(y_test, prediction, average='weighted')))


# In[8]:


#knn algoritmasını optimize ederek en uygun k değerini bulalım.
knnscore = {}
for i in range(3,100,2):
    knnoptimize = KNeighborsClassifier(n_neighbors=i,metric='minkowski')
    knnoptimize.fit(x_train, y_train)
    knnscore[i] = knnoptimize.score(x_test,y_test)
max_k_value = max(zip(knnscore.values(),knnscore.keys()))[1]
print(f"Arama içerisindeki en yüksek doğruluk değerine sahip k değeri: {max_k_value}")

plt.plot(knnscore.keys(), knnscore.values())
plt.xlabel("k değeri")
plt.ylabel("doğruluk")
plt.show()


# In[9]:


#minkowski uzaklığına göre basit oylama sınıflandırma sonuçları k=99 için
knn = KNeighborsClassifier(n_neighbors=max_k_value,metric='minkowski')
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("Sınıflandırma işleminin ilk örnek sonucu: {}".format(prediction[0]))
print("{} komşuluk derecesine göre sınıflandırma işleminin doğruluk skoru: {}, \nprecision skoru: {}, recall skoru: {}, f1 skoru: {}"
      .format(knn.n_neighbors, knn.score(x_test, y_test), precision_score(y_test, prediction, average='weighted'),recall_score(y_test, prediction, average='weighted'), f1_score(y_test, prediction, average='weighted')))


# In[10]:


#minkowski uzaklığına göre ağırlıklı oylama sınıflandırma sonuçları k=99 için
knn = KNeighborsClassifier(n_neighbors=max_k_value,metric='minkowski',weights='distance')#weight parametresi herhangi bir değer almazsa default olarak basit oylama şeklinde çalışıyor. Distance parametresini alırsa ağırlıklı oylama olarak hesaplama yapıyor.
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("Sınıflandırma işleminin ilk örnek sonucu: {}".format(prediction[0]))
print("{} komşuluk derecesine göre sınıflandırma işleminin doğruluk skoru: {}, \nprecision skoru: {}, recall skoru: {}, f1 skoru: {}"
      .format(knn.n_neighbors, knn.score(x_test, y_test), precision_score(y_test, prediction, average='weighted'),recall_score(y_test, prediction, average='weighted'), f1_score(y_test, prediction, average='weighted')))


# In[11]:


#Karmaşıklık Matrisi
cnf_matrix = confusion_matrix(y_test, prediction)
cm_df = pd.DataFrame(cnf_matrix,index = ['Ölüm', 'Yaşam'], columns = ['Ölüm', 'Yaşam'])
plt.figure(figsize=(12,8))
sns.heatmap(cm_df, annot=True, fmt='g')
plt.title('Karmaşıklık Matrisi', fontsize=16)
plt.ylabel('Gerçek Değerler',fontsize=16)
plt.xlabel('Tahmin Edilen Değerler',fontsize=16)
plt.show()


# In[12]:


#veri seti dışındna bir örnek geldiğinde knn algoritmasının yaptığı tahmin.
prediction = knn.predict([[72,0,1205,1,23,0,315052,2.5,105,1,0,5]])
prediction


# In[13]:


#manhattan uzaklığına göre basit oylama sınıflandırma sonuçları k=99 için
knn = KNeighborsClassifier(n_neighbors=max_k_value,metric='manhattan')
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("Sınıflandırma işleminin ilk örnek sonucu: {}".format(prediction[0]))
print("{} komşuluk derecesine göre sınıflandırma işleminin doğruluk skoru: {}, \nprecision skoru: {}, recall skoru: {}, f1 skoru: {}"
      .format(knn.n_neighbors, knn.score(x_test, y_test), precision_score(y_test, prediction, average='weighted'),recall_score(y_test, prediction, average='weighted'), f1_score(y_test, prediction, average='weighted')))


# In[14]:


#manhattan uzaklığına göre ağırlıklı oylama sınıflandırma sonuçları k=99 için
knn = KNeighborsClassifier(n_neighbors=max_k_value,metric='manhattan',weights='distance')
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("Sınıflandırma işleminin ilk örnek sonucu: {}".format(prediction[0]))
print("{} komşuluk derecesine göre sınıflandırma işleminin doğruluk skoru: {}, \nprecision skoru: {}, recall skoru: {}, f1 skoru: {}"
      .format(knn.n_neighbors, knn.score(x_test, y_test), precision_score(y_test, prediction, average='weighted'),recall_score(y_test, prediction, average='weighted'), f1_score(y_test, prediction, average='weighted')))


# In[15]:


#öklid uzaklığına göre basit oylama sınıflandırma sonuçları k=99 için
knn = KNeighborsClassifier(n_neighbors=max_k_value,metric='euclidean')
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("Sınıflandırma işleminin ilk örnek sonucu: {}".format(prediction[0]))
print("{} komşuluk derecesine göre sınıflandırma işleminin doğruluk skoru: {}, \nprecision skoru: {}, recall skoru: {}, f1 skoru: {}"
      .format(knn.n_neighbors, knn.score(x_test, y_test), precision_score(y_test, prediction, average='weighted'),recall_score(y_test, prediction, average='weighted'), f1_score(y_test, prediction, average='weighted')))


# In[16]:


#öklid uzaklığına göre ağırlıklı oylama sınıflandırma sonuçları k=99 için
knn = KNeighborsClassifier(n_neighbors=max_k_value,metric='euclidean',weights='distance')
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("Sınıflandırma işleminin ilk örnek sonucu: {}".format(prediction[0]))
print("{} komşuluk derecesine göre sınıflandırma işleminin doğruluk skoru: {}, \nprecision skoru: {}, recall skoru: {}, f1 skoru: {}"
      .format(knn.n_neighbors, knn.score(x_test, y_test), precision_score(y_test, prediction, average='weighted'),recall_score(y_test, prediction, average='weighted'), f1_score(y_test, prediction, average='weighted')))

