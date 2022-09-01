
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


# In[16]:


#Veri okuma işlemini yapıyoruz.
data = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
data.head(5)


# In[17]:


#Veri kümesindeki öznitelik isimlerini Türkçe'ye çeviriyoruz.
data = data.rename(columns={'Gender':'Cinsiyet','Age':'Yaş','Height':'Boy','Weight':'Kilo','family_history_with_overweight':'obezite_aile_geçmişi','FAVC':'yüksek_kalorili_gıda','FCVC':'sebze_tüketim_sıklığı','NCP':'ana_öğün_sayısı','CAEC':'öğünler_arası_gida_tüketimi','SMOKE':'sigara','CH2O':'günlük_su_tüketimi','SCC':'kalori_izleme','FAF':'fiziksel_aktivite_sıklığı','TUE':'teknoloji_cihazı_kullanma_süresi','CALC':'alkol','MTRANS':'ulaşım_aracı','NObeyesdad':'obezite'})
data.head()


# In[18]:


#Özniteliklerimizi sınıflandırma algoritmaları için sayısal hale getiriyoruz.
labelEncoder = LabelEncoder()

data['Cinsiyet'] = labelEncoder.fit_transform(data['Cinsiyet'])
data['obezite_aile_geçmişi'] = labelEncoder.fit_transform(data['obezite_aile_geçmişi'])
data['yüksek_kalorili_gıda'] = labelEncoder.fit_transform(data['yüksek_kalorili_gıda'])
data['öğünler_arası_gida_tüketimi'] = labelEncoder.fit_transform(data['öğünler_arası_gida_tüketimi'])
data['sigara'] = labelEncoder.fit_transform(data['sigara'])
data['kalori_izleme'] = labelEncoder.fit_transform(data['kalori_izleme'])
data['alkol'] = labelEncoder.fit_transform(data['alkol'])
data['ulaşım_aracı'] = labelEncoder.fit_transform(data['ulaşım_aracı'])


#Bu alanda bulunan kodlar obezite sütunundaki verileri türkçeleri ile değiştiriyor.
data.loc[data["obezite"] == "Normal_Weight", "obezite"] = 'Normal Ağırlık'
data.loc[data["obezite"] == "Overweight_Level_I", "obezite"] = 'Kilolu Seviye I'
data.loc[data["obezite"] == "Overweight_Level_II", "obezite"] = 'Kilolu Seviye II'
data.loc[data["obezite"] == "Obesity_Type_I", "obezite"] = 'Obezite Tip I'
data.loc[data["obezite"] == "Obesity_Type_II", "obezite"] = 'Obezite Tip II'
data.loc[data["obezite"] == "Obesity_Type_III", "obezite"] = 'Obezite Tip III'
data.loc[data["obezite"] == "Insufficient_Weight", "obezite"] = 'Yetersiz Ağırlık'

data.head(5)


# In[19]:


#Veri kümemizde bulunan bağımlı(sınıf) değişkeni ve test kümesi boyutunu %30 olacak şekilde belirtiyoruz.y = data.obezite.values
y = data.obezite.values
x = data.drop(['obezite'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=1)


# In[20]:


#Sınfılandırma algoritmasının kurulumu yapıyoruz. k=5 ve uzaklık ölçüsü öklid seçilmiştir.
knn = KNeighborsClassifier(n_neighbors=5,metric='euclidean')
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("Sınıflandırma işleminin ilk örnek sonucu: {}".format(prediction[0]))
print("{} komşuluk derecesine göre sınıflandırma işleminin doğruluk skoru: {}, precision skoru {}, recall skoru {}, f1 skoru {}"
      .format(knn.n_neighbors, knn.score(x_test, y_test), precision_score(y_test, prediction, average='weighted'),recall_score(y_test, prediction, average='weighted'), f1_score(y_test, prediction, average='weighted')))


# In[22]:


#knn algoritmasını optimize ederek en uygun k değerini bulalım.
knnscore = {}
for i in range(3,100,2):
    knnoptimize = KNeighborsClassifier(n_neighbors=i,metric='euclidean')
    knnoptimize.fit(x_train, y_train)
    knnscore[i] = knnoptimize.score(x_test,y_test)
max_k_value = max(zip(knnscore.values(),knnscore.keys()))[1]
print("Arama içerisindeki en yüksek doğruluk değerine sahip k değeri: {max_k_value}")

plt.plot(knnscore.keys(), knnscore.values())
plt.xlabel("k değeri")
plt.ylabel("doğruluk")
plt.show()


# In[23]:


#öklid uzaklığına göre basit oylama sınıflandırma sonuçları k=3 için
knn = KNeighborsClassifier(n_neighbors=max_k_value,metric='euclidean')
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("Sınıflandırma işleminin ilk örnek sonucu: {}".format(prediction[0]))
print("{} komşuluk derecesine göre sınıflandırma işleminin doğruluk skoru: {}, precision skoru {}, recall skoru {}, f1 skoru {}"
      .format(knn.n_neighbors, knn.score(x_test, y_test), precision_score(y_test, prediction, average='weighted'),recall_score(y_test, prediction, average='weighted'), f1_score(y_test, prediction, average='weighted')))


# In[24]:


#öklid uzaklığına göre ağırlıklı oylama sınıflandırma sonuçları k=3 için
knn = KNeighborsClassifier(n_neighbors=max_k_value,metric='euclidean',weights='distance')#weight parametresi herhangi bir değer almazsa default olarak basit oylama şeklinde çalışıyor. Distance parametresini alırsa ağırlıklı oylama olarak hesaplama yapıyor.
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("Sınıflandırma işleminin ilk örnek sonucu: {}".format(prediction[0]))
print("{} komşuluk derecesine göre sınıflandırma işleminin doğruluk skoru: {}, precision skoru {}, recall skoru {}, f1 skoru {}"
      .format(knn.n_neighbors, knn.score(x_test, y_test), precision_score(y_test, prediction, average='weighted'),recall_score(y_test, prediction, average='weighted'), f1_score(y_test, prediction, average='weighted')))


# In[25]:


#Karmaşıklık Matrisi
cnf_matrix = confusion_matrix(y_test, prediction)
cm_df = pd.DataFrame(cnf_matrix,index = ['Normal Ağırlık', 'Kilolu Seviye I', 'Kilolu Seviye II','Obezite Tip I','Yetersiz Ağırlık','Obezite Tip II','Obezite Tip III'], columns = ['Normal Ağırlık', 'Kilolu Seviye I', 'Kilolu Seviye II','Obezite Tip I','Yetersiz Ağırlık','Obezite Tip II','Obezite Tip III'])
plt.figure(figsize=(12,8))
sns.heatmap(cm_df, annot=True, fmt='g')
plt.title('Karmaşıklık Matrisi')
plt.ylabel('Gerçek Değerler')
plt.xlabel('Tahmin Edilen Değerler')
plt.show()


# In[26]:


#veri seti dışındna bir örnek geldiğinde knn algoritmasının yaptığı tahmin.
prediction = knn.predict([[0,40,1.74,79,1,0,3.0,1.0,2,1,2.0,1,0.0,1,3,3]])
prediction


# In[27]:


#manhattan uzaklığına göre basit oylama sınıflandırma sonuçları k=3 için
knn = KNeighborsClassifier(n_neighbors=max_k_value,metric='manhattan')
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("Sınıflandırma işleminin ilk örnek sonucu: {}".format(prediction[0]))
print("{} komşuluk derecesine göre sınıflandırma işleminin doğruluk skoru: {}, precision skoru {}, recall skoru {}, f1 skoru {}"
      .format(knn.n_neighbors, knn.score(x_test, y_test), precision_score(y_test, prediction, average='weighted'),recall_score(y_test, prediction, average='weighted'), f1_score(y_test, prediction, average='weighted')))


# In[28]:


#manhattan uzaklığına göre ağırlıklı oylama sınıflandırma sonuçları k=3 için
knn = KNeighborsClassifier(n_neighbors=max_k_value,metric='manhattan',  weights='distance')
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("Sınıflandırma işleminin ilk örnek sonucu: {}".format(prediction[0]))
print("{} komşuluk derecesine göre sınıflandırma işleminin doğruluk skoru: {}, precision skoru {}, recall skoru {}, f1 skoru {}"
      .format(knn.n_neighbors, knn.score(x_test, y_test), precision_score(y_test, prediction, average='weighted'),recall_score(y_test, prediction, average='weighted'), f1_score(y_test, prediction, average='weighted')))


# In[16]:


#minkowski mesafesine göre basit oylama yöntemine göre sonuçlar nedir?
knn = KNeighborsClassifier(n_neighbors=max_k_value,metric='minkowski')
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("Sınıflandırma işleminin ilk örnek sonucu: {}".format(prediction[0]))
print("{} komşuluk derecesine göre sınıflandırma işleminin doğruluk skoru: {}, precision skoru {}, recall skoru {}, f1 skoru {}"
      .format(knn.n_neighbors, knn.score(x_test, y_test), precision_score(y_test, prediction, average='weighted'),recall_score(y_test, prediction, average='weighted'), f1_score(y_test, prediction, average='weighted')))


# In[29]:


#manhattan uzaklığına göre ağırlıklı oylama sınıflandırma sonuçları k=3 için
knn = KNeighborsClassifier(n_neighbors=max_k_value,metric='minkowski',weights='distance')
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("Sınıflandırma işleminin ilk örnek sonucu: {}".format(prediction[0]))
print("{} komşuluk derecesine göre sınıflandırma işleminin doğruluk skoru: {}, precision skoru {}, recall skoru {}, f1 skoru {}"
      .format(knn.n_neighbors, knn.score(x_test, y_test), precision_score(y_test, prediction, average='weighted'),recall_score(y_test, prediction, average='weighted'), f1_score(y_test, prediction, average='weighted')))

