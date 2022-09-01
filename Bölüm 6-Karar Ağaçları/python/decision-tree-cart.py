
# coding: utf-8

# In[19]:


import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


#Veri okuma işlemini yapıyoruz.
dir_path = os.path.dirname(os.path.realpath('__file__'))
data = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
data.head()


# In[7]:


#Veri kümesindeki öznitelik isimlerini Türkçe'ye çeviriyoruz.
data = data.rename(columns={'Gender':'Cinsiyet','Age':'Yaş','Height':'Kilo','Weight':'Boy','family_history_with_overweight':'obezite_aile_geçmişi','FAVC':'yüksek_kalorili_gıda','FCVC':'sebze_tüketim_sıklığı','NCP':'ana_öğün_sayısı','CAEC':'öğünler_arası_gida_tüketimi','SMOKE':'sigara','CH2O':'günlük_su_tüketimi','SCC':'kalori_izleme','FAF':'fiziksel_aktivite_sıklığı','TUE':'teknoloji_cihazı_kullanma_süresi','CALC':'alkol','MTRANS':'ulaşım_aracı','NObeyesdad':'obezite'})
data.head()


# In[8]:


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

data.head()


# In[11]:


#Veri kümemizde bulunan bağımlı(sınıf) ve bağımsız(öznitelik) değişkenlerini belirtiyoruz.
y = data.obezite.values
x = data.drop(['obezite'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=1)


# In[12]:


#Sınfılandırma algoritmasının kurulumu yapıyoruz.
classifier = DecisionTreeClassifier(random_state=1)
classifier.fit(x_train, y_train)
prediction = classifier.predict(x_test)
print()
print("Sınıflandırma işleminin ilk örnek sonucu: {}".format(prediction[0]))
print("Sınıflandırma işleminin doğruluk skoru: {}, precision skoru: {}, \nrecall skoru: {}, f1 skoru: {}"
      .format(classifier.score(x_test, y_test), precision_score(y_test, prediction, average='weighted'),recall_score(y_test, prediction, average='weighted'), f1_score(y_test, prediction, average='weighted')))


# In[13]:


#Karmaşıklık Matrisi
cnf_matrix = confusion_matrix(y_test, prediction)
cm_df = pd.DataFrame(cnf_matrix,index = ['Normal Ağırlık', 'Kilolu Seviye I', 'Kilolu Seviye II','Obezite Tip I','Yetersiz Ağırlık','Obezite Tip II','Obezite Tip III'], columns = ['Normal Ağırlık', 'Kilolu Seviye I', 'Kilolu Seviye II','Obezite Tip I','Yetersiz Ağırlık','Obezite Tip II','Obezite Tip III'])
plt.figure(figsize=(12,8))
sns.heatmap(cm_df, annot=True, fmt='g')
plt.title('Karmaşıklık Matrisi', fontsize=16)
plt.ylabel('Gerçek Değerler',fontsize=16)
plt.xlabel('Tahmin Edilen Değerler',fontsize=16)
plt.show()


# In[15]:


#Oluşturulan karar ağacını görüyoruz.
fig = plt.figure(figsize=(10,20))
#max_depth parametresi ile ağacın gösterimdeki derinliği ayarlanmaktadır.
plot_tree(classifier, max_depth=2, fontsize=10, filled=True, feature_names=list(x_train.columns), class_names=list(dict.fromkeys(list(y_train))))
fig.savefig('imagename.png') #bu satır sadece çıktı elde etmek için.


# In[14]:


#Sisteme yeni bir test örneği veriyoruz.
prediction = classifier.predict([[0,72,2.74,47,1,1,1.0,1.0,2,1,7.0,1,1.0,1,7,8]])
prediction

