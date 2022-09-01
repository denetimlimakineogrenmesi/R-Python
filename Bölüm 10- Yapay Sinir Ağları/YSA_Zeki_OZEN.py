##############################
# Author:  Dr. Zeki ÖZEN
# Dataset: https://www.kaggle.com/datasets/sagnik1511/car-insurance-data
# Dataset  Collaborator: Sagnik Roy

# Problem: YSA ile Ikili Siniflandirma
##############################


##############################
# Gerekli kutuphaneler yukleniyor
##############################
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings

warnings.filterwarnings('ignore')
# veriyi manipule etmek icin
import pandas as pd
# matematiksel islemelr icin
import numpy as np
# veriyi bolumleme, performans degerlendirmesi icin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, auc, \
    precision_score, recall_score, f1_score, classification_report, roc_curve, plot_roc_curve, \
    precision_recall_curve, plot_precision_recall_curve, average_precision_score
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, PowerTransformer
# oversampling uygulamak icin
from imblearn.over_sampling import ADASYN
# ysa mimarisi kurmak icin
import tensorflow as tf
from keras import Sequential
from keras.layers import InputLayer, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam

# grafik cidirmek icin
from matplotlib import pyplot as plt
import seaborn as sns

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 15)

##############################
# Veri seti calisma ortamina yukleniyor
##############################
df = pd.read_csv('Car_Insurance_Claim.csv')

##############################
# 1. Veriyi Anlama
##############################
# Veri setinde gerek duyulmayan ID niteligi siliniyor
df = df.drop(['ID'], axis=1)

# Veri setinin boyutlarina (satir ve sutun sayilarina) bakiliyor
df.shape

# Veri seti nitelik adlarina bakiliyor
print(df.columns)

# Niteliklerin veri tiplerine bakiliyor
df.info()

# Veri setinin ozetine bakiliyor
print(df.describe().apply(lambda s: s.apply('{0:.2f}'.format)).T)
# print(df.describe(include="all"))

# Veri setinin ilk birkac satirina goz atiyoruz
df.head()

##############################
# 2. Veri On-isleme adimi
##############################

# Kategorik nitelikler float veri tipinden int veri tipine cevriliyor
df = df.astype({'VEHICLE_OWNERSHIP': 'int', 'MARRIED': 'int', 'CHILDREN': 'int', 'OUTCOME': 'int'})

####### Kayip deger (missing value) analizi #######
# veri setinde kayip deger olup olmadigi kontrol ediliyor
df.isna().sum()
# Kayip degerler, o niteligin ortalamasi ile dolduruluyor
df["CREDIT_SCORE"] = df["CREDIT_SCORE"].fillna(df["CREDIT_SCORE"].mean())
df["ANNUAL_MILEAGE"] = df["ANNUAL_MILEAGE"].fillna(df["ANNUAL_MILEAGE"].mean())
# kayip degerlerin tamamlandigini kontrol ediyoruz
df.isna().sum()

# Veri setinde tekrar eden satirlarin varligi kontrol ediliyor
df.duplicated().sum()
# Veri setinde tekrar eden satirlar siliniyor
df = df.drop_duplicates()
# Veri setinde tekrar eden satirlarin varligi tekrar kontrol ediliyor
df.duplicated().sum()

# Kategorik nitelikler dummy table haline getiriliyor
df = pd.get_dummies(df, drop_first=True)
df = pd.get_dummies(df, columns=['POSTAL_CODE'], drop_first=True)

# Veri setindeki niteliklerin histogram grafigi cizdiriliyor
df.hist(figsize=(15, 15))
plt.show()

# Veri setindeki niteliklerin korelasyon matrisi hesaplaniyor
kor = df.corr().round(2)

# Korelasyon matrisi, isi haritasi ile gorsellestiriliyor
plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(kor, dtype=bool))
cmap = sns.diverging_palette(230, 20)
sns.heatmap(kor, annot=True, vmin=-1, vmax=1, mask=mask, cmap=cmap)
plt.show()
# plt.savefig('heatmap.png')

####### Aykiri Deger (Outlier) Analizi #######
# Surekli niteliklerin kutu grafikleri cizdiriliyor
numerik_nitelikler = ['CREDIT_SCORE', 'ANNUAL_MILEAGE', 'SPEEDING_VIOLATIONS', 'DUIS', 'PAST_ACCIDENTS']
# kutu grafigi yanyana cizdirmek icin 1 satir 5 kolonluk grafik olusturuluyor
fig, ax = plt.subplots(1, 5, figsize=(10, 6))
plt.subplots_adjust(wspace=1)
colors = ['pink', 'lightblue', 'lightgreen', 'brown', 'red']
# Surekli niteliklerin kutu grafikleri ciziliyor
for x in range(5):
    sns.boxplot(data=df[numerik_nitelikler[x]], ax=ax[x], color=colors[x])
    ax[x].set_xlabel(numerik_nitelikler[x])
plt.show()

# Surekli niteliklerin frekanslari kontrol ediliyor
for kolon in numerik_nitelikler:
    print(df[kolon].value_counts())


# Aykiri deger tespit fonksiyonu
def outlier_sil(df, kolon):
    birinci_ceyrek = df[kolon].quantile(0.25)
    ucuncu_ceyrek = df[kolon].quantile(0.75)
    IQR = ucuncu_ceyrek - birinci_ceyrek
    dusuk = birinci_ceyrek - 1.5 * IQR
    yuksek = ucuncu_ceyrek + 1.5 * IQR
    yeni_df = df.loc[(df[kolon] > dusuk) & (df[kolon] < yuksek)]
    return yeni_df


# Normal dagilim gosteren numerik niteliklerin aykiri degerleri siliniyor
outlier_kontrol_edilecek_nitelikler = ['CREDIT_SCORE', 'ANNUAL_MILEAGE']
for kolon in outlier_kontrol_edilecek_nitelikler:
    df = outlier_sil(df, kolon)

# Aykiri degerlerin silinmesi sonrasi veri setinin satir ve sutun sayisi
df.shape

# Bagimsiz nitelikler X degiskenine, bagimli (hedef) nitelik y degiskenine ataniyor
X = df.drop(['OUTCOME'], axis=1)
y = df['OUTCOME']

# Hedef niteligin sinif dagilimi kontrol ediliyor
print(y.value_counts())

# Sinif dagilimi pasta grafigi ile goruntuleniyor
df['OUTCOME'].groupby(df['OUTCOME']).count().plot.pie(figsize=(5, 5), autopct='%1.1f%%', startangle=30.)
plt.show()
# print("Sigorta talep edenlerin yuzdesi: %" + str(round(y.value_counts()[0] * 100 / y.shape[0])))
# print("Sigorta talep etmeyenlerin yuzdesi: %" + str(round(y.value_counts()[1] * 100 / y.shape[0])))


# Veri seti rastgele bicimde %70 egitim ve %30 test veri seti olarak bolunuyor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(y_train.value_counts())
print(y_train.value_counts(normalize=True))

# Egitim veri seti smote sampling yapiliyor
# ADASYN SMOTE metodu ile egitim veri setinde sınıf dagilimlari dengeli hale getiriliyor
adasyn_over_sample = ADASYN(sampling_strategy='minority', random_state=42, n_neighbors=5)
X_train, y_train = adasyn_over_sample.fit_resample(X_train, y_train)

# Asiri ornekleme sonrasi hedef niteligin sinif dagilimlari tekrar kontrol ediliyor
print(y_train.value_counts())
print(y_train.value_counts(normalize=True))

# Egitim ve test veri setindeki numerik nitelikler 0-1 araliginda yeniden olcekleniyor
min_max_scaler = MinMaxScaler()
normalize_edilecek_nitelikler = ['ANNUAL_MILEAGE', 'SPEEDING_VIOLATIONS', 'PAST_ACCIDENTS', 'DUIS']
X_train[normalize_edilecek_nitelikler] = min_max_scaler.fit_transform(X_train[normalize_edilecek_nitelikler])
X_test[normalize_edilecek_nitelikler] = min_max_scaler.transform(X_test[normalize_edilecek_nitelikler])

##############################
# 2. Modelleme adimi
##############################
# Rasgele sayi ataniyor
tf.random.set_seed(42)
# Girdi katmanindan sonra sirasiyla 20 ve 5 noronlu iki katmanli bir mimari kurgulaniyor
ysa_model = None
ysa_model = Sequential()
ysa_model.add(InputLayer(input_shape=(X_train.shape[1],), name="Girdi_Katmani"))
ysa_model.add(Dense(20, activation='relu', name="Ara_Katman_1"))  # Birinci ara katman
ysa_model.add(BatchNormalization())
ysa_model.add(Dropout(0.3))
ysa_model.add(Dense(5, activation='relu', name="Ara_Katman_2"))  # İkinci ara katman
ysa_model.add(BatchNormalization())
ysa_model.add(Dropout(0.3))
ysa_model.add(Dense(1, activation='sigmoid', name="Cikti_Katmani"))  # Cikti katmani

# model parametreleri ayarlaniyor
ysa_model.compile(
    optimizer='sgd',
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['binary_accuracy']
)

# model egitiliyor
history = ysa_model.fit(x=X_train, y=y_train,
                        validation_split=0.2,
                        epochs=200,
                        batch_size=32,
                        verbose=1,
                        # use_multiprocessing = True
                        )
# modelin ozetine bakiliyor
ysa_model.summary()

# modelin egitim dogruluguna ve kayip degerine bakiliyor
test_loss, test_acc = ysa_model.evaluate(x=X_test, y=y_test, verbose=0)
# test_loss
test_acc

# Egitilmis YSA modeli test veri seti ile test ediliyor
predictions = ysa_model.predict(x=X_test)

# tahmin sonuclarina goz gezdiriliyor
print(predictions[0:5])

# tahmin sonuclari 0.5 degerinin uzerindeyse 1'e, altindaysa 0'a yuvarlaniyor
prediction_classes = [1 if prob > 0.5 else 0 for prob in np.ravel(predictions)]
print(prediction_classes[0:5])

# ConfusionMatrix ile modelin performans degerleri elde ediliyor
conf_mat = confusion_matrix(y_true=y_test, y_pred=prediction_classes)
print(conf_mat)

# Modelin performans raporu ciktilaniyor
print(classification_report(y_true=y_test, y_pred=prediction_classes))

# Muhim performans metrikleri ciktilaniyor
print(f'Accuracy: {accuracy_score(y_test, prediction_classes):.2f}')
print(f'Precision: {precision_score(y_test, prediction_classes):.2f}')
print(f'Recall: {recall_score(y_test, prediction_classes):.2f}')
print(f'F1Score: {f1_score(y_test, prediction_classes):.2f}')

history.history.keys()