# -*- coding: utf-8 -*-
"""
Created on 2022
@author: Elif Kartal
@title: Destek Vektör Makineleri - DVM  (Support Vector Machines - SVM)
Veri Seti: UCI Machine Learning Data Repository, Credit Approval Data Set
https://archive.ics.uci.edu/ml/datasets/credit+approval
"""

# 2.1. Python Kutuphanelerinin Hazirligi
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# 2.2. Veriyi Okuma
# Internette yer alan veri seti okunur
URL1 = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
veri = pd.read_csv(URL1, header=None)

# 2.3. Veriyi Anlama
# Sutun adlari eklenir
veri.columns = [f'A{i}' for i in range(1, 17)]
# Veri ozeti alinir
veri.describe(include='all')
veri.dtypes

# 2.4. Veri Onisleme

# Soru isareti yerine None atanir
veri = veri.replace(["?"], None)

# Surekli niteliklere float, kategoriklere ise category veri tipi atanir
veri["A2"] = veri["A2"].astype("float")
veri["A14"] = veri["A14"].astype("float")
for col in veri.columns:
    if veri.dtypes[col] == 'O':
        veri[col] = veri[col].astype('category')

# Kategorik nitelikler
kSut = veri.columns[veri.dtypes == "category"]

# Numerik nitelikler
nSut = veri.columns[~(veri.dtypes == "category")]

# 2.4.1. Eksik Verinin Kontrolu
# Hangi nitelikte kac adet eksik veri oldugu bulunur
print(veri.isnull().sum())
bs = veri.columns[veri.isnull().any()]
# Eksik veri iceren niteliklerin indeks degerleri bulunur
bs_index = list(map(lambda col: veri.columns.get_loc(col), bs))

# 2.4.2. Eksik Verinin Tamamlanmasi
# Niteliklerdeki eksik veri A2de ortalama, digerlerinde ise en fazla tekrar eden kategori ile doldurulmustur
for i in range(0, len(bs_index)):
    if (bs[i - 1] == "A2") or (bs[i - 1] == "A14"):
        veri[bs[i - 1]].fillna(veri[bs[i - 1]].mean().round(2), inplace=True)
    else:
        veri[bs[i - 1]].fillna(veri[bs[i - 1]].mode()[0], inplace=True)

# 2.4.3.  Hedef Niteligin Analizlere Hazirlanmasi ve Dengesiz Veri Seti Kontrolu
veri["A16"].value_counts()
veri["A16"] = np.where(veri["A16"] == "+", 1, 0)
veri["A16"].value_counts()

# Hedef niteligin kategorilerine ait frekans dagiliminin pasta grafigi ile gösterimi
frekanslar = veri["A16"].value_counts()
yuzdeler = ((frekanslar * 100)/np.sum(frekanslar)).round(2).values
etiketler = [x + str(y) + " (n=" + str(z) + ")" for x, y, z in zip(["red (0) %", "onay (1) %"], yuzdeler, list(frekanslar))]
renkler = sns.color_palette("bright")
plt.pie(yuzdeler, labels = etiketler, colors = renkler)
plt.show()

# 2.4.4. Surekli Nitelikler Arasinda Iliski Kontrolu
# Numerik nitelikler arasinda iliski (Pearson)
kor = veri.corr()[nSut].round(2)
kor
# Isi haritasi ile korelasyonlarin gorsellestirilmesi
sns.heatmap(kor, annot = True, square=True,  cmap="Blues")

# 2.4.5. One-Hot Encoding: Kategorik Niteliklerin Ikili (0/1) Sayisal Forma Donusturulmesi
dummy_nitelikler = pd.get_dummies(veri[kSut[0:len(kSut)]], drop_first=True)
veri = pd.concat([veri.drop(kSut, axis=1), dummy_nitelikler], axis=1)

# Hedef nitelik veri setinde sona tasinir
veri["karar"] = veri["A16"]
veri["karar"] = veri["karar"].astype("int")
veri = veri.drop(["A16"], axis=1)

# Veri ozeti
for i in range(0,(len(veri.columns)),6):
    print(veri.iloc[:, i:i+6].describe(include="all"))
veri.columns # Sutun adlari
veri.dtypes # Veri tipleri

# 2.5. Egitim ve Test Veri Setlerinin Olusturulmasi
# 70/30 Tabakali Hold-out (Stratified Hold-out)
X_egitim, X_test, y_egitim, y_test = train_test_split(veri.iloc[:, 0:37], veri.karar, test_size=0.3, random_state=1)

# 2.5.1. Veri Normalizasyonu
# 3.1) Min-max yontemi kullanilarak veri normalizasyonu
X_egitim[nSut].describe().T
X_test[nSut].describe().T
sclr = MinMaxScaler()
X_egitim[nSut] = sclr.fit_transform(X_egitim[nSut])
X_test[nSut] = sclr.transform(X_test[nSut])
X_egitim[nSut].describe().T
X_test[nSut].describe().T


# 2.6. Modelleme
# 2.6.1. Dogrusal Kernel Fonksiyonu ile DVM Modelinin Oluşturulması
DVM_d = svm.SVC(kernel='linear')
DVM_d.fit(X_egitim, y_egitim)

# Destek vektorleri
DVM_d.support_vectors_

# Destek vektorlerinin veri seti icindeki indeksleri
DVM_d.support_

# Her bir sinifin destek vektorleri sayisi
DVM_d.n_support_

# Dogrusal DVM karar fonksiyonu agirliklari
W = DVM_d.coef_

# Dogrusal DVM karar fonksiyonu sabitleri b
b = DVM_d.intercept_

# 2.6.2. Destek Vektorleri ve Siniflarin 2-D Grafikle Incelenmesi
# Temel Bilesenler Analizi uygulanarak egitim veri seti iki boyuta indirgenmistir.
# Siniflar renklerle, Destek Vektorleri ise arti sembolleri ile temsil edilmektedir.
pca = PCA(n_components=2)
X_egitim_ = pca.fit_transform(X_egitim)

# Grafik cizimi
isaretciler = list()
for i in X_egitim.index:
    if i in DVM_d.support_:
        isaretciler.append("+")
    else:
        isaretciler.append("o")
renkler = np.where(y_egitim == 0, "hotpink", "blue")

for xp, yp, col, m in zip(X_egitim_[:,0], X_egitim_[:,1], renkler, isaretciler):
   plt.scatter(xp, yp, marker=m, s=50, c=col)
plt.xlabel("Birinci Temel Bilesen")
plt.ylabel("Ikinci Temel Bilesen")
plt.show()

# 2.7. Model Tahminlerinin Elde Edilmesi

# 2.7.1. I. YOL: predict()/Karar fonksiyonu kullanarak
tahminler_p = DVM_d.predict(X_test)

# Karar fonksiyonundan elde edilen sonuclar
tahminler = DVM_d.decision_function(X_test)
tahminler[0:5]
tahminler_p[0:5]

# 2.7.2. II.YOL: X * t(W) + b formulunu kullanarak
tahminler_f = np.array(np.dot(np.matrix(X_test), np.transpose(W)) + b).reshape(207,)

# Model tahminlerinin sinif degeri haline getirilmesi
tahminler_f_p = np.where(tahminler_f < 0, 0, 1).reshape(207,)
tahminler_f[0:5]
tahminler_f_p[0:5]

# 2.8. Model Performans Degerlendirmesi
my_cm = confusion_matrix(y_true = y_test, y_pred = tahminler_p, labels = [0, 1])
my_cm
my_cm_p = plot_confusion_matrix(DVM_d, X_test, y_test,
                                labels = [0, 1],
                                display_labels = ["0", "1"],
                                cmap = plt.cm.Blues)
my_cm_p.ax_.set_title("Confusion Matrix")

tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=tahminler_p, labels=[0, 1]).reshape(-1)
print("True Positives = ", tp)
print("False Positives = ", fp)
print("False Negatives = ", fn)
print("True Negatives = ", tn)

# Dogruluk (Accuracy)
dogruluk = ((tp + tn) / (tp + fp + fn + tn)).round(2)
# sklearn.metrics.accuracy_score(y_true=y_test, y_pred=tahminler_p).round(2)

# Duyarlilik (Sensitivity)
duyarlilik = (tp / (tp + fn)).round(2)
# sklearn.metrics.recall_score(y_true=y_test, y_pred=tahminler_p).round(2)

# Kesinlik (Positive Predictive Value / Precision)
kesinlik = (tp / (tp + fp)).round(2)
# sklearn.metrics.precision_score(y_true=y_test, y_pred=tahminler_p).round(2)

# F-Olcusu (F-measure)
FOlcusu = ((2 * duyarlilik * kesinlik) / (duyarlilik + kesinlik)).round(2)
# sklearn.metrics.f1_score(y_true=y_test, y_pred=tahminler_p).round(2)

print("Dogruluk = ", dogruluk)
print("Duyarlilik = ", duyarlilik)
print("Kesinlik = ", kesinlik)
print("F-Olcusu = ", FOlcusu)
print(classification_report(y_true=y_test, y_pred=tahminler_p, target_names=["0", "1"]))


# 2.9. DVM Modeli Parametre Ayari
parametreler = {"C": [0.5, 1.0, 2.0, 4.0, 8.0, 16, 32], "gamma": [0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05], 'kernel': ['rbf', 'poly', 'sigmoid']}
DVM_a = GridSearchCV(svm.SVC(), parametreler, refit=True, verbose=2)
DVM_a.fit(X_egitim,y_egitim)

# En iyi model parametreleri ve performansi
print(DVM_a.best_estimator_)
print(DVM_a.best_score_.round(2))

# En iyi parametrelerle elde edilen tahminler
tahminler_g = DVM_a.predict(X_test)
print(classification_report(y_true=y_test, y_pred=tahminler_g, target_names=["0", "1"]))