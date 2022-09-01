# Yazar: Dr. Elif KARTAL                            
# Tarih: 2022                                                        
# Konu: Destek Vektör Makineleri - DVM  (Support Vector Machines - SVM)
# Veri Seti: UCI Machine Learning Data Repository, Credit Approval Data Set
# https://archive.ics.uci.edu/ml/datasets/credit+approval

# 3.1. R Kutuphanelerinin Hazirligi

# install.packages("gplots")
# install.packages("fastDummies")
# install.packages("caret")
# install.packages("e1071")
library("gplots")
library("fastDummies")
library("caret")
library("e1071")

# 3.2. Veriyi Okuma
# Internette yer alan veri seti okunur
URL1 <- "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
veri <- read.table(file = URL1, header = FALSE, sep = ",", dec = ".", na.strings = "?", stringsAsFactors = T)

# 3.3. Veriyi Anlama 
# Sutun adlari eklenir
colnames(veri) <- paste0("A", seq(1,16,1))
# Veri ozeti alinir
summary(veri)
str(veri)


# 3.4. Veri Onisleme

# Kategorik nitelikler
kSut <- names(veri)[sapply(veri, is.factor)]

# Numerik nitelikler
nSut <- names(veri)[!(names(veri) %in% kSut)]

# 3.4.1. Eksik Verinin Kontrolu
# Hangi nitelikte kac adet eksik veri oldugu bulunur
bs <- colnames(veri)[colSums(is.na(veri)) > 0]
colSums(is.na(veri[,bs]))
# Eksik veri iceren niteliklerin indeks degerleri bulunur
sapply(veri[,bs], class)
bs_index <- which(names(veri) %in% bs) 

# 3.4.2. Eksik Verinin Tamamlanmasi
# Niteliklerdeki eksik veri A2de ortalama, digerlerinde ise en fazla tekrar eden kategori ile doldurulmustur
for(i in 1:ncol(veri)){
  if(i %in% bs_index){
    if(i == 2 | i == 14){
      veri[is.na(veri[,i]),i] <- round(mean(x = veri[,i], na.rm = TRUE),2)
    }else{
      veri[is.na(veri[,i]),i] <- names(which.max(table(veri[,i])))
    }
  }
}

# 3.4.3. Hedef Niteligin Analizlere Hazirlanmasi ve Dengesiz Veri Seti Kontrolu
table(veri$A16)
levels(veri$A16) <- c("red", "onay")
table(veri$A16)

# Hedef niteligin kategorilerini yeniden siralanir
levels(veri$A16)
veri$A16 <- factor(veri$A16, levels = c("onay", "red"))
levels(veri$A16)
table(veri$A16)

# Hedef niteligin kategorilerine ait frekans dagiliminin pasta grafigi ile gösterimi
frekanslar <- c(table(veri$A16))
yuzdeler <- round(((frekanslar * 100)/sum(frekanslar)), 2)
etiketler <- paste0(levels(veri$A16), " %", yuzdeler, " (n=", frekanslar, ")")
pie(frekanslar, labels = etiketler)

# 3.4.4. Surekli Nitelikler Arasinda Iliski Kontrolu
(kor <- cor(x = veri[,nSut], method = "pearson"))

# Isi haritasi ile korelasyonlarin gorsellestirilmesi
heatmap.2(kor, Rowv = FALSE, Colv = FALSE, dendrogram = "none", cellnote = round(kor,2), notecol = "black", key = FALSE, trace = 'none', margins = c(10,10))

# 3.4.5. One-Hot Encoding: Kategorik Niteliklerin Ikili (0/1) Sayisal Forma Donusturulmesi
veri <- dummy_cols(veri, select_columns = kSut[kSut != "A16"], remove_first_dummy = TRUE, remove_selected_columns = TRUE)

# Hedef nitelik veri setinde sona tasinir
veri$karar <- veri$A16
veri <- veri[,-c(7)]

# 3.5. Egitim ve Test Veri Setlerinin Olusturulmasi
# 70/30 Tabakali Hold-out (Stratified Hold-out)
set.seed(1)
e_ind <- createDataPartition(y = veri$karar, p = .70, list = FALSE)

egitimVS <- veri[e_ind,]
testVS <- veri[-e_ind,]

# 3.5.1. Veri Normalizasyonu 
# Min-max yontemi kullanilarak veri normalizasyonu
summary(egitimVS[,nSut])
summary(testVS[,nSut])
n_params <- caret::preProcess(egitimVS[,nSut], method = "range", rangeBounds = c(0,1))
egitimVS[,nSut] <- predict(n_params, egitimVS[,nSut])
testVS[,nSut] <- predict(n_params, testVS[,nSut])
summary(egitimVS[,nSut])
summary(testVS[,nSut])

# 3.6. Modelleme
# DVM algoritmasinin uygulanmasi
# 3.6.1. Doğrusal Kernel Fonksiyonu ile DVM Modelinin Olusturulmasi
# Varsayilan C = 1
DVM_d <- svm(formula = karar ~ ., data = egitimVS, type = "C-classification", kernel = "linear", scale = FALSE)

# Destek vektorleri
DVM_d$SV

# Destek vektorlerinin veri seti icindeki satir numaralari
DVM_d$index

# Dogrusal DVM modelinin negatif sabiti rho
DVM_d$rho

# Dogrusal DVM karar fonksiyonu agirliklari
W <- t(DVM_d$coefs) %*% DVM_d$SV

# Dogrusal DVM karar fonksiyonu sabitleri b
b <- -DVM_d$rho

# 3.6.2. Destek Vektorleri ve Siniflarin 2-D Grafikle Incelenmesi
# Temel Bilesenler Analizi uygulanarak egitim veri seti iki boyuta indirgenmistir.
# Siniflar renklerle, Destek Vektorleri ise arti sembolleri ile temsil edilmektedir.
X_egitim_ <- prcomp(egitimVS[,-38])
X_egitim_$x[,1]

plot(X_egitim_$x[,1:2],
     col = as.integer(egitimVS[,38]),
     pch = c("0","+")[rownames(egitimVS) %in% DVM_d$index + 1])

# 3.7. Model Tahminlerinin Elde Edilmesi

# 3.7.1. I. YOL: predict()/Karar fonksiyonu kullanarak
tahminler_p <- predict(DVM_d, newdata = testVS[,-38], decision.values = TRUE)

# # Karar fonksiyonundan elde edilen sonuclar
tahminler <- attr(tahminler_p, "decision.values")

head(tahminler)
head(tahminler_p)

# 3.7.2. II. YOL: X * t(W) + b formulunu kullanarak
tahminler_f <- as.matrix(testVS[,-38]) %*% t(W) + b
tahminler_f_p = factor(ifelse(tahminler_f < 0, "red", "onay"), levels = c("onay", "red"))

head(tahminler_f)
head(tahminler_f_p)

# 3.8. Model Performans Değerlendirmesi
DVM_d_cm <- confusionMatrix(data = tahminler_p, reference = testVS[,38], mode = "everything")
DVM_d_cm
(tp <- DVM_d_cm$table[1])
(fp <- DVM_d_cm$table[3])
(fn <- DVM_d_cm$table[2])
(tn <- DVM_d_cm$table[4])

# Dogruluk (Accuracy)
dogruluk = round(((tp + tn) / (tp + fp + fn + tn)),2)
# round(DVM_d_cm$overall["Accuracy"],2)

# Duyarlilik (Sensitivity)
duyarlilik = round((tp / (tp + fn)), 2)
# round(DVM_d_cm$byClass["Sensitivity"],2)

# Kesinlik (Positive Predictive Value / Precision)
kesinlik = round((tp / (tp + fp)), 2)
# round(DVM_d_cm$byClass["Precision"],2)

# F-Olcusu (F-measure)
FOlcusu = round(((2 * duyarlilik * kesinlik) / (duyarlilik + kesinlik)), 2)
# round(DVM_d_cm$byClass["F1"],2)

# Karmasiklik matrisinin cizimi
tablo <- as.table(matrix(DVM_d_cm$table, nrow = 2, byrow = TRUE))
fourfoldplot(tablo, color = c("#CC6666", "#99CC99"), 
             conf.level = 0, margin = 1, main = "Confusion Matrix")

# Siniflandirma raporu
DVM_d_cm$table

# 3.9. DVM Modeli Parametre Ayarı
DVM_a <- tune.svm(x = egitimVS[,-38], y = egitimVS[,38], kernel = "radial", cost = 2^(-1:5), gamma = c(0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05))

# Elde edilen performans degerleri
DVM_a$performances

# En iyi parametrelerle elde edilen model
DVM_a$best.model

# cost vs. gamma
plot(DVM_a)

# En iyi parametrelerle elde edilen tahminler
tahminler_DVM_a <- predict(DVM_a$best.model, newdata = testVS[,-38], decision.values = TRUE)
DVM_a_cm <- confusionMatrix(data = tahminler_DVM_a, reference = testVS[,38], mode = "everything")
DVM_a_cm