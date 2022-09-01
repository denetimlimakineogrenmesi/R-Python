##############################
# Author:  Dr. Zeki ÖZEN
# Dataset: https://www.kaggle.com/datasets/sagnik1511/car-insurance-data
# Dataset  Collaborator: Sagnik Roy

# Problem: YSA ile Ikili Siniflandirma
##############################

# calisma klasoru ayarlaniyor
# veri setinin csv dosyasi R kodunun oldugu klasorde bulunmalidir
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# gereken kutuphaneler sistemde kurulu degilse kuruluyor
# install.packages("clusterSim")
# install.packages("caret")
# install.packages("smotefamily")
# install.packages("fastDummies")
# install.packages("neuralnet")

# gereken kutuphaneler calisma ortamina dahil ediliyor
library(clusterSim)
library(caret)
library(smotefamily)
library(fastDummies)
#library(neuralnet)
library(RSNNS)



# Veri seti calisma ortamina yukleniyor
df <- read.csv(file="Car_Insurance_Claim.csv", header=TRUE, sep=",")


##############################
# 1. Veriyi Anlama
##############################

# Veri setinde gerek duyulmayan ID niteligi siliniyor
df$ID <- NULL

# Veri setinin boyutlarina (satir ve sutun sayilarina) bakiliyor
dim(df)

# Veri seti nitelik adlarina bakiliyor
colnames(df)

# Niteliklerin veri tiplerine bakiliyor
str(df)

# Veri setinin ozetine bakiliyor
summary(df)

# Veri setinin ilk birkac satirina goz atiyoruz
head(df)


##############################
# 2. Veri On-isleme
##############################
# veri setinde kayip deger olup olmadigi kontrol ediliyor
sum(is.na(df))


# kayip degerler, o niteligin ortalamasi ile dolduruluyor
df$CREDIT_SCORE[is.na(df$CREDIT_SCORE)] <- mean(df$CREDIT_SCORE, na.rm = TRUE)
df$ANNUAL_MILEAGE[is.na(df$ANNUAL_MILEAGE)] <- mean(df$ANNUAL_MILEAGE, na.rm = TRUE)

# kayip degerlerin tamamlandigini kontrol ediyoruz
sum(is.na(df))

# veri setinde tekrar eden satirlarin varligi kontrol ediliyor
sum(duplicated(df))
# veri setinde tekrar eden satirlar siliniyor
df = unique(df)

# Hedef nitelik numerik veri tipine cevriliyor
df[, 'OUTCOME'] <- as.numeric(df[, 'OUTCOME'])

numerik_nitelikler <-  c('CREDIT_SCORE', 'ANNUAL_MILEAGE', 'SPEEDING_VIOLATIONS', 'DUIS', 'PAST_ACCIDENTS')
kategorik_donusecek_nitelikler <- colnames(df)[! colnames(df) %in% numerik_nitelikler]

# kategorik nitelikler dummy table haline getiriliyor
df <- dummy_cols(df, select_columns = kategorik_donusecek_nitelikler, remove_first_dummy = TRUE, remove_selected_columns = TRUE)

# kolon isimlerinde duzeltme yapiliyor
colnames(df) <- c(  "CREDIT_SCORE", "ANNUAL_MILEAGE", "SPEEDING_VIOLATIONS", "DUIS",
                      "PAST_ACCIDENTS", "AGE_26_39", "AGE_40_64", "AGE_UPPER_65",
                      "GENDER", "RACE", "DRIVING_EXPERIENCE_10_19Y", "DRIVING_EXPERIENCE_20_29Y",
                      "DRIVING_EXPERIENCE_UPPER_30Y", "EDUCATION_NONE", "EDUCATION_UNIVERSITY", "INCOME_POVERTY",
                      "INCOME_UPPER_CLASS", "INCOME_WORKING_CLASS", "VEHICLE_OWNERSHIP", "VEHICLE_YEAR_BEFORE_2015",
                      "MARRIED", "CHILDREN", "POSTAL_CODE_21217", "POSTAL_CODE_32765",
                      "POSTAL_CODE_92101", "VEHICLE_TYPE", "OUTCOME")



# aykiri deger tespit fonksiyonu
outlier_sil <- function (df, kolon){

  birinci_ceyrek <- quantile(df[, kolon], .25)
  ucuncu_ceyrek  <- quantile(df[, kolon], .75)
  IQR <- IQR(df[, kolon])
  
  dusuk <- birinci_ceyrek - 1.5 * IQR
  yuksek <- ucuncu_ceyrek + 1.5 * IQR
  
  yeni_df <- subset(df, df[, kolon] > dusuk & df[, kolon] < yuksek)
  return (yeni_df)
}

# normal dagilim gosteren numerik niteliklerin aykiri degerleri siliniyor
outlier_kontrol_edilecek_nitelikler = c('CREDIT_SCORE', 'ANNUAL_MILEAGE')
for ( kolon in outlier_kontrol_edilecek_nitelikler)
  df <- outlier_sil(df, kolon)



# Hedef nitelikteki sinif degerlerinin sayilari kontrol ediliyor
table(df$OUTCOME)

# Sinif dagilimi pasta grafigi ile goruntuleniyor
pie(table(df$OUTCOME), 
    labels = paste0(c("0","1"),
                    " - ", 
                    table(df$OUTCOME),
                    " (%",round(prop.table(table(df$OUTCOME))*100), ")"), 
    col=c("pink", "light green"),
    main="Veri Seti Sýnýf Deðerlerinin Pasta Grafiði")



# Veri seti rastgele bicimde %70 egitim ve %30 test veri seti olarak bolunuyor
set.seed(1)
egitimIndisleri <- createDataPartition(y = df$OUTCOME, p = .70, list = FALSE) 
train <- df[egitimIndisleri,]
test <- df[-egitimIndisleri,]



# Egitim veri seti smote sampling yapiliyor
# ADASYN SMOTE metodu ile egitim veri setinde sýnýf dagilimlari dengeli hale getiriliyor
adasyn_over_sample <- NULL

adasyn_over_sample <- smotefamily::ADAS( X = train[, names(train) != "OUTCOME"] ,
                                         target = train[, 'OUTCOME'],
                                         K=5)$data


# ADAS fonksiyonu hedef niteligin adini class olarak degistiriyor
# bu nedenle hedef niteliðin adýný eski ismi olan OUTCOME olarak degistiriliyor
names(adasyn_over_sample)[27] <- "OUTCOME"

table(train[, 'OUTCOME'])
table(adasyn_over_sample[, 'OUTCOME'])
train <- adasyn_over_sample

# oversampling sonrasi egitim ve test veri setlerinin sinif dagilimi goruntuleniyor
table(train$OUTCOME)
table(test$OUTCOME)

train[, 'OUTCOME'] <- as.numeric(train[, 'OUTCOME'])
test[, 'OUTCOME'] <- as.numeric(test[, 'OUTCOME'])

# egitim ve test veri setindeki numerik nitelikler 0-1 araliginde yeniden olcekleniyor
normalize_edilecek_nitelikler = c('ANNUAL_MILEAGE', 'SPEEDING_VIOLATIONS', 'PAST_ACCIDENTS', 'DUIS')

n_params <- caret::preProcess(train[, normalize_edilecek_nitelikler], method = "range", rangeBounds = c(0,1))
train[, normalize_edilecek_nitelikler] <- predict(n_params, train[, normalize_edilecek_nitelikler])
test[, normalize_edilecek_nitelikler] <- predict(n_params, test[, normalize_edilecek_nitelikler])


# YSA modelini kurmak ve egitmek icin RSNNS paketi kullanilmistir
ysa_model <- NULL
ysa_model <- mlp(x=train[, -27], 
             y=train$OUTCOME, 
             size=c(20,5), 
             learnFuncParams=c(0.1), 
             maxit=200,
             learnFunc = "Std_Backpropagation",
             hiddenActFunc = "Act_Logistic",
           )

summary(ysa_model)


# Egitilmis Yapay sinir agi modeli test veri seti ile test ediliyor
tahminler <- NULL
tahminler <- predict (object=ysa_model, newdata=test[, -27])

# tahmin sonuclarina goz gezdiriliyor
head(tahminler)


# tahmin sonuclari 0.5 degerinin uzerindeyse 1'e, altindaysa 0'a yuvarlaniyor
kategorik_tahminler <- NULL
for (i in 1:nrow(tahminler)) {
  if ( (tahminler[i,]) >= 0.5) kategorik_tahminler[i] <- 1 else kategorik_tahminler[i] <- 0
}

head(kategorik_tahminler)
# Table fonksiyonu ile modelin tahminlerine goz atabiliriz
table(kategorik_tahminler, test$OUTCOME)



# confusionMatrix fonksiyonu kategorik karsilastirma yapmak icin
# tahmin sonuclarinda ve asil degerlerde bir sinif degerinin 
# referans alinmasini istemektedir
# Bu nedenle hem tahmin sonuclarinda hem de asil degerlerde 
# kisinin sigorta talep etme durumunu ifade eden 1 sinifini
# referans aliyoruz.

kategorik_tahminler <- as.factor(kategorik_tahminler)
kategorik_tahminler <- relevel(kategorik_tahminler, ref="1")
  
test$OUTCOME <- as.factor(test$OUTCOME)
test$OUTCOME <- relevel(test$OUTCOME, ref="1")

# ConfusionMatrix ile modelin performans degerleri elde ediliyor 
cMatris<- NULL
cMatris<- caret::confusionMatrix(data =  kategorik_tahminler, reference = test$OUTCOME, positive = "1"  )
cMatris


# Modelin performans raporu ciktilaniyor
cMatris$table


# Muhim performans metrikleri ciktilaniyor
print(round(cMatris$overall["Accuracy"],2))
print(round(cMatris$byClass["Sensitivity"],2))
print(round(cMatris$byClass["Precision"],2))
print(round(cMatris$byClass["F1"],2))


#Confussion Matrix Grafigi cizdiriliyor
fourfoldplot(cMatris$table, color = c("#CC6666", "#99CC99"), conf.level = 0, margin = 1, main = "Confusion Matrix")

