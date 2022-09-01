


install.packages("ISLR")
install.packages("tidyverse")
install.packages("caret")
install.packages("randomForest") 
install.packages("forecast")

library(ISLR)
library(tidyverse)
library(caret)
library(randomForest) 
library(forecast)

veri <- Credit
str(veri)
summary(veri)

veri$ID <- NULL 
veri$Gender <- NULL
veri$Ethnicity <-NULL

set.seed(123)
egitim_indeks <- createDataPartition(veri$Balance, 
                                     p = .8, 
                                     list = FALSE, 
                                     times = 1)

egitim <- veri[egitim_indeks,]  
test  <- veri[-egitim_indeks,]

egitim_x <- egitim %>% dplyr::select(-Balance)
egitim_y <- egitim$Balance

test_x <- test %>% dplyr::select(-Balance)
test_y <- test$Balance

rf <- randomForest(Balance ~., data = egitim, importance = TRUE, ntree = 1000)
rf

plot(rf)

importance(rf)

varImpPlot(rf)

tahmin<-predict(rf, test_x)
defaultSummary(data.frame(obs = test_y, pred = tahmin))

sonuc <- accuracy(test$Balance, tahmin)
sonuc

tablo <- data.frame(gerçek = test_y, tahmin )
head(tablo, 10)
