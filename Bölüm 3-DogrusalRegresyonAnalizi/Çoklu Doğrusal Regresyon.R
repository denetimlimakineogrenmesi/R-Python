install.packages("ISLR")
install.packages("tidyverse")
install.packages("caret")
install.packages("ResourceSelection")
install.packages("Rcmdr")  
install.packages("hardhat")
install.packages("magrittr") 
install.packages("dplyr")

library(ISLR)
library(tidyverse)
library(caret)
library(ResourceSelection)
library(Rcmdr)   
library(hardhat)
library(magrittr)
library(dplyr)

veri<- X2019
veri$`Overall rank`<- NULL
veri$`Country or region`<- NULL

str(veri)
summary(veri)

install.packages("Rcmdr") 
library(Rcmdr)

cor(veri[,c("Freedom.to.make.life.choices","GDP.per.capita","Generosity",
            "Healthy.life.expectancy","Perceptions.of.corruption","Social.support")], 
    use="complete")

library(caret)
library(magrittr)

set.seed(123)
egitim_indeks <- createDataPartition(veri$Score, 
                                     p = .8, 
                                     list = FALSE, 
                                     times = 1)

egitim <- veri[egitim_indeks,]  
test  <- veri[-egitim_indeks,]

egitim_X <- egitim %>% dplyr::select(-Score)
egitim_Y <- egitim$Score
test_X <- test %>% dplyr::select(-Score)
test_Y <- test$Score

head (egitim, 3)
head (test, 3)

model <- lm(formula <- Score ~ ., data <- egitim)

summary(model)

car::vif(model)

coef(model)

tahmin_egitim <- predict(model, newdata <- egitim)
head (tahmin_egitim, 10)

sonuc_egitim <- data.frame (gercek_egitim <- egitim$Score, tahmin_egitim)
head (sonuc_egitim, 10)

SSE_egitim <- sum((gercek_egitim-tahmin_egitim)^2)
MAE_egitim <- mean(abs(gercek_egitim-tahmin_egitim))
MAPE_egitim <- mean(abs((gercek_egitim-tahmin_egitim))/gercek_egitim)
RMSE_egitim <- sqrt (mean((gercek_egitim-tahmin_egitim)^2))
ortalama_egitim <- mean(gercek_egitim)
TSS_egitim <- sum((gercek_egitim-ortalama_egitim)^2)
Rkare_egitim <- (1-(SSE_egitim/TSS_egitim))
n_egitim <- nrow(egitim)
k_egitim <- ncol(egitim)-1
DRkare_egitim <- 1-(((1-Rkare_egitim)*(n_egitim-1))/(n_egitim-k_egitim-1))

SSE_egitim
MAE_egitim
MAPE_egitim
RMSE_egitim
Rkare_egitim
DRkare_egitim

tahmin_test <- predict(model, newdata <- test)
head (tahmin_test, 10) 
sonuc_test <- data.frame (gercek_test <-test$Score, tahmin_test)
head (sonuc_test, 10)


SSE_test <- sum((gercek_test-tahmin_test)^2)
MAE_test <- mean(abs(gercek_test-tahmin_test))
MAPE_test <- mean(abs((gercek_test-tahmin_test))/gercek_test)
RMSE_test <- sqrt (mean((gercek_test-tahmin_test)^2))
ortalama_test <- mean(gercek_test)
TSS_test <- sum((gercek_test-ortalama_test)^2)
Rkare_test <- (1-(SSE_test/TSS_test))
n_test <- nrow(test)
k_test <- ncol(test)-1
DRkare_test <- 1-(((1-Rkare_test)*(n_test-1))/(n_test-k_test-1))

SSE_test
MAE_test
MAPE_test
RMSE_test
Rkare_test
DRkare_test

         


