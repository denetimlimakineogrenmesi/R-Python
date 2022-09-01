

install.packages("tidyverse")
install.packages("caret") 
install.packages("randomForest")

library(tidyverse)
library(caret) 
library(randomForest)

veri <- telcoCustomerChurn 
sum(is.na(veri))
str(veri)
summary(veri)

veri$TotalCharges[is.na(veri$TotalCharges)] <- mean(veri$TotalCharges, na.rm = TRUE)

veri$customerID <- NULL 
veri$gender <- NULL
veri$SeniorCitizen <- NULL
veri$Partner <- NULL
veri$Dependents <- NULL
veri$PhoneService <- NULL
veri$MultipleLines <- NULL
veri$StreamingTV <- NULL
veri$StreamingMovies <- NULL
veri$MonthlyCharges <- NULL
veri$PaperlessBilling <- NULL 

str(veri)

veri$Churn <- factor(veri$Churn, 
                     levels = c('No', 'Yes'), 
                     labels = c(0,1)) 
summary(veri$Churn)

set.seed(123)
egitim_indeks <- createDataPartition(veri$Churn, 
                                     p = .80, 
                                     list = FALSE, 
                                     times = 1)

egitim <- veri[egitim_indeks,]
test <- veri [-egitim_indeks,] 

egitim_x <- egitim %>% dplyr::select(-Churn) 
egitim_y<- egitim$Churn
test_x <- test %>% dplyr::select(-Churn)
test_y <- test$Churn

rf <- randomForest(Churn ~., data = egitim, importance = TRUE, confusion = TRUE, ntree= 1000)
rf

plot(rf)

importance(rf) 

varImpPlot(rf)

predict(rf, test)
confusionMatrix(predict(rf, test_x),test_y, mode = "everything", positive = "0")

tablo <- data.frame(gerçek = test_y, tahmin = predict(rf, test_x))
head(tablo, 10)
