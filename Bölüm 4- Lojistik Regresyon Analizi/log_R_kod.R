

install.packages("caret")
install.packages("dslabs")
install.packages("dplyr")
install.packages("pROC")

library(caret)
library(dslabs)
library(dplyr)
library(pROC)

veri <- Airline_Passenger_Satisfaction


veri$`Arrival Delay in Minutes` <- NULL
veri$...1 <- NULL
veri$id <- NULL
veri$`Inflight wifi service` <- NULL
veri$`Departure/Arrival time convenient` <- NULL
veri$`Ease of Online booking` <- NULL
veri$`Gate location` <- NULL
veri$`Food and drink` <- NULL
veri$`Online boarding` <- NULL
veri$`Seat comfort`<- NULL
veri$`Inflight entertainment` <- NULL
veri$`On-board service` <- NULL
veri$`Leg room service` <- NULL
veri$`Baggage handling` <- NULL
veri$`Checkin service` <- NULL
veri$`Inflight service` <- NULL
veri$Cleanliness <- NULL



veri$satisfaction <-factor(veri$satisfaction, levels = c("neutral or dissatisfied","satisfied"), labels = c(0,1))


str(veri)
summary(veri)




set.seed(123)
egitim_indeksi <- createDataPartition(veri$satisfaction, 
                                      p = .8, 
                                      list = FALSE, 
                                      times = 1)


egitim <- veri[egitim_indeksi,]
test <- veri[-egitim_indeksi,]

egitim_x <- egitim %>% dplyr::select(-satisfaction)
egitim_y <- egitim$satisfaction

test_x <- test %>% dplyr::select(-satisfaction)
test_y <- test$satisfaction


#Lojistik model
model = glm(satisfaction ~ .,data=egitim, family= "binomial")
summary(model)
levels(egitim$satisfaction)[2]


exp(coef(model))

#Vif deðerleri
car::vif(model)


predict(model)
head(predict(model))


head(predict(model, type = "response"))


model_tahmin_egitim_ol <- ifelse(predict(model, type = "response") > 0.5, "1","0")
head(model_tahmin_egitim_ol)
table(model_tahmin_egitim_ol)



Karsilastirma_egitim = cbind(model_tahmin_egitim_ol,egitim[,8])
Karsilastirma_egitim = data.frame(Karsilastirma_egitim)
class(Karsilastirma_egitim)
caret::confusionMatrix(as.factor(Karsilastirma_egitim[,1]),as.factor(Karsilastirma_egitim[,2]), mode="everything",positive="1")



model_tahmin_test_ol <- ifelse(predict(model,newdata = test_x, type = "response") > 0.5, "1","0")
head(model_tahmin_test_ol)
table(model_tahmin_test_ol)


Karsilastirma_test = cbind(model_tahmin_test_ol,test[,8])
Karsilastirma_test = data.frame(Karsilastirma_test)
class(Karsilastirma_test)
caret::confusionMatrix(as.factor(Karsilastirma_test[,1]),as.factor(Karsilastirma_test[,2]), mode="everything",positive="1")


#Roc
model_tahmin_test_ol <- predict(model, newdata = test_x, type = "response")
roc(test_y ~ model_tahmin_test_ol, plot = TRUE, print.auc = TRUE)



tahmin <- ifelse(predict(model,newdata = test_x, type = "response") > 0.5, "1","0")
head(tahmin)
gercek<-test$satisfaction
tablo <- data.frame(gercek , tahmin)
head(tablo, 10) 


