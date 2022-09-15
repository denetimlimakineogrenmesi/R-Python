library(tidyverse) #for graphing and data manipulation 
library(caTools) #for partitioning the data
library(MASS) #for LDA
Social_Network_Ads <- read_csv("LDA/Social_Network_Ads.csv")
veri <- Social_Network_Ads
veri$satisfaction <- factor(veri$Purchased, levels = c("not purchased","purchased"), labels = c(0,1))
summary(veri)

Social_Network_Ads <- Social_Network_Ads[, -1]
set.seed(123)
split = sample.split( Social_Network_Ads$Purchased, SplitRatio = 0.8)
training_set = subset(Social_Network_Ads, split == TRUE)
test_set = subset(Social_Network_Ads, split == FALSE)
summary(training_set)
hist_age <- ggplot(training_set, aes(x = Age)) + geom_histogram() + labs(title = "Distribution of Market Age")
hist_salary <- ggplot(training_set, aes(x = EstimatedSalary)) + geom_histogram(bins = 40) + labs(title = "Distribution of Market Salary")
qq_salary <- ggplot(training_set$EstimatedSalary)
qq_age <-ggplot(training_set$Age)
gridExtra::grid.arrange(hist_age,qq_age, hist_salary, qq_salary, nrow = 2)
lda_model <- lda(Purchased ~ Age + EstimatedSalary, data = training_set)
lda_model
plot(lda_model)
lda_predict <- predict(lda_model, test_set)
table(test_set$Purchased, lda_predict$class)
cat("Test Model Accuracy:", mean(lda_predict$class == test_set$Purchased)*100, "%")

