# ---
# title: "Own project - indian liver"
# author: "Florian Kern"
# date: "14 June 2019"
# output: pdf_document
# github repo: https://github.com/kernf/own_project_indian_liver
# comment: R script file for the capstone own project - indian liver.
# ---

# Load libraries:
library(dplyr) # a grammar for data manipulation
library(ggplot2) # grammar of graphics
library(caret) # classification and regression training package
library(reshape2) # data reshaping

# Data was downloaded from: https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset)
# Load and read the data from csv file (provided in github):
liver_data <- read.csv("indian_liver_patient.csv", sep = ",", header = TRUE)

# data set structure
str(liver_data)

# data set summary
summary(liver_data)

# vars in dataset
colnames(liver_data)

# disease distribution
table(liver_data$Dataset)

# gender distribution

# factorize gender
liver_data$Gender <- as.factor(liver_data$Gender)
class(liver_data$Gender)

# plot age histogram
histogram(liver_data$Age, main = "Age histogram", xlab = "Age", ylab = "%", col = "white")

# change column name
colnames(liver_data)[colnames(liver_data) == "Dataset"] = "Liver_Disease"
colnames(liver_data)

# change disease status to 0 an 1
liver_data$Liver_Disease[liver_data$Liver_Disease == 2] = 0
table(liver_data$Liver_Disease)

# factorize disease status
liver_data$Liver_Disease = as.factor(liver_data$Liver_Disease)
str(liver_data)

liver_data %>% ggplot(aes(Age, color= Liver_Disease)) +
  geom_freqpoly(bins=15)
liver_data %>% ggplot(aes(Age, color= Gender)) +
  geom_freqpoly(bins=15)

# check for NAs
sapply(liver_data, function(x) sum(is.na(x)))

# calculate the percentage of NAs
sum(is.na(liver_data)/nrow(liver_data))

# remove NA entries
liver_data = na.omit(liver_data)
dim(liver_data)

# calculate the correlations of variables without Gender and Disease Status
correlations <- cor(liver_data[,c(-2,-11)])
correlations

# plot the correlation triangle as heatmap
# function to get lower triangle of the correlation matrix
get_lower_tri<-function(cormat){
    cormat[upper.tri(cormat)] <- NA
    return(cormat)
  }
# function to get upper triangle of the correlation matrix
get_upper_tri <- function(cormat){
    cormat[lower.tri(cormat)]<- NA
    return(cormat)
  }
  
# function to reorder data
reorder_cormat <- function(cormat){
# Use correlation between variables as distance
dd <- as.dist((1-cormat)/2)
hc <- hclust(dd)
cormat <-cormat[hc$order, hc$order]
}

# reorder the correlation matrix
cormat <- round(reorder_cormat(correlations),2)
# get the upper triangle
upper_tri <- get_upper_tri(cormat)
# melt the correlation matrix
melted_cormat <- melt(upper_tri, na.rm = TRUE)
# create a ggheatmap
ggheatmap <- ggplot(melted_cormat, aes(Var2, Var1, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0,
                       limit = c(-1,1), space = "Lab", name="Pearson\nCorrelation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 8, hjust = 1)) +
  coord_fixed()

# add correlation coefficient labels onto the heatmap
ggheatmap + 
  geom_text(aes(Var2, Var1, label = value), color = "black", size = 3) +
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank(),
    axis.ticks = element_blank(),
    legend.justification = c(1, 0),
    legend.position = c(0.5, 0.7),
    legend.direction = "horizontal") +
  guides(fill = guide_colorbar(barwidth = 5, barheight = 1, title.position = "top", title.hjust = 0.5))

# find highly correlated variables
high_coor_cols <- findCorrelation(correlations, cutoff = 0.7, names = TRUE)
high_coor_cols

# remove highly correlated (redundant) data
liver_data <- liver_data[, !names(liver_data) %in% high_coor_cols]
dim(liver_data)

# set a seed
set.seed(1)
# split data set into train and test
index <- createDataPartition(liver_data$Liver_Disease, p = 0.8, list = FALSE)
train = liver_data[index,]
test = liver_data[-index,]
dim(train)
dim(test)

# remove gender factor
train$Gender <- as.numeric(train$Gender)
test$Gender <- as.numeric(test$Gender)

########### 
# Results #
###########
# Naive Bayes
# set seed
set.seed(1)
# fit/train the model
model1_nb_fit <- train(Liver_Disease ~ ., data = train, method = "nb")
model1_nb_fit

# var importance
model1_nb_imp <- varImp(model1_nb_fit, scale = FALSE)
model1_nb_imp

# plot var importance
plot(model1_nb_imp)

# predict outcomes on test
model1_nb_pred <- predict(model1_nb_fit, test)
# confusion matrix
model1_cm <- confusionMatrix(model1_nb_pred, test$Liver_Disease)
model1_cm
# model accuracy
model1_acc <- model1_cm$overall["Accuracy"]
model1_acc

# Logistic Regression - GLM
# set seed
set.seed(2)
# fit/train the model
model2_glm_fit <- train(Liver_Disease ~., data = train, method = "glm", family = "binomial")
model2_glm_fit

# var importance
model2_glm_imp <- varImp(model2_glm_fit, scale = FALSE)

# plot var importance
plot(model2_glm_imp)

# predict outcomes on test
model2_glm_pred <- predict(model2_glm_fit, test)
# confusion matrix
model2_cm <- confusionMatrix(model2_glm_pred, test$Liver_Disease)
model2_cm
# model accuracy
model2_acc <- model2_cm$overall["Accuracy"]
model2_acc

# knn
# set seed
set.seed(2)
# fit/train the model
model3_knn_fit <- train(Liver_Disease ~ ., data = train, method = "knn")
model3_knn_fit

# var importance
model3_knn_imp <- varImp(model3_knn_fit, scale = FALSE)
model3_knn_imp

# plot var importance
plot(model3_knn_imp)

# predict outcomes on test
model3_knn_pred <- predict(model3_knn_fit, test)
# confusion matrix
model3_cm <- confusionMatrix(model3_knn_pred, test$Liver_Disease)
model3_cm
# model accuracy
model3_acc <- model3_cm$overall["Accuracy"]
model3_acc

# Random Forest
# set seed
set.seed(3)
# fit/train the model
model4_rf = train(Liver_Disease ~ ., method = "rf", data = train, prox = TRUE)
model4_rf

# plot trained models mtrys
plot(model4_rf)

# var importance
model4_rf_imp <- varImp(model4_rf, scale = FALSE)
model4_rf_imp

# plot var importance
plot(model4_rf_imp)

# predict outcomes on test
model4_pred <- predict(model4_rf, test)
# confusion matrix
model4_cm <- confusionMatrix(model4_pred, test$Liver_Disease)
model4_cm
# model accuracy
model4_acc <- model4_cm$overall["Accuracy"]
model4_acc

# Overview accuracy
# create table with accuracies overview
tibble(method = c("naive bayes", "glm", "knn", "random forest"),
       Accuracy = c(model1_acc, model2_acc, model3_acc, model4_acc))

# Ensembler
# list of models
models <- c("glm", "lda",  "naive_bayes",  "svmLinear", "gamLoess",
            "qda", "knn", "kknn", "loclda", "gam", "rf", "ranger", 
            "wsrf", "Rborist", "avNNet", "mlp", "monmlp", "adaboost",
            "gbm", "svmRadial", "svmRadialCost", "svmRadialSigma")

# set seed
set.seed(1)
# fit/train models
fits <- lapply(models, function(model){ 
  print(model)
  train(Liver_Disease ~ ., method = model, data = train)
}) 

# predict with each model
fits_predicts <- sapply(fits, function(fits){
  predict(fits, test) 
})

# calculate accuracies for each model
acc <- colMeans(fits_predicts == test$Liver_Disease)
acc

# obtain mean score of predictions
votes <- rowMeans(fits_predicts == 1)
# accumulated voting, below mean of 0.5 vote is 0
y_hat <- ifelse(votes > 0.5, 1, 0)
# ensembler accuracy mean is mean of accumulated votes accuracy
ensembler_mean <- mean(y_hat == test$Liver_Disease)
ensembler_mean

# which individual models are better than the ensembler mean
better <- ensembler_mean < acc
acc[which(better==T)]