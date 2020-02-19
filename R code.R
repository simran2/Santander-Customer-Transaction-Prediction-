library(caTools)
require(caTools)
library(pROC)
library(caret)
library(e1071)
library(C50)
library(randomForest)

rm(list=ls())

#check current working directory
getwd()

#read data
df = read.csv('train.csv')
test = read.csv('test.csv')

#check data
str(df)
summary(df)
head(df)

#ID_code identifies each row uniquely. It is not helping in predicting the target variables thus it can be dropped.
df = df[,!names(df) %in% "ID_code"]

#Check for missing values
df[is.na(df)]

std_dev = data.frame(initial = apply(df,2,sd))

#Detect and impute outliers
for(i in 2:ncol(df)){
  val = df[df[[i]]%in%boxplot.stats(df[[i]])$out,i]
  df[which(df[[i]]%in%val),i] = NaN
  df[is.na(df[[i]]), i] = mean(df[[i]], na.rm = TRUE)
  
  std_dev$final = apply(df,2,sd)
}
std_dev$diff = std_dev$initial - std_dev$final
std_dev$diff

#Check correlation
heatmap(cor(df))

#Sampling
sample = sample.split(df, SplitRatio = 0.8)
train1 = subset(df, sample==TRUE)
test1 = subset(df, sample==FALSE)

#Scaling
train1_scaled = scale(train1)
test1_scaled = scale(test1)


#1) Logistic Regression
glm_model = glm(target~., data = train1, family = "binomial")
summary(glm_model)

pred = predict(glm_model, test1[,2:201], type = "response")
pred = ifelse(pred>0.5,1,0)
confusionMatrix(table(as.vector(pred), unname(unlist(test1[,1]))))
auc(pred, test1[,1])

#Feature selection using Backward elimination
pvalues = data.frame(summary(glm_model)$coefficients[,4])
selected_columns = c()
for(i in 0:199){
  idx = paste('var_',i,sep='')
  if (pvalues[idx,]<0.05){
  selected_columns = c(selected_columns, idx)
  }
}
selected_columns = c('target',selected_columns)

df_new = df[selected_columns]

sample = sample.split(df_new, SplitRatio = 0.8)
train1 = subset(df_new, sample==TRUE)
test1 = subset(df_new, sample==FALSE)

glm_model = glm(target~., data = train1, family = "binomial")
pred = predict(glm_model, test1[,2:176], type = "response")
pred = ifelse(pred>0.5,1,0)
confusionMatrix(table(as.vector(pred), unname(unlist(test1[,1]))))
auc(pred, test1[,1])

#2) Naive Bayes classifier
train1$target = as.factor(train1$target)
NB = naiveBayes(target~., train1)
NB_predict = predict(NB, test1[2:176],type="class")
confusionMatrix(table(as.vector(NB_predict), unname(unlist(test1[,1]))))
auc(NB_predict, test1[,1])

#3) Decision tree classifier
train1$target = as.factor(train1$target)
c50_model = C5.0(target~., train1, trials = 10)
c50_predict = predict(c50_model, test1[,2:176],type="class")
confusionMatrix(table(as.vector(c50_predict), unname(unlist(test1[,1]))))
str(c50_predict)

#4) Random Forest classifier
RF_model =randomForest(target~., data=data.frame(train1), ntree=500)
RF_pred = predict(RF_model, data.frame(test1[,2:176]))
confusionMatrix(table(as.vector(RF_pred), unname(unlist(test1[,1]))))

