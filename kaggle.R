library("Hmisc")  

# Getting data into R
setwd("D:\\Kaggle\\Santander Customer Satisfaction")
set.seed(1234)
train = read.csv("train.csv")
test = read.csv("test.csv")
#numer of 0s = 73012
#number of 1s = 3008
train = replace(train,NULL,NA)

#Method for dimensional reductinoality
#1. Correlation Analysis
#2. High variation removal (ID var removal)
#3. Low Variation(Constant) vars removal

# Correlation Analysis -Start

#myvars <- names(train) %in% c("ind_var27_0","ind_var28_0","ind_var28","ind_var27","ind_var41","ind_var46","num_var27_0","num_var28_0","num_var28","num_var27","num_var41","num_var46_0","num_var46","saldo_var28","saldo_var27","saldo_var41","saldo_var46","imp_amort_var18_hace3","imp_amort_var34_hace3","imp_reemb_var13_hace3","imp_reemb_var33_hace3","imp_trasp_var17_out_hace3","imp_trasp_var33_out_hace3","num_var2_0_ult1	num_var2_ult1","num_reemb_var13_hace3","num_reemb_var33_hace3","num_trasp_var17_out_hace3","num_trasp_var33_out_hace3","saldo_var2_ult1","saldo_medio_var13_medio_hace3","ind_var2_0","ind_var2","ind_var46_0","num_var2_0_ult1","num_var2_ult1") 
#newdata <- train[!myvars]
#M <- cor(train)
#M <- cor(newdata)
#NA_Value_data <- train[myvars]

#Removing ID variable
trainID<- train$ID
train$ID <- NULL
testID <- test$ID       #saving Tes Ids for submission purpose
test$ID <- NULL

#Removing constat Variables - If the variance of the variable is 0. Deleting those from the data frame
# 34 vars removed Varcount - 336

Const_vars <- names(train[, sapply(train, function(v) var(v, na.rm=TRUE)==0.000)])
train <- train[setdiff(names(train),Const_vars)]

# Creating a correlation matrix
Corr_Mat <- cor(train)

#Replacing the upper triagle of the matrix with zero values to remove 
#the duplicate correlatons and for determining the variables with high colinearity
Corr_Mat[upper.tri(Corr_Mat)] <- 0 

#Setting diagonal values in the matrix to 0
diag(Corr_Mat) <- 0

#Removing all columns with value more than 0.90------CountVars = 169, 167 vars removed
Corr_Mat <- Corr_Mat[, apply(Corr_Mat,2,function(x) all(x<=0.90))]

#Removing all columns with value less than -0.90
Corr_Mat <- Corr_Mat[, apply(Corr_Mat,2,function(x) all(x>=-0.80))]

#Post correlation analysis, number of vriables reduced from 371 to 204

non_correlated_vars <- colnames(Corr_Mat)

#selecting only un correlated variables in the training set

train <- train[non_correlated_vars]
non_correlated_vars <- non_correlated_vars[! non_correlated_vars %in% "TARGET"]
test <- test[non_correlated_vars]


#Performing Random forest

library(randomForest)
class(train$TARGET)
train$TARGET <- as.factor(train$TARGET) #Category target variable is read as Integer by default in R
mySampSize <- ceiling(table(train$TARGET))
mySampSize[1] <- mySampSize[1]/18

library(ROSE)
#Getting only complete cases from the data where there are no NA values in a record
##train<-complete.cases(train) 
train <- na.omit(train)
train <- sapply(names(train), function(x) table(train[x], useNA = "ifany"))
#Performing oversampling of the complete cases due to small data size
data.balanced <- ovun.sample(train$TARGET~.,data = train, method="over", p=0.24)
train <- data.balanced$data

n <- names(train)
f <- as.formula(paste(parse(text = text1), paste(n[!n %in% "TARGET"], collapse = " + ")))
text1 = "TARGET"

form <- as.formula(paste('train$TARGET', paste(names(train), 
                                    collapse='+'), sep='~')) 

data.balanced.ou <- ovun.sample(train$TARGET~., data=train,N=nrow(train), p=0.5,seed=1, method="over")$data

#Training randomforest model with stratefied sampling
rf = randomForest(train$TARGET ~ .,data=train,ntree=150, replace = TRUE, do.trace = 10,strata = train$TARGET,sampsize=c(mySampSize[1],mySampSize[2]))
rf = randomForest(train$TARGET ~ .,data=train,ntree=50, replace = TRUE, do.trace = 5,strata = train$TARGET)
rf1 = randomForest(train$TARGET ~ .,data=train,ntree=100, replace = TRUE, do.trace = 5)

varImpPlot(rf)

factorVars <- names(train)[sapply(train, function(x) length(unique(x))<10)]
sapply(factorVars, function(x) train[x]<-as.factor(x))
sapply(factorVars, function(x) class(train[x]))


#Calculating AUC for ROC for Random Forest
install.packages("AUC")

library(AUC)

rocc <- roc(predictor = as.numeric(predict(rf,data = train)),response = train$TARGET, plot = TRUE)
#count(unique.data.frame(rf$predicted)


#write.csv(data.frame(rf$predicted,train$TARGET),"prediction.csv")
library(pROC)
aucval <- auc(predictor = as.ordered(predict(rf,data = train)),response = train$TARGET,plot  = TRUE)
print(aucval)

#Performing logistic regression

mylogit <- glm(train$TARGET ~ ., data = train, family = "binomial")

logit_auc <- roc(predictor = as.ordered(predict(mylogit,type = "response",newdata = train)),response = train$TARGET, plot = TRUE)
print(logit_auc)

library(SDMTools)


confusion.matrix(train$TARGET,predict(mylogit,type = "response",newdata = train))



for (f in names(train)) {
  if (length(unique(train[f])) < 15) {
    print("Converted to factor")
    print(f)
    train[f] <- as.factor(train[f])
    
  }
}

test = read.csv("test.csv")
TARGET <- predict.glm(mylogit,type = "response", newdata = test)
TARGET <- predict(rf,type = "response",newdata = test)
ID <- testID
Output <- data.frame(ID,TARGET)
write.csv(Output,"Output.csv",row.names = FALSE)