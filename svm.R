# SVM
library(e1071)
library(caret)
library(pROC)
################################# Q1 #################################
# identify predictors (BMD) of osteoporotic fracture in whole cohort
# ** whole cohort ----
# * data manipulation ----
impute <- impute.transcan(impute_arg, data=df, imputation=1, list.out=TRUE, pr=FALSE, check=FALSE)
impute <- as.data.frame(do.call(cbind,impute))
impute <- impute[,-c(4:6)]
impute <- impute%>% mutate(fracture=as.factor(ifelse((OSQ010A=="1"|OSQ010B=="1"|OSQ010C=="1"), 1, 0)))
data.svm <- impute
# define function to find columns with less than 5 levels and factor them
checkLevels <- function(col,level=5){
  ifelse(length(unique(col)) <= level,1,0)
}
factor.lst <- apply(data.svm,2,checkLevels)
data.svm[,which(factor.lst==1)] <- lapply(data.svm[,which(factor.lst==1)],factor)
# define dataset for question 1, which consider all types of fracture
data1 <- data.svm[,!names(data.svm) %in% c("OSQ010A","OSQ010B","OSQ010C")]
data1.train <- data1[train_ind,] # set train dataset for question 1
data1.test <- data1[-train_ind,]

# * find the best kernel using 5 fold cv ----
set.seed(5250)
type <- c("linear","radial","polynomial")
cv.full <- c()
for (i in 1:3){
  mod <- tune.svm(fracture~.,kernel=type[i], #cost=c(0.1,0.5,1,5,10,50),
                  type="C-classification", tunecontrol=tune.control(cross=5),
                  data=data1, probability=FALSE)
  cv.full[i] <- mod$best.performance
}

# * assess the test error using tuned kernel type on predefined test set ----
svmfit = svm(fracture~.,kernel=type[which.min(cv.full)], 
             data=data1.train,cost=1)
preds.full <- predict(svmfit, data1.test)
auc(data1.test$fracture, as.numeric(preds.full))

# * assess the feature importance ----
# method1 - failed
# library(rminer)
# fit1 <- fit(fracture~., data=data1, model="svm", C=1)
# svm.imp <- Importance(fit1, data=data1)
# method2
# A simple backwards selection, recursive feature elimination (RFE) algorithm
set.seed(5250)
svmprf.type <- c("svmLinear", "svmRadial","svmPolynomial")
svmProfile <- rfe(select(data1,-fracture), data1$fracture,
                  sizes = c(2, 5, 10, 20),
                  rfeControl = rfeControl(functions = rfFuncs,
                                          number = 20),
                  method = svmprf.type[which.min(cv.full)])
svmProfile$optVariables

## ** identify predictors (BMD) of osteoporotic fracture in men ----
data1.men <- split(data1,data1$RIAGENDR,drop = TRUE)$`1`[,-1]
set.seed(5250)
cv.men <- c()
for (i in 1:3){
  mod <- tune.svm(fracture~.,kernel=type[i], #cost=c(0.1,0.5,1,5,10,50),
                  type="C-classification", tunecontrol=tune.control(cross=5),
                  data=data1.men, probability=FALSE)
  cv.men[i] <- mod$best.performance
}
svmProfile.men <- rfe(data1.men[,-25],data1.men$fracture,
                      sizes = c(2, 5, 10, 20),
                      rfeControl = rfeControl(functions = rfFuncs,
                                              number = 20),
                      method = svmprf.type[which.min(cv.men)])
svmProfile.men$optVariables

## ** identify predictors (BMD) of osteoporotic fracture in women ----
data1.women <- split(data1,data1$RIAGENDR,drop = TRUE)$`2`[,-1]
set.seed(5250)
cv.women <- c()
for (i in 1:3){
  mod <- tune.svm(fracture~.,kernel=type[i], #cost=c(0.1,0.5,1,5,10,50),
                  type="C-classification", tunecontrol=tune.control(cross=5),
                  data=data1.women, probability=FALSE)
  cv.women[i] <- mod$best.performance
}
svmProfile.women <- rfe(data1.women[,-25],data1.women$fracture,
                        sizes = c(2, 5, 10, 20),
                        rfeControl = rfeControl(functions = rfFuncs,
                                                number = 20),
                        method = svmprf.type[which.min(cv.women)])
svmProfile.women$optVariables

#################################### Q2 #######################################
# which BMD measure is the best predictor of hip fracture
data2 <- data.svm[,!names(data.svm) %in% c("fracture","OSQ010B","OSQ010C")]
set.seed(5250)
cv.hip <- c()
for (i in 1:3){
  mod <- tune.svm(OSQ010A~.,kernel=type[i], #cost=c(0.1,0.5,1,5,10,50),
                  type="C-classification", tunecontrol=tune.control(cross=5),
                  data=data2, probability=FALSE)
  cv.hip[i] <- mod$best.performance
}
svmProfile.hip <- rfe(data2[,-which(names(data2)=="OSQ010A")],data2$OSQ010A,
                      sizes = c(2, 5, 10, 20),
                      rfeControl = rfeControl(functions = rfFuncs,
                                              number = 20),
                      method = svmprf.type[which.min(cv.hip)])
svmProfile.hip$optVariables
