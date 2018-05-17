library(datasets)

help("datasets")
data("iris")
str(iris)

#############################################Bootstrap sampling###################################################
# define training control - Bootstrap
## takes smples randomly
train_control <- trainControl(method="boot", number=100)
# train the model
model <- train(Species~., data=iris, trControl=train_control, method="nb")
# summarize results
print(model)
model$bestTune
model$resample
model$results
sd(model$resample$Accuracy)*100

###########################################Decission Tree#################################################


# define training control - K fold - decission tree with rpart2
# rpart2 can use maxdepth param and rpart can use cp(cost complexity) cp has to be less usually
train_control <- trainControl(method="cv", number=5)
# fix the parameters of the algorithm
grid <- expand.grid(cp = seq(0.001,0.1,by = 0.001))
# train the model
model <- train(Species~., data=iris, trControl=train_control, method="rpart",tuneGrid = grid)
# summarize results
print(model)
model$results
model$resample
sd(model$resample$Accuracy)*100
mean(model$resample$Accuracy)

###########################################Logistic#################################################

## logistic with glmnet
train_cantrol = trainControl(method = "cv", number = 10)
grid = expand.grid(alpha = 1 , lambda = seq(0.001,0.1,by = 0.001))
model <- train(Species~., data=iris, trControl=train_control, tuneGrid = grid,method="glmnet")
print(model)
model$bestTune
model$resample
sd(model$resample$Accuracy)*100
mean(model$resample$Accuracy)

independent = as.matrix(subset(iris,select = -c(Species)))
m = glmnet(x = independent, y = iris$Species, family="multinomial",alpha = 1,lambda = 0.002)
summary(m)
tm = predict(m,newx = independent,type = "class")
confusionMatrix(tm,iris$Species)

#########################################Logistic#####################################
### glmnet with both category and numeric data in data set

######################## alpha = 0 means ridge 1 means lasso
library(glmnet)

age     <- c(4, 8, 7, 12, 6, 9, 10, 14, 7) 
gender  <- as.factor(c(1, 0, 1, 1, 1, 0, 1, 0, 0))
bmi_p   <- c(0.86, 0.45, 0.99, 0.84, 0.85, 0.67, 0.91, 0.29, 0.88) 
m_edu   <- as.factor(c(0, 1, 1, 2, 2, 3, 2, 0, 1))
p_edu   <- as.factor(c(0, 2, 2, 2, 2, 3, 2, 0, 0))
f_color <- as.factor(c("blue", "blue", "yellow", "red", "red", "yellow", 
                       "yellow", "red", "yellow"))
asthma <- c(1, 1, 0, 1, 0, 0, 0, 1, 1)

xfactors <- model.matrix(asthma ~ gender + m_edu + p_edu + f_color)[, -1]
x        <- as.matrix(data.frame(age, bmi_p, xfactors))

# Note alpha=1 for lasso only and can blend with ridge penalty down to
# alpha=0 ridge only.
glmmod <- glmnet(x, y=as.factor(asthma), alpha=1, family="binomial")

# Plot variable coefficients vs. shrinkage parameter lambda.
plot(glmmod, xvar="lambda")

######################################Random Forest######################################


# define training control - Repeated K fold cross validation
train_control <- trainControl(method="cv", number=5)
# train the model
#grid = expand.grid(mtry = c(1,2,3))
model <- train(Species~., data=iris, trControl=train_control, method="rf")
# summarize results
print(model)
model$bestTune
model$resample
mean(model$resample$Accuracy)
sd(model$resample$Accuracy)*100


######################################Random Forest######################################
## random forest
train_control <- trainControl(method="cv", number=5,search = "grid")
tunegrid <- expand.grid(.mtry=c(1,2,3))
modellist <- list()
for (ntree in c(10, 15, 20, 25)) {
  set.seed(100)
  fit <- train(Species ~ ., data=iris, method="rf", metric = "Accuracy",tuneGrid=tunegrid, trControl=train_control, ntree=ntree)
  key <- toString(ntree)
  modellist[[key]] <- fit
}
results = resamples(modellist)
summary(results)
dotplot(results)


######################################Random Forest######################################
## random forest with custom function
customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
customRF$parameters <- data.frame(parameter = c("mtry", "ntree"), class = rep("numeric", 2), label = c("mtry", "ntree"))
customRF$grid <- function(x, y, len = NULL, search = "grid") {}
customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  randomForest(x, y, mtry = param$mtry, ntree=param$ntree, ...)
}
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes


control <- trainControl(method="cv", number=5,search = "grid")
tunegrid <- expand.grid(.mtry=c(1:3), .ntree=c(100, 150, 200, 250))
set.seed(101)
custom <- train(Species ~ ., data=iris, method=customRF, metric= "Accuracy",tuneGrid=tunegrid, trControl=control)
summary(custom)
plot(custom)


######################################Random Forest#################################################################
## RF with mtry and depth

control <- trainControl(method="cv", number=5,search = "grid")
tunegrid <- expand.grid(.mtry=c(1:3), .maxdepth = c(5,6,7,8,9,10))
set.seed(101)
custom <- train(Species ~ ., data=iris, method="rfRules", metric= "Accuracy",tuneGrid=tunegrid, trControl=control)
summary(custom)
plot(custom)



######################################Leave one of out#################################################################
# leaves one row and builds model on others an dtest with that row
# define training control - leave one out
train_control <- trainControl(method="LOOCV")
# train the model
model <- train(Species~., data=iris, trControl=train_control, method="nb")
# summarize results
print(model)
model$bestTune



# Random Search
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")
set.seed(seed)
mtry <- sqrt(ncol(x))
rf_random <- train(Class~., data=dataset, method="rf", metric=metric, tuneLength=15, trControl=control)
print(rf_random)
plot(rf_random)

# Random Search
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")
set.seed(seed)
rf_random <- train(Class~., data=dataset, method="rf", metric=metric, tuneLength=15, trControl=control)
print(rf_random)
plot(rf_random)

# Grid Search
control <- trainControl(method="cv", number=10)
set.seed(7)
tunegrid <- expand.grid(.mtry=c(1:15), .ntree=c(100, 150, 200, 250))
rf_gridsearch <- train(Class~., data=dataset, method="customRF", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)
rf_gridsearch$resample
plot(rf_gridsearch)


######################################xgboost #################################################################

## xgboost with xglinear method
control = trainControl(method = "cv",number = 10)
## alpha means L1(lasso) lamda is L2(ridge) , eta is learning rate , nrounds = no of trees
tunegrid = expand.grid(nrounds = 1:10 , eta = c(0.01,0.02,0.03),alpha = 1, lambda = 0.1)
model= train(Species ~ ., data = iris,method = "xgbLinear" , trControl = control , tuneGrid = tunegrid)
print(model)
model$results
model$resample

?xgboost


######################################xgboost #################################################################

## xgboost with xgbtree
control = trainControl(method = "cv",number = 5)
# nrounds, max_depth, eta, gamma, colsample_bytree, min_child_weight, subsample
## maxdepth = depth of tree , gamma = cost function for next tree node to split , colsample = how many samples taken in each tree , minchildweight = weight of child to split, subsample = how much sample is required to build to tree
## read this url for more details : https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
##min_child_weight [default=1] minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression mode, this simply corresponds to minimum number of instances needed to be in each node. The larger, the more conservative the algorithm will be.
tunegrid = expand.grid(nrounds = 1:10 , eta = c(0.01,0.02,0.03) , max_depth = 5,gamma = 0.1,colsample_bytree = 0.9,min_child_weight = 0.1,subsample = 0.8)
model= train(Species ~ ., data = iris,method = "xgbTree" , trControl = control , tuneGrid = tunegrid)
model$results
model$resample


######################################GradientDescent #################################################################
## gradient boosting
#ntrees, max_depth, min_rows, learn_rate, col_sample_rate
## min_rows  = min rows required to split further
## col_sample_rate = no of columns to be taken to split in each tree
h2o.init()
control = trainControl(method = "cv",number = 5)
tunegrid = expand.grid(ntrees= 5, max_depth = 5, min_rows = 4, learn_rate = 0.01, col_sample_rate = 0.9)
model= train(Species ~ ., data = iris,method = "gbm_h2o" , trControl = control , tuneGrid = tunegrid)
model$results
model$resample

