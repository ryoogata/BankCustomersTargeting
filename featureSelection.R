require(caret)
require(doParallel)


#
# 前処理
#
source("./Data-pre-processing.R")

# rfe() のオプション指定
rfectrl <- rfeControl(
  functions = rfFuncs
  ,method = "cv"
  ,verbose = TRUE
  ,returnResamp = "final"
  ,allowParallel=TRUE
)

# トレーニングのカスタマイズオプション
trctrl <- trainControl(
  method = "cv"
  ,number = 10
)

cl <- makeCluster(detectCores(), type = 'PSOCK', outfile = " ")
registerDoParallel(cl)

fit.rpart.rfe <- rfe(
  x = subset(train.dummy, select = -c(response))
  ,y = train.dummy$response
  #,method = "rpart"
  #,metric = "Precision"
  ,sizes = c(30, 50)
  ,rfeControl = rfectrl
  ,trControl = trctrl
)

stopCluster(cl)
registerDoSEQ()

fit.rpart.rfe
fit.rpart.rfe$optVariables

cl <- makeCluster(detectCores(), type = 'PSOCK', outfile = " ")
registerDoParallel(cl)

fit.glm.rfe <- rfe(
  x = subset(train.dummy, select = -c(response))
  ,y = t$response
  ,method = "glm"
  #,metric = "Precision"
  ,sizes = c(30, 40)
  ,rfeControl = rfectrl
  ,trControl = trctrl
)

stopCluster(cl)
registerDoSEQ()

fit.glm.rfe
fit.glm.rfe$optVariables