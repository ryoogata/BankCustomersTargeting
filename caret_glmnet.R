require(caret)
require(caretEnsemble)
require(pROC)
require(doParallel)

require(glmnet)

source("script/R/fun/summaryResult.R")
result.glmnet.df <- readRDS("result/result.glmnet.df.data")

#
# 前処理
#
source("./Data-pre-processing.R")

my_preProcess <- c("center", "scale")

data_preProcess <- "none"
data_preProcess <- "nzv"
data_preProcess <- "dummy"
data_preProcess <- "dummy.nzv.highlyCorDescr"

if ( data_preProcess == "none") {
  TRAIN <- all.train
  TRAIN.TRAIN <- train.train
  TRAIN.TEST <- train.test
  TEST <- test
} else if ( data_preProcess == "nzv") {
  TRAIN <- all.nzv.train
  TRAIN.TRAIN <- train.nzv.train
  TRAIN.TEST <- train.nzv.test
  TEST <- test
} else if ( data_preProcess == "dummy") {
  TRAIN <- train.dummy
  TRAIN.TRAIN <- train.dummy.train
  TRAIN.TEST <- train.dummy.test
  TEST <- test.dummy
} else if ( data_preProcess == "dummy.nzv.highlyCorDescr") {
  TRAIN <- train.dummy.nzv.highlyCorDescr
  TRAIN.TRAIN <- train.dummy.nzv.highlyCorDescr.train
  TRAIN.TEST <- train.dummy.nzv.highlyCorDescr.test
  TEST <- test.dummy.nzv.highlyCorDescr
}


#
# glmnet
#

# seeds の決定
set.seed(123)
seeds <- vector(mode = "list", length = 51)
for(i in 1:50) seeds[[i]] <- sample.int(1000, 500)
seeds[[51]] <- sample.int(1000, 1)

my_control <- trainControl(
  method = "cv"
  ,number = 10
  ,summaryFunction = twoClassSummary
  ,classProbs = TRUE
  ,verbose = TRUE
  ,savePredictions = "final"
  ,index = createResample(TRAIN.TRAIN$response, 10)
  ,seeds = seeds
)

doParallel <- trainControl(
  method = "cv"
  ,number = 10
  ,summaryFunction = twoClassSummary
  ,classProbs = TRUE
  ,allowParallel=TRUE
  ,verboseIter=TRUE
  ,savePredictions = "final"
  ,index = createResample(TRAIN.TRAIN$response, 10)
  ,seeds = seeds
)


# 説明変数一覧の作成
explanation_variable <- names(subset(TRAIN, select = -c(response)))

# fit.glmnet <-
#   train(x = TRAIN.TRAIN[,explanation_variable],
#         y = TRAIN.TRAIN$response,
#         method = "glmnet",
#         tuneGrid = expand.grid(alpha = 0 , lambda = 0),
#         trControl = trainControl(method = "cv", number = 10, verbose = TRUE))

cl <- makeCluster(detectCores(), type = 'PSOCK', outfile = " ")
registerDoParallel(cl)

model_list <- caretList(
  x = TRAIN.TRAIN[,explanation_variable]
  ,y = TRAIN.TRAIN$response
  #,trControl = my_control
  ,trControl = doParallel
  #,preProcess = my_preProcess
  ,tuneList = list(
    glmnet = caretModelSpec(
      method = "glmnet"
      ,metric = "ROC"
      ,family = "binomial"
      ,tuneGrid = expand.grid(
                              alpha = c(1:10/100)
                              ,lambda = c(1:10/100)
                             )
    )
  )
)

stopCluster(cl)
registerDoSEQ()

fit.glmnet <- model_list[[1]]

fit.glmnet$times
# $everything
# ユーザ   システム       経過  
# 30.309      0.356     30.758 
# 
# $final
# ユーザ   システム       経過  
# 2.983      0.022      3.007 
# 
# $prediction
# [1] NA NA NA

#
# モデル比較
#
allProb <- caret::extractProb(
                              list(fit.glmnet)
                              ,testX = subset(TRAIN.TEST, select = -c(response))
                              ,testY = unlist(subset(TRAIN.TEST, select = c(response)))
                             )

# dataType 列に Test と入っているもののみを抜き出す
testProb <- subset(allProb, dataType == "Test")

tp <- subset(testProb, object == "Object1")
confusionMatrix(tp$pred, tp$obs)$overall[1]

# ROC
pROC::roc(tp$obs, tp$yes)

# 結果の保存
result.glmnet.df <- rbind(result.glmnet.df, summaryResult(model_list[[1]]))
saveRDS(result.glmnet.df, "result/result.glmnet.df.data")


# predict() を利用した検算 
if (is.null(fit.glmnet$preProcess)){
  # preProcess を指定していない場合
  pred_test.verification <- predict(
    fit.glmnet$finalModel
    ,newx = Matrix(as.matrix(subset(TRAIN.TEST, select = -c(response))), sparse = TRUE )
    ,s = fit.glmnet$bestTune$lambda
    ,type = "response"
  )
} else {
  # preProcess を指定している場合
  pred_test.verification <- preProcess(
    subset(TRAIN.TEST, select = -c(response))
    ,method = my_preProcess
  ) %>%
    predict(., subset(TRAIN.TEST, select = -c(response))) %>%
    as.matrix(.) %>%
    Matrix(., sparse = TRUE ) %>%
    predict(
      fit.glmnet$finalModel
      ,newx = .
      ,s = fit.glmnet$bestTune$lambda
      ,type = "response"
    )
}

#ROC
pROC::roc(TRAIN.TEST[,"response"], as.vector(pred_test.verification))
