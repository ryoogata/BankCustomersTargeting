require(caret)
require(caretEnsemble)
require(pROC)
require(doParallel)

require(partykit)
require(rattle)
require(glmnet)

#
# 前処理
#
source("./Data-pre-processing.R")

my_preProcess <- c("center", "scale")

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
  ,index = createResample(train.dummy.train$response, 10)
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
  ,index = createResample(train.dummy.train$response, 10)
  ,seeds = seeds
)


# 説明変数一覧の作成
explanation_variable.glmnet.dummy <- names(subset(train.dummy, select = -c(response)))

# fit.glmnet <-
#   train(x = train.dummy.train[,explanation_variable.glmnet],
#         y = train.dummy.train$response,
#         method = "glmnet",
#         tuneGrid = expand.grid(alpha = 0 , lambda = 0),
#         trControl = trainControl(method = "cv", number = 10, verbose = TRUE))

cl <- makeCluster(detectCores(), type = 'PSOCK', outfile = " ")
registerDoParallel(cl)

model_list_glmnet.dummy <- caretList(
  x = train.dummy.train[,explanation_variable.glmnet.dummy]
  ,y = train.dummy.train$response
  #,trControl = my_control
  ,trControl = doParallel
  #,preProcess = my_preProcess
  ,tuneList = list(
    glmnet = caretModelSpec(
      method = "glmnet"
      ,metric = "ROC"
      ,family = "binomial"
      ,tuneGrid = expand.grid(
                              alpha = c(1:3/10)
                              ,lambda = c(1:3/10)
                             )
    )
  )
)

stopCluster(cl)
registerDoSEQ()

fit.glmnet.dummy <- model_list_glmnet.dummy[[1]]

fit.glmnet.dummy$times
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
allProb <- caret::extractProb(list(fit.glmnet.dummy),
                                    testX = subset(train.dummy.test, select = -c(response)),
                                    testY = unlist(subset(train.dummy.test, select = c(response))))

# dataType 列に Test と入っているもののみを抜き出す
testProb <- subset(allProb, dataType == "Test")

tp_glmnet.dummy <- subset(testProb, object == "Object1")

confusionMatrix(tp_glmnet.dummy$pred, tp_glmnet.dummy$obs)$overall[1]

# ROC
pROC::roc(tp_glmnet.dummy$obs, tp_glmnet.dummy$yes)

# ToDo: predict() が利用できない問題を解決
# predict() を利用した検算 
if (is.null(fit.glmnet.dummy$preProcess)){
  # preProcess を指定していない場合
  pred_test.verification <- predict(
    fit.glmnet.dummy$finalModel
    ,subset(train.dummy.test, select = -c(response))
    #,type = "response"
  )
} else {
  # preProcess を指定している場合
  pred_test.verification <- preProcess(
    subset(train.dummy.test ,select = -c(response))
    ,method = my_preProcess
  ) %>%
    predict(., subset(train.dummy.test, select = -c(response))) %>%
    predict(fit.glmnet.dummy$finalModel, ., type="response")
}
