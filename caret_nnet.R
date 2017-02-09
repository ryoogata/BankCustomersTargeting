require(caret)
require(caretEnsemble)
require(pROC)
require(doParallel)

require(nnet)
require(doParallel)

source("script/R/fun/summaryResult.R")
result.nnet.df <- readRDS("result/result.nnet.df.data")

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
# nnet
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

cl <- makeCluster(detectCores())
registerDoParallel(cl)

# fit.nnet <-
#   train(
#         x = TRAIN.TRAIN[,explanation_variable]
#         ,y = TRAIN.TRAIN$response
#         ,method = "nnet" 
#         ,metric = "ROC" 
#         ,tuneGrid = expand.grid(decay= c(0, 1e-4, 1e-3,1e-2,1e-1), size = (1:5)*1)
#         ,trace = FALSE
#         ,trControl = trainControl(method = "cv",
#                                   number = 10,
#                                   summaryFunction = twoClassSummary,
#                                   classProbs = TRUE,
#                                   verbose = TRUE,
#                                   savePredictions = "final")
#         )

model_list <- caretList(
  x = TRAIN.TRAIN[,explanation_variable]
  ,y = TRAIN.TRAIN$response
  ,trControl = doParallel
  #,preProcess = my_preProcess
  ,tuneList = list(
    fit.ranger = caretModelSpec(
      method = "nnet"
      ,metric = "ROC" 
      ,tuneGrid = expand.grid(
                              decay= c(0, 1e-4, 1e-3,1e-2,1e-1)
                              ,size = c(1:5)
                             )
    )
  )
)

stopCluster(cl)
registerDoSEQ()


fit.nnet$times
# $everything
# ユーザ   システム       経過  
# 1301.228     12.188   1324.913 

fit.nnet$bestTune$size
# [1] 2

fit.nnet$bestTune$decay
# [1] 0.1

#
# モデル比較
#
allProb <- caret::extractProb(
                              list(fit.nnet)
                              ,testX = subset(TRAIN.TEST, select = -c(response))
                              ,testY = unlist(subset(TRAIN.TEST, select = c(response)))
                             )

# dataType 列に Test と入っているもののみを抜き出す
testProb <- subset(allProb, dataType == "Test")
tp <- subset(testProb, object == "Object1")

# confusionMatrix で比較
confusionMatrix(tp$pred, tp$obs)$overall[1]

# ROC
pROC::roc(tp$obs, tp$yes)

# 結果の保存
result.nnetdf <- rbind(result.nnet.df, summaryResult(model_list[[1]]))
saveRDS(result.nnet.df, "result/result.nnet.df.data")

# predict() を利用した検算 
if (is.null(fit.nnet$preProcess)){
  # preProcess を指定していない場合
  pred_test.verification <- predict(
    fit.nnet$finalModel
    ,subset(TRAIN.TEST, select = -c(response))
    ,type = "raw"
  )
} else {
  # preProcess を指定している場合
  pred_test.verification <- preProcess(
    subset(TRAIN.TEST, select = -c(response))
    ,method = my_preProcess
  ) %>%
    predict(., subset(TRAIN.TEST, select = -c(response))) %>%
    predict(fit.nnet$finalModel, . ,type = "raw")
}

#ROC
pROC::roc(TRAIN.TEST[,"response"], pred_test.verification[,"yes"])


#
# 予測データにモデルの当てはめ
#
#pred_test <- predict(fit.nnet$finalModel, test, type="raw")

if (is.null(fit.nnet$preProcess)){
  # preProcess を指定していない場合
  pred_test <- predict(fit.nnet$finalModel, TEST, type = "raw")
  
  PREPROCESS <- "no_preProcess"
} else {
  # preProcess を指定している場合
  pred_test <- preProcess(TEST, method = my_preProcess) %>%
    predict(., TEST) %>%
    predict(fit.nnet$finalModel, ., type = "raw")
  
  PREPROCESS <- paste(my_preProcess, collapse = "_")
}


#submitの形式で出力(CSV)
#データ加工
out <- data.frame(test$id, pred_test)

# 予測データを保存
for(NUM in 1:10){
  DATE <- format(jrvFinance::edate(from = Sys.Date(), 0), "%Y%m%d")
  SUBMIT_FILENAME <- paste("./submit/submit_", DATE, "_", NUM, "_", PREPROCESS, "_nnet.csv", sep = "")
  
  if ( !file.exists(SUBMIT_FILENAME) ) {
    write.table(out, #出力データ
                SUBMIT_FILENAME, #出力先
                quote = FALSE, #文字列を「"」で囲む有無
                col.names = FALSE, #変数名(列名)の有無
                row.names = FALSE, #行番号の有無
                sep = "," #区切り文字の指定
    )
    break
  }
}

saveRDS(fit.nnet, file = "fit.nnet.rdata")

fit.nnet <- readRDS("fit.nnet.rdata")
