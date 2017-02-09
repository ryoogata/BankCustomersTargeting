require(caret)
require(caretEnsemble)
require(doParallel)

require(ranger)

source("./summaryResult.R")
result.ranger.df <- readRDS("result/result.ranger.df.data")

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
# ranger
#

# 説明変数一覧の作成
explanation_variable <- names(subset(TRAIN, select = -c(response)))

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
  ,index = createResample(train.train$response, 10)
  ,seeds = seeds
)

# fit.ranger <-
#   train(
#         x = TRAIN.TRAIN[,explanation_variable]
#         ,y = TRAIN.TRAIN$response
#         ,trControl = my_control
#         ,method = "ranger"
#         ,metric = "ROC" 
#         ,tuneGrid = expand.grid(mtry = 5)
#         ,importance = 'impurity'
#         )

cl <- makeCluster(detectCores(), type = 'PSOCK', outfile = " ")
registerDoParallel(cl)

model_list <- caretList(
  x = TRAIN.TRAIN[,explanation_variable]
  ,y = TRAIN.TRAIN$response
  ,trControl = doParallel
  #,preProcess = my_preProcess
  ,tuneList = list(
    fit.ranger = caretModelSpec(
        method = "ranger"
        ,metric = "ROC" 
        ,tuneGrid = expand.grid(mtry = c(5))
        ,importance = 'impurity'
    )
  )
)

stopCluster(cl)
registerDoSEQ()

fit.ranger <- model_list[[1]]

fit.ranger$times
# $everything
# ユーザ   システム       経過  
# 873.334     13.257    267.033 

fit.ranger$bestTune$mtry
# [1] 5

trellis.par.set(caretTheme())
ggplot(fit.ranger, plotType = "scatter")

plot(fit.ranger$finalModel)

# 特徴量の確認
varImp(fit.ranger, scale = FALSE)
plot(varImp(fit.ranger, scale = FALSE))

#
# モデル比較
#
allProb <- caret::extractProb(
                              list(fit.ranger)
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
result.ranger.df <- rbind(result.ranger.df, summaryResult(model_list[[1]]))
saveRDS(result.ranger.df, "result/result.ranger.df.data")

# predict() を利用した検算 
if (is.null(fit.ranger$preProcess)){
  # preProcess を指定していない場合
  pred_test.verification <- predict(
    fit.ranger$finalModel
    ,subset(TRAIN.TEST, select = -c(response))
  )
} else {
  # preProcess を指定している場合
  pred_test.verification <- preProcess(
    subset(TRAIN.TEST ,select = -c(response))
    ,method = my_preProcess
  ) %>%
    predict(., subset(TRAIN.TEST, select = -c(response))) %>%
    predict(fit.ranger$finalModel, .)
}

#ROC
pROC::roc(TRAIN.TEST[,"response"], pred_test.verification$predictions[,"yes"])

#
# 予測データにモデルの当てはめ
#
# preProcValues <- preProcess(TEST, method = c("center", "scale"))
# test.transformed <- predict(preProcValues, TEST)
# 
# test.transformed <- TEST
# 
# pred_test <- predict(fit.ranger$finalModel, test.transformed)$prediction[,2]

#
# 予測データにモデルの当てはめ
#
if (is.null(fit.ranger$preProcess)){
  # preProcess を指定していない場合
  pred_test <- predict(fit.ranger$finalModel, TEST, type="response")$prediction[,2]
  
  PREPROCESS <- "no_preProcess"
} else {
  # preProcess を指定している場合
  pred_test <- preProcess(TEST, method = my_preProcess) %>%
    predict(., TEST) %>%
    predict(fit.ranger$finalModel, .)
  
  pred_test <- pred_test[,2]
  
  PREPROCESS <- paste(my_preProcess, collapse = "_")
}

#submitの形式で出力(CSV)
#データ加工
out <- data.frame(TEST$id, pred_test)

# 予測データを保存
for(NUM in 1:10){
  DATE <- format(jrvFinance::edate(from = Sys.Date(), 0), "%Y%m%d")
  SUBMIT_FILENAME <- paste("./submit/submit_", DATE, "_", NUM, "_ranger.csv", sep = "")
  
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
