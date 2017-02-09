require(caret)
require(caretEnsemble)
require(pROC)
require(doParallel)

require(kernlab)


source("./summaryResult.R")
result.svmRadial.df <- readRDS("result/result.svmRadial.df.data")

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
# svmRadial
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

cl <- makeCluster(detectCores(), type = 'PSOCK', outfile = " ")
registerDoParallel(cl)

model_list <- caretList(
  x = TRAIN.TRAIN[,explanation_variable]
  ,y = TRAIN.TRAIN$response
  #,trControl = my_control
  ,trControl = doParallel
  ,preProcess = my_preProcess
  ,tuneList = list(
    svmRadial = caretModelSpec(
      method = "svmRadial"
      ,metric = "ROC"
      ,tuneGrid = expand.grid(
        C = seq(0.5)
        ,sigma = c(0.01)
      )
    )
  )
)

stopCluster(cl)
registerDoSEQ()

fit.svmRadial <- model_list[[1]]

fit.svmRadial$times
# $everything
# ユーザ   システム       経過  
# 10.936      0.448     11.450 

fit.svmRadial
fit.svmRadial$finalModel
#rattle::fancyRpartPlot(fit.svmRadial$finalModel)
#fit.svmRadial$finalModel$variable.importance
varImp(fit.svmRadial, scale = FALSE, useModel = FALSE)
varImp(fit.svmRadial, scale = FALSE)
plot(varImp(fit.svmRadial, scale = FALSE))

ggplot(fit.svmRadial) 


#
# テストデータにモデルを当てはめる ( Prob )
#
allProb <- caret::extractProb(
                              list(fit.svmRadial)
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
result.svmRadial.df <- rbind(result.svmRadial.df, summaryResult(model_list[[1]]))
saveRDS(result.svmRadial.df, "result/result.svmRadial.df.data")

# predict() を利用した検算 
if (is.null(fit.svmRadial$preProcess)){
  # preProcess を指定していない場合
  pred_test.verification <- predict(
    fit.svmRadial$finalModel
    ,subset(TRAIN.TEST, select = -c(response))
    ,type = "prob"
  )
} else {
  # preProcess を指定している場合
  pred_test.verification <- preProcess(
    subset(TRAIN.TEST, select = -c(response))
    ,method = my_preProcess
  ) %>%
    predict(., subset(TRAIN.TEST, select = -c(response))) %>%
    predict(fit.svmRadial$finalModel, ., type = "prob")
}

#ROC
pROC::roc(TRAIN.TEST[,"response"], pred_test.verification[,"yes"])


#
# 予測データにモデルの当てはめ
#
if (is.null(fit.svmRadial$preProcess)){
  # preProcess を指定していない場合
  pred_test <- predict(fit.svmRadial$finalModel, TEST, type="prob")[,"yes"]
  
  PREPROCESS <- "no_preProcess"
} else {
  # preProcess を指定している場合
  pred_test <- preProcess(TEST, method = my_preProcess) %>%
    predict(., TEST) %>%
    predict(fit.svmRadial$finalModel, .)
  
  pred_test <- pred_test[,"yes"]
  
  PREPROCESS <- paste(my_preProcess, collapse = "_")
}


#submitの形式で出力(CSV)
#データ加工
out <- data.frame(TEST$id, pred_test)

# 予測データを保存
for(NUM in 1:10){
  DATE <- format(jrvFinance::edate(from = Sys.Date(), 0), "%Y%m%d")
  SUBMIT_FILENAME <- paste("./submit/submit_", DATE, "_", NUM, "_", PREPROCESS, "_svmRadial.csv", sep = "")
  
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
