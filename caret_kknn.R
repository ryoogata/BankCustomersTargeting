require(caret)
require(caretEnsemble)
require(pROC)
require(doParallel)

require(kknn)

source("script/R/fun/summaryResult.R")
result.kknn.df <- readRDS("result/result.kknn.df.data")

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
# kknn
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

# fit.kknn.dummy <-
#   train(
#         x = TRAIN.TRAIN[,explanation_variable]
#         ,y = TRAIN.TRAIN$response
#         ,trControl = my_control
#         ,method = "kknn"
#         ,metric = "ROC" 
#         ,tuneGrid = expand.grid(mtry = 5)
#         ,importance = 'impurity'
#         )

cl <- makeCluster(detectCores(), type = 'PSOCK', outfile = " ")
registerDoParallel(cl)

model_list <- caretList(
  x = TRAIN.TRAIN[,explanation_variable]
  ,y = TRAIN.TRAIN$response
  ,trControl = my_control
  ,tuneList = list(
    fit.kknn.dummy = caretModelSpec(
        method = "kknn"
        ,metric = "ROC" 
        ,tuneGrid = expand.grid(
                                kmax = 1
                                ,distance = 1
                                ,kernel = "rectangular"
                               )
    )
  )
)

stopCluster(cl)
registerDoSEQ()

fit.kknn <- model_list[[1]]

fit.kknn$times
# tuneGrid = expand.grid(kmax = 1,distance = 1,kernel = "rectangular")
# $everything
# ユーザ   システム       経過  
# 205.239      6.891   1708.582 

# Fitting kmax = 5, distance = 5, kernel = rectangular on full training set
# $everything
# user    system   elapsed 
# 252.280     7.936 13221.714 


fit.kknn$bestTune$kmax
fit.kknn$bestTune$distance
fit.kknn$bestTune$kernel

# 特徴量の確認
caret::varImp(fit.kknn)

#
# モデル比較
#
allProb <- caret::extractProb(
                              list(fit.kknn)
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
result.kknn.df <- rbind(result.kknn.df, summaryResult(model_list[[1]]))
saveRDS(result.kknn.df, "result/result.kknn.df.data")

# predict() を利用した検算 
if (is.null(fit.nnet$preProcess)){
  # preProcess を指定していない場合
  pred_test.verification <- predict(
    fit.kknn$finalModel
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
    predict(fit.kknn$finalModel, . ,type = "prob")
}

#ROC
pROC::roc(TRAIN.TEST[,"response"], pred_test.verification[,"yes"])

#
# 予測データにモデルの当てはめ
#
if (is.null(fit.kknn$preProcess)){
  # preProcess を指定していない場合
  pred_test <- predict(fit.kknn$finalModel, TEST, type = "prob")[,2]
  
  PREPROCESS <- "no_preProcess"
} else {
  # preProcess を指定している場合
  pred_test <- preProcess(TEST, method = my_preProcess) %>%
    predict(., TEST) %>%
    predict(fit.kknn$finalModel, ., type = "prob")
  
  pred_test <- pred_test[,2]
  
  PREPROCESS <- paste(my_preProcess, collapse = "_")
}


#submitの形式で出力(CSV)
#データ加工
out <- data.frame(test$id, pred_test)

# 予測データを保存
for(NUM in 1:10){
  DATE <- format(jrvFinance::edate(from = Sys.Date(), 0), "%Y%m%d")
  SUBMIT_FILENAME <- paste("./submit/submit_", DATE, "_", NUM, "_", PREPROCESS, "_kknn.csv", sep = "")
  
  
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
