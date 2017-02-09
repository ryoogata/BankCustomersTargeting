require(caret)
require(caretEnsemble)
require(pROC)
require(doParallel)

require(partykit)
#require(rattle)
require(mlbench)

source("script/R/fun/summaryResult.R")
result.glm.df <- readRDS("result/result.glm.df.data")

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
# glm
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

# fit.glm <-
#   train(
#         x = train.train[,explanation_variable]
#         ,y = train.train$response
#         ,trControl = my_control
#         ,method = "glm"
#         ,family = binomial(link="logit")
#        )

cl <- makeCluster(detectCores(), type = 'PSOCK', outfile = " ")
registerDoParallel(cl)

model_list <- caretList(
  x = TRAIN.TRAIN[,explanation_variable]
  ,y = TRAIN.TRAIN$response
  #,trControl = my_control
  ,trControl = doParallel
  ,preProcess = my_preProcess
  ,tuneList = list(
    glm = caretModelSpec(
      method = "glm"
      ,metric = "ROC"
      ,family = binomial(link="logit")
      )
    )
)

stopCluster(cl)
registerDoSEQ()

fit.glm <- model_list[[1]]

fit.glm$times
# $everything
# ユーザ   システム       経過  
# 17.208      1.650     18.716 

fit.glm
fit.glm$finalModel
summary(fit.glm$finalModel)
fit.glm$preProcess

varImp(fit.glm, scale = FALSE)

#
# テストデータにモデルを当てはめる ( Prob )
#
allProb <- caret::extractProb(
                              list(fit.glm)
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
result.glm.df <- rbind(result.glm.df, summaryResult(model_list[[1]]))
saveRDS(result.glm.df, "result/result.glm.df.data")

# predict() を利用した検算 
if (is.null(fit.glm$preProcess)){
  # preProcess を指定していない場合
  pred_test.verification <- predict(
                                    fit.glm$finalModel
                                    ,subset(TRAIN.TEST, select = -c(response))
                                    ,type = "response"
                                   )
} else {
  # preProcess を指定している場合
  pred_test.verification <- preProcess(
                                        subset(TRAIN.TEST ,select = -c(response))
                                        ,method = my_preProcess
                                      ) %>%
    predict(., subset(TRAIN.TEST, select = -c(response))) %>%
    predict(fit.glm$finalModel, ., type="response")
}

# ROC
pROC::roc(TRAIN.TEST[,"response"], pred_test.verification)


#
# 予測データにモデルの当てはめ
#
if (is.null(fit.glm$preProcess)){
  # preProcess を指定していない場合
  pred_test <- predict(fit.glm$finalModel, TEST, type="response")
  
  PREPROCESS <- "no_preProcess"
} else {
  # preProcess を指定している場合
  pred_test <- preProcess(TEST, method = my_preProcess) %>%
    predict(., TEST) %>%
    predict(fit.glm$finalModel, ., type="response")
  
  PREPROCESS <- paste(my_preProcess, collapse = "_")
}

#submitの形式で出力(CSV)
#データ加工
out <- data.frame(test$id, pred_test)

# 予測データを保存
for(NUM in 1:10){
  DATE <- format(jrvFinance::edate(from = Sys.Date(), 0), "%Y%m%d")
  SUBMIT_FILENAME <- paste("./submit/submit_", DATE, "_", NUM, "_", PREPROCESS, "_glm.csv", sep = "")
  
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
