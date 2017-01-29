require(caret)
require(caretEnsemble)
require(pROC)
require(doParallel)

require(partykit)
#require(rattle)
require(mlbench)

#
# 前処理
#
source("./Data-pre-processing.R")

my_preProcess <- c("center", "scale")

# train.dummy <- train.dummy.nzv.highlyCorDescr
# train.dummy.train <- train.dummy.nzv.highlyCorDescr.train
# train.dummy.test <- train.dummy.nzv.highlyCorDescr.test
# test.dummy <- test.dummy.nzv.highlyCorDescr

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
explanation_variable.glm.dummy <- names(subset(train.dummy, select = -c(response)))

# fit.glm.dummy <-
#   train(
#         x = train.train[,explanation_variable.glm ]
#         ,y = train.train$response
#         ,trControl = my_control
#         ,method = "glm"
#         ,family = binomial(link="logit")
#        )

# cl <- makeCluster(detectCores(), type = 'PSOCK', outfile = " ")
# registerDoParallel(cl)

model_list_glm.dummy <- caretList(
  x = train.dummy.train[,explanation_variable.glm.dummy]
  ,y = train.dummy.train$response
  ,trControl = my_control
  #,trControl = doParallel
  #,preProcess = my_preProcess
  ,tuneList = list(
    glm = caretModelSpec(
      method = "glm"
      ,metric = "ROC"
      ,family = binomial(link="logit")
      )
    )
)

# stopCluster(cl)
# registerDoSEQ()

fit.glm.dummy <- model_list_glm.dummy[[1]]

fit.glm.dummy$times
# $everything
# ユーザ   システム       経過  
# 17.208      1.650     18.716 

fit.glm.dummy
fit.glm.dummy$finalModel
summary(fit.glm.dummy$finalModel)
fit.glm.dummy$preProcess

varImp(fit.glm.dummy, scale = FALSE)

#
# テストデータにモデルを当てはめる ( Prob )
#
allProb.glm.dummy <- caret::extractProb(
                                        list(fit.glm.dummy)
                                        ,testX = subset(train.dummy.test, select = -c(response))
                                        ,testY = unlist(subset(train.dummy.test, select = c(response)))
                                       )

# dataType 列に Test と入っているもののみを抜き出す
testProb.glm.dummy <- subset(allProb.glm.dummy, dataType == "Test")
tp_glm.dummy <- subset(testProb.glm.dummy, object == "Object1")

# confusionMatrix で比較
confusionMatrix(tp_glm.dummy$pred, tp_glm.dummy$obs)$overall[1]

# ROC
pROC::roc(tp_glm.dummy$obs, tp_glm.dummy$yes)

# predict() を利用した検算 
if (is.null(fit.glm.dummy$preProcess)){
  # preProcess を指定していない場合
  pred_test.verification <- predict(
                                    fit.glm.dummy$finalModel
                                    ,subset(train.dummy.test, select = -c(response))
                                    ,type = "response"
                                   )
} else {
  # preProcess を指定している場合
  pred_test.verification <- preProcess(
                                        subset(train.dummy.test ,select = -c(response))
                                        ,method = my_preProcess
                                      ) %>%
    predict(., subset(train.dummy.test, select = -c(response))) %>%
    predict(fit.glm.dummy$finalModel, ., type="response")
}

# ROC
pROC::roc(train.dummy.test[,"response"], pred_test.verification)


#
# 予測データにモデルの当てはめ
#
if (is.null(fit.glm.dummy$preProcess)){
  # preProcess を指定していない場合
  pred_test <- predict(fit.glm.dummy$finalModel, test.dummy, type="response")
  
  PREPROCESS <- "no_preProcess"
} else {
  # preProcess を指定している場合
  pred_test <- preProcess(test.dummy, method = my_preProcess) %>%
    predict(., test.dummy) %>%
    predict(fit.glm.dummy$finalModel, ., type="response")
  
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
