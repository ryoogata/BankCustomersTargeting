require(caret)
require(caretEnsemble)
require(pROC)
require(doParallel)

require(rpart)
require(partykit)
#require(rattle)

#
# 前処理
#
source("./Data-pre-processing.R")

my_preProcess <- c("center", "scale")

TRAIN <- train.dummy.nzv.highlyCorDescr
TRAIN.TRAIN <- train.dummy.nzv.highlyCorDescr.train
TRAIN.TEST <- train.dummy.nzv.highlyCorDescr.test
TEST <- test.dummy.nzv.highlyCorDescr

TRAIN <- train.dummy
TRAIN.TRAIN <- train.dummy.train
TRAIN.TEST <- train.dummy.test
TEST <- test.dummy

TRAIN <- all.train
TRAIN.TRAIN <- train.train
TRAIN.TEST <- train.test
TEST <- test

#
# rpart
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
explanation_variable.rpart <- names(subset(TRAIN, select = -c(response)))

minbucket_variable <- 100
maxdepth_variable <- 14 
cp_variable <- 0.00142

# fit.rpart <-
#   train(
#          x = TRAIN.TRAIN[,explanation_variable.rpart ]
#         ,y = TRAIN.TRAIN$response
#         ,method = "rpart"
#         ,metric = "ROC"
#         ,tuneGrid = expand.grid(
#           cp = seq(0, 0.001,by = 0.0001)
#         )
#         ,trControl = trainControl(
#                                   method = "cv"
#                                   ,number = 10
#                                   ,summaryFunction = twoClassSummary
#                                   ,classProbs = TRUE
#                                   ,verbose = TRUE
#                                   ,savePredictions = "final"
#                                   ,index = createResample(TRAIN.TRAIN$response, 10)
#                                   ,seeds = seeds
#                                  )
#         ,control = rpart.control(
#                                   maxdepth = maxdepth_variable
#                                   ,minbucket = minbucket_variable
#                                   ,method = "class"
#                                 )
#         )

cl <- makeCluster(detectCores(), type = 'PSOCK', outfile = " ")
registerDoParallel(cl)

model_list_rpart <- caretList(
  x = TRAIN.TRAIN[,explanation_variable.rpart]
  ,y = TRAIN.TRAIN$response
  ,trControl = my_control
  #,trControl = doParallel
  #,preProcess = my_preProcess
  ,tuneList = list(
    rpart = caretModelSpec(
      method = "rpart"
      ,metric = "ROC"
      ,tuneGrid = expand.grid(
        cp = seq(0, 0.001,by = 0.0001)
      )
      ,control = rpart.control(
                                maxdepth = maxdepth_variable
                                ,minbucket = minbucket_variable
                                ,method = "class"
      )
    )
  )
)

stopCluster(cl)
registerDoSEQ()

fit.rpart <- model_list_rpart[[1]]

fit.rpart$times
# $everything
# ユーザ   システム       経過  
# 10.936      0.448     11.450 

fit.rpart
fit.rpart$finalModel
#rattle::fancyRpartPlot(fit.rpart$finalModel)
#fit.rpart$finalModel$variable.importance
varImp(fit.rpart, scale = FALSE, useModel = FALSE)
varImp(fit.rpart, scale = FALSE)
plot(varImp(fit.rpart, scale = FALSE))

ggplot(fit.rpart) 


#
# テストデータにモデルを当てはめる ( Prob )
#
allProb.rpart <- caret::extractProb(
                                    list(fit.rpart)
                                    ,testX = subset(TRAIN.TEST, select = -c(response))
                                    ,testY = unlist(subset(TRAIN.TEST, select = c(response)))
                                    )

# dataType 列に Test と入っているもののみを抜き出す
testProb.rpart <- subset(allProb.rpart, dataType == "Test")
tp_rpart <- subset(testProb.rpart, object == "Object1")

# confusionMatrix で比較
confusionMatrix(tp_rpart$pred, tp_rpart$obs)$overall[1]

# ROC
pROC::roc(tp_rpart$obs, tp_rpart$yes)

# predict() を利用した検算 
if (is.null(fit.rpart$preProcess)){
  # preProcess を指定していない場合
  pred_test.verification <- predict(
    fit.rpart$finalModel
    ,subset(TRAIN.TEST, select = -c(response))
  )
} else {
  # preProcess を指定している場合
  pred_test.verification <- preProcess(
    subset(TRAIN.TEST, select = -c(response))
    ,method = my_preProcess
  ) %>%
    predict(., subset(TRAIN.TEST, select = -c(response))) %>%
    predict(fit.rpart$finalModel, .)
}

#ROC
pROC::roc(TRAIN.TEST[,"response"], pred_test.verification[,2])


#
# 予測データにモデルの当てはめ
#
if (is.null(fit.rpart$preProcess)){
  # preProcess を指定していない場合
  pred_test <- predict(fit.rpart$finalModel, test, type="response")[,2]
  
  PREPROCESS <- "no_preProcess"
} else {
  # preProcess を指定している場合
  pred_test <- preProcess(test, method = my_preProcess) %>%
    predict(., test) %>%
    predict(fit.rpart$finalModel, .)
  
  pred_test <- pred_test[,2]
  
  PREPROCESS <- paste(my_preProcess, collapse = "_")
}


#submitの形式で出力(CSV)
#データ加工
out.rpart <- data.frame(test$id, pred_test)

# 予測データを保存
for(NUM in 1:10){
  DATE <- format(jrvFinance::edate(from = Sys.Date(), 0), "%Y%m%d")
  SUBMIT_FILENAME <- paste("./submit/submit_", DATE, "_", NUM, "_", PREPROCESS, "_rpart.csv", sep = "")
  
  if ( !file.exists(SUBMIT_FILENAME) ) {
    write.table(out.rpart, #出力データ
                SUBMIT_FILENAME, #出力先
                quote = FALSE, #文字列を「"」で囲む有無
                col.names = FALSE, #変数名(列名)の有無
                row.names = FALSE, #行番号の有無
                sep = "," #区切り文字の指定
    )
    break
  }
}
