require(caret)
require(caretEnsemble)
require(pROC)
require(doParallel)

require(rpart)
require(partykit)
require(rattle)

source("script/R/fun/tools.R")
result.rpart.df <- readRDS("result/result.rpart.df.data")

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
# 欠損値処理
#

# 欠損値の確認
sapply(train, function(x) sum(is.na(x)))
sapply(test, function(x) sum(is.na(x)))
sapply(all, function(x) sum(is.na(x)))


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

minbucket_variable <- 100
maxdepth_variable <- 14 
cp_variable <- 0.00142

cl <- makeCluster(detectCores(), type = 'PSOCK', outfile = " ")
registerDoParallel(cl)

combi <- NA

# 説明変数一覧の作成
explanation_variable.orig <- names(subset(TRAIN, select = -c(response)))

for(i in 21:2){
  combi <- gtools::combinations(n = 21, r = i , v = explanation_variable.orig)
  
  for(j in 1:nrow(combi)){  
    print(combi[j,])
    model_list <- caretList(
      x = TRAIN.TRAIN[,as.vector(combi[j,])]
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
     
    
    fit.rpart <- model_list[[1]]
    
    #
    # テストデータにモデルを当てはめる ( Prob )
    #
    allProb <- caret::extractProb(
                                  list(fit.rpart)
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
    
    explanation_variable <- combi[j,]
    
    # 結果の保存
    result.rpart.df <- rbind(result.rpart.df, summaryResult(model_list[[1]]))
    #saveRDS(result.rpart.df, "result/result.rpart.df.data")
  }
}

stopCluster(cl)
registerDoSEQ()

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
pROC::roc(TRAIN.TEST[,"response"], pred_test.verification[,"yes"])


#
# 予測データにモデルの当てはめ
#
if (is.null(fit.rpart$preProcess)){
  # preProcess を指定していない場合
  pred_test <- predict(fit.rpart$finalModel, TEST, type="response")[,"yes"]
  
  PREPROCESS <- "no_preProcess"
} else {
  # preProcess を指定している場合
  pred_test <- preProcess(TEST, method = my_preProcess) %>%
    predict(., TEST) %>%
    predict(fit.rpart$finalModel, .)
  
  pred_test <- pred_test[,"yes"]
  
  PREPROCESS <- paste(my_preProcess, collapse = "_")
}


#submitの形式で出力(CSV)
#データ加工
out <- data.frame(TEST$id, pred_test)

# 予測データを保存
for(NUM in 1:10){
  DATE <- format(jrvFinance::edate(from = Sys.Date(), 0), "%Y%m%d")
  SUBMIT_FILENAME <- paste("./submit/submit_", DATE, "_", NUM, "_", PREPROCESS, "_rpart.csv", sep = "")
  
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
