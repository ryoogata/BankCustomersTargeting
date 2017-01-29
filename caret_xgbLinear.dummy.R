require(xgboost)
require(Matrix)
require(doParallel)
require(caretEnsemble)

#
# 前処理
#
source("./Data-pre-processing.R")

my_preProcess <- c("center", "scale")

#
# xgbLinear
#

# 説明変数一覧の作成
explanation_variable.xgbLinear.dummy <- names(subset(train.dummy, select = -c(response)))

# seeds の決定
set.seed(123)
seeds <- vector(mode = "list", length = 51)
for(i in 1:50) seeds[[i]] <- sample.int(1000, 500)
seeds[[51]] <- sample.int(1000, 1)

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

# fit.xgbLinear <-
#   train(
#         x = train.dummy.train[,explanation_variable.xgbLinear]
#         ,y = train.dummy.train$response
#         ,trControl = doParallel
#         ,method = "xgbLinear"
#         ,metric = "ROC" 
#         ,label = train.dummy.train$response
#         #,objective = "binary:logistic"
#         #,eval_metric = "auc"
#         ,tuneGrid = expand.grid(
#                                 nrounds = c(1:20*10)
#                                 ,lambda = c(.1, .4)
#                                 ,alpha =  c(.1, .4)
#                                 ,eta = c(.1, .4)
#                     )
#         )

cl <- makeCluster(detectCores(), type = 'PSOCK', outfile = " ")
registerDoParallel(cl)

model_list_xgbLinear.dummy <- caretList(
  x = train.dummy.train[,explanation_variable.xgbLinear.dummy]
  ,y = train.dummy.train$response
  ,trControl = doParallel
  #,preProcess = my_preProcess
  ,tuneList = list(
    xgboost = caretModelSpec(
        method = "xgbLinear"
        ,metric = "ROC" 
        ,label = train.dummy.train$response
        ,tuneGrid = expand.grid(
                                nrounds = c(50)
                                ,lambda = c(.3)
                                ,alpha =  c(1)
                                ,eta = c(.1)
                    )
        )
    )
)

stopCluster(cl)
registerDoSEQ()

fit.xgbLinear.dummy <- model_list_xgbLinear.dummy[[1]]

# 2017/01/15
# Fitting nrounds = 50, lambda = 0.1, alpha = 0.4, eta = 0.1 on full training set

# tuneGrid = expand.grid(nrounds = c(50),lambda = c(1:10/10),alpha =  c(1:10/10),eta = c(1:10/10))
# Fitting nrounds = 50, lambda = 0.3, alpha = 1, eta = 0.1 on full training set
# $everything
# user    system   elapsed 
# 446.952    80.248 74894.204 ( 21 hour )

# tuneGrid = expand.grid(nrounds = c(50, 60, 70),lambda = c(1:2/10),alpha =  c(10:11/10),eta = c(1:2/10)
# Fitting nrounds = 50, lambda = 0.2, alpha = 1, eta = 0.1 on full training set
# $everything
# user  system elapsed 
# 14.12    1.86 2158.57 ( 35 minute )

# tuneGrid = expand.grid(nrounds = c(30, 40, 50),lambda = c(1:4/10),alpha = c(1),eta = c(.1)
# Fitting nrounds = 50, lambda = 0.3, alpha = 1, eta = 0.1 on full training set
# $everything
# user  system elapsed 
# 8.492   0.544 739.235 ( 13 minute )

# tuneGrid = expand.grid(nrounds = c(48:53),lambda = c(.3),alpha = c(1),eta = c(.1)
# Fitting nrounds = 50, lambda = 0.3, alpha = 1, eta = 0.1 on full training set
# $everything
# user  system elapsed 
# 5.532   0.292 456.456 ( 8 minute )


fit.xgbLinear.dummy$times
# $everything
# ユーザ   システム       経過  
# 12.079      0.517     12.710 

fit.xgbLinear.dummy$finalModel
ggplot(fit.xgbLinear.dummy) 
fit.xgbLinear.dummy$preProcess

#
# テストデータにモデルを当てはめる ( Prob )
#
allProb.xgbLinear <- caret::extractProb(list(fit.xgbLinear.dummy),
                                    testX = subset(train.dummy.test, select = -c(response)),
                                    testY = unlist(subset(train.dummy.test, select = c(response))))

# dataType 列に Test と入っているもののみを抜き出す
testProb.xgbLinear <- subset(allProb.xgbLinear, dataType == "Test")

tp_xgbLinear <- subset(testProb.xgbLinear, object == "Object1")
confusionMatrix(tp_xgbLinear$pred, tp_xgbLinear$obs)$overall[1]

# ROC
pROC::roc(tp_xgbLinear$obs, tp_xgbLinear$yes)

# predict() を利用した検算 
if (is.null(fit.xgbLinear.dummy$preProcess)){
  # preProcess を指定していない場合
  pred_test.verification <- predict(
    fit.xgbLinear.dummy$finalModel
    ,as.matrix(subset(train.dummy.test, select = -c(response)))
  )
} else {
  # preProcess を指定している場合
  pred_test.verification <- preProcess(
    subset(train.dummy.test ,select = -c(response))
    ,method = my_preProcess
  ) %>%
    predict(., subset(train.dummy.test, select = -c(response))) %>%
    as.matrix(.) %>%
    predict(fit.xgbLinear.dummy$finalModel, .)
}

# ROC
pROC::roc(train.dummy.test[,"response"], pred_test.verification)


#
# 予測データにモデルの当てはめ
#
#pred_test.xgbLinear <- predict(fit.xgbLinear.dummy$finalModel, Matrix::Matrix(as.matrix(test.dummy), sparse=T))
#pred_test.xgbLinear <- predict(fit.xgbLinear.dummy$finalModel, as.matrix(test.dummy))
#pred_test.xgbLinear <- 1 - pred_test.xgbLinear

if (is.null(fit.xgbLinear.dummy$preProcess)){
  # preProcess を指定していない場合
  pred_test.xgbLinear  <- predict(
    fit.xgbLinear.dummy$finalModel
    ,as.matrix(test.dummy)
  )
  pred_test.xgbLinear <- 1 - pred_test.xgbLinear
  PREPROCESS <- "no_preProcess"
} else {
  # preProcess を指定している場合
  pred_test.xgbLinear  <- preProcess(
    test.dummy
    ,method = my_preProcess
  ) %>%
    predict(., test.dummy) %>%
    as.matrix(.) %>%
    predict(fit.xgbLinear.dummy$finalModel, .)
  pred_test.xgbLinear <- 1 - pred_test.xgbLinear
  PREPROCESS <- paste(my_preProcess, collapse = "_")
}


#submitの形式で出力(CSV)
#データ加工
out.xgbLinear <- data.frame(test.dummy$id, pred_test.xgbLinear)

# 予測データを保存
for(NUM in 1:10){
  DATE <- format(jrvFinance::edate(from = Sys.Date(), 0), "%Y%m%d")
  SUBMIT_FILENAME <- paste("./submit/submit_", DATE, "_", NUM, "_", PREPROCESS, "_xgbLinear.dummy.csv", sep = "")
  
  if ( !file.exists(SUBMIT_FILENAME) ) {
    write.table(out.xgbLinear, #出力データ
                SUBMIT_FILENAME, #出力先
                quote = FALSE, #文字列を「"」で囲む有無
                col.names = FALSE, #変数名(列名)の有無
                row.names = FALSE, #行番号の有無
                sep = "," #区切り文字の指定
    )
    break
  }
}

saveRDS(fit.xgbLinear, file = "fit.xgbLinear.rdata")
fit.xgbLinear <- readRDS("fit.xgbLinear.rdata")