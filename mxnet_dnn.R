require(caret)
require(mxnet)
require(dplyr)

source("./mxnetResult.R")
result.mxnet.df <- readRDS("result/result.mxnet.df.data")

#
# 前処理
#
source("./Data-pre-processing.R")

my_preProcess <- c("range")

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

# 説明変数一覧の作成
explanation_variable <- names(subset(TRAIN, select = -c(response)))


# response を数値化
TRAIN.TRAIN$response <- as.character(TRAIN.TRAIN$response)
TRAIN.TRAIN$response[TRAIN.TRAIN$response == "yes"] <- 1
TRAIN.TRAIN$response[TRAIN.TRAIN$response == "no"] <- 0
TRAIN.TRAIN$response <- as.numeric(TRAIN.TRAIN$response)

TRAIN.TEST$response <- as.character(TRAIN.TEST$response)
TRAIN.TEST$response[TRAIN.TEST$response == "yes"] <- 1
TRAIN.TEST$response[TRAIN.TEST$response == "no"] <- 0
TRAIN.TEST$response <- as.numeric(TRAIN.TEST$response)

TRAIN.TRAIN.PRED <- preProcess(TRAIN.TRAIN, method = my_preProcess) %>%
  predict(., TRAIN.TRAIN) %>%
  data.matrix(.) %>%
  t(.)

TRAIN.TEST.PRED <- preProcess(TRAIN.TEST, method = my_preProcess) %>%
  predict(., TRAIN.TEST) %>%
  data.matrix(.) %>%
  t(.)


#
TEST.PRED <- preProcess(TEST, method = my_preProcess) %>%
  predict(., TEST) %>%
  data.matrix(.) %>%
  t(.)


# Deep NN
data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name = "fc1", num_hidden = 128)
act1 <- mx.symbol.Activation(fc1, name = "relu1", act_type = "relu")
fc2 <- mx.symbol.FullyConnected(act1, name = "fc2", num_hidden = 64)
act2 <- mx.symbol.Activation(fc2, name = "relu2", act_type = "relu")
fc3 <- mx.symbol.FullyConnected(act2, name = "fc3", num_hidden = 2)
softmax <- mx.symbol.SoftmaxOutput(fc3, name = "sm")
devices <- mx.cpu()
mx.set.seed(0)

#for(i in seq(10,500, by = 10)){
for(i in 30:50){
  NUM.ROUND <- i
  ARRAY.BATCH.SIZE <- 100
  LEARNING.RATE <- 0.01
  MOMENTUM <- 0.9
  
  model <- mx.model.FeedForward.create(
    softmax
    ,X = TRAIN.TRAIN.PRED[explanation_variable,]
    ,y = TRAIN.TRAIN.PRED["response",]
    ,ctx = devices
    ,num.round = NUM.ROUND
    ,array.batch.size = ARRAY.BATCH.SIZE
    ,learning.rate = LEARNING.RATE
    ,momentum = MOMENTUM
    ,eval.metric = mx.metric.accuracy
    ,initializer = mx.init.uniform(0.07)
    ,epoch.end.callback = mx.callback.log.train.metric(100)
  )
  
  
  # 学習用データで検証
  preds_train.train <- predict(model, TRAIN.TRAIN.PRED[explanation_variable,], ctx = devices) %>%
            t(.)
  
  pROC::roc(TRAIN.TRAIN.PRED["response",], preds_train.train[,1])
  pROC::roc(TRAIN.TRAIN.PRED["response",], preds_train.train[,2])
  
  
  # テスト用データで検証
  preds_train.test <- predict(model, TRAIN.TEST.PRED[explanation_variable,], ctx = devices) %>%
            t(.)
  
  pROC::roc(TRAIN.TEST.PRED["response",], preds_train.test[,1])
  pROC::roc(TRAIN.TEST.PRED["response",], preds_train.test[,2])
  
  
  # 結果の保存
  result.mxnet.df <- rbind(result.mxnet.df, mxnetResult())
  #saveRDS(result.mxnet.df, "result/result.mxnet.df.data")
}


# グラフ描画
g <- ggplot(NULL) + ylab("ROC")
g <- g + geom_line(data = result.mxnet.df, aes(x = num.round, y = test_ROC_1, colour = "test"))
g <- g + geom_line(data = result.mxnet.df, aes(x = num.round, y = train_ROC_1, colour = "train"))
print(g)


#
# 予測データにモデルの当てはめ
#
pred_test <- predict(model, TEST.PRED, ctx = devices) %>%
  t(.)

out <- data.frame(TEST$id, pred_test[,2])
PREPROCESS <- paste(my_preProcess, collapse = "_")


# 予測データを保存
for(NUM in 1:10){
  DATE <- format(jrvFinance::edate(from = Sys.Date(), 0), "%Y%m%d")
  SUBMIT_FILENAME <- paste("./submit/submit_", DATE, "_", NUM, "_", PREPROCESS, "_mxnet.csv", sep = "")
  
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