mxnetResult <- function() {
  df <- data.frame(
    test_ROC_1 = pROC::roc(TRAIN.TEST.PRED["response",], preds_train.test[,1])$auc[1]
    ,test_ROC_2 = pROC::roc(TRAIN.TEST.PRED["response",], preds_train.test[,2])$auc[1]
    ,train_ROC_1 = pROC::roc(TRAIN.TRAIN.PRED["response",], preds_train.train[,1])$auc[1]
    ,train_ROC_2 = pROC::roc(TRAIN.TRAIN.PRED["response",], preds_train.train[,2])$auc[1]
    ,num.round = NUM.ROUND
    ,array.batch.size = ARRAY.BATCH.SIZE
    ,learning.rate = LEARNING.RATE
    ,momentum = MOMENTUM
  )
  
  return(df)
}