mxnetResult <- function() {
  df <- data.frame(
    test_ROC_1 = pROC::roc(test.y, preds_test[,1])$auc[1]
    ,test_ROC_2 = pROC::roc(test.y, preds_test[,2])$auc[1]
    ,train_ROC_1 = pROC::roc(train.y, preds_train[,1])$auc[1]
    ,train_ROC_2 = pROC::roc(train.y, preds_train[,2])$auc[1]
    ,num.round = NUM.ROUND
    ,array.batch.size = ARRAY.BATCH.SIZE
    ,learning.rate = LEARNING.RATE
    ,momentum = MOMENTUM
  )
  
  return(df)
}