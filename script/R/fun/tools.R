summaryResult <- function(MODEL_LIST) {
  # control
  if ( is.null(MODEL_LIST$call$control)) {
    cont <- "NA"
  } else cont <- data.frame(control = MODEL_LIST$call$control)
  
  # preProcess
  if ( is.null(names(MODEL_LIST$preProcess$method))){
    prep <- "NA"
  } else prep <- names(MODEL_LIST$preProcess$method)
  
  # tuneGrid
  if ( is.null(MODEL_LIST$call$tuneGrid)){
    grid <- "NA"
  } else grid <- apply(MODEL_LIST$call$tuneGrid, 2, paste, collapse = ",") %>%
    stringr::str_split(string = ., pattern = ",") %>%
    lapply(., unique) %>%
    lapply(., paste, collapse = ",") %>%
    data.frame(.)
  
  
  names(grid) <- paste0("tuneGrid.", names(model_list[[1]]$call$tuneGrid))
  
  # 
  df <- data.frame(
    test_ROC = pROC::roc(tp$obs, tp$yes)$auc[1]
    ,train_ROC = max(model_list[[1]]$results[,model_list[[1]]$metric])
    ,method = MODEL_LIST$method
    ,elapsed = MODEL_LIST$times$everything["elapsed"]
    ,data_preProcess = data_preProcess
    ,caret_preProcess = data.frame(caret_preProcess = paste(prep, collapse = ", "))
    ,length_exp = length(explanation_variable)
    ,exp = data.frame(exp = paste(explanation_variable, collapse = ", "))
    ,bestTune = MODEL_LIST$bestTune
    ,grid
    ,cont
    ,row.names = NULL
    ,stringsAsFactors = FALSE
  )
  
  return(df)
}

# data.frame の列が factor のものを、character に変換する関数
tf <- function(DF){
  variable.is.factor <- sapply(DF, is.factor)
  variable.factor <- names(variable.is.factor[variable.is.factor == TRUE])
  DF <- dplyr::mutate_each_(DF, dplyr::funs(as.character), variable.factor)
  return(DF)
}

# Dummy 化された data.frame の不要な列名を排除する関数
exclude_variables <- function(variables){
  id <- NULL
  for(i in variables){
    id <- c(id, grep(i, names(TRAIN)))
  }
  return(names(TRAIN[, -id]))
}