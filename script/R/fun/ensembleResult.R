ensembleResult <- function(ENSEMBLE_LIST) {
  # preProcess
  if ( is.null(names(ENSEMBLE_LIST$models[[1]]$preProcess$method))){
    prep <- "NA"
  } else prep <- names(ENSEMBLE_LIST$models[[1]]$preProcess$method)
  
  df <- data.frame(
    roc.df
    ,train_ROC =  ENSEMBLE_LIST$ens_model$results[,ENSEMBLE_LIST$ens_model$metric]
    #,models = paste(names(ENSEMBLE_LIST$models), collapse = ", ")
    ,data_preProcess = data_preProcess
    ,caret_preProcess = data.frame(caret_preProcess = paste(prep, collapse = ", "))
    ,data.frame(t(sapply(names(ENSEMBLE_LIST$models), function(MODEL) {return (ENSEMBLE_LIST$models[[MODEL]]$times$everything["elapsed"])})))
    ,length_exp = length(explanation_variable)
    ,exp = data.frame(exp = paste(explanation_variable, collapse = ", "))
    ,bestTune = sapply(names(ENSEMBLE_LIST$models), function(MODEL) {return (ENSEMBLE_LIST$models[[MODEL]]$bestTune)})
  )  
  
  return(df)
}