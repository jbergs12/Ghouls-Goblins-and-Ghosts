ggg_recipe <- function(traindata){
  recipe(type~., data=traindata) |> 
    step_mutate_at(all_nominal_predictors(), fn = factor) |>
    step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) |>
    #step_zv(all_predictors()) |> 
    step_normalize(all_numeric_predictors()) #|>
  #step_pca(all_predictors(), threshold = .85)
}

run_cv <- function(wf, folds, grid, metric=metric_set(rmse), cores=8, parallel = TRUE){
  if(parallel == TRUE){
    library(doParallel)
    
    cl <- makePSOCKcluster(cores)
    registerDoParallel(cl)
  }
  
  results <- wf |>
    tune_grid(resamples=folds,
              grid=grid,
              metrics=metric)
  if(parallel == TRUE){
    stopCluster(cl)
  }
  return(results)
}

ggg_smote_recipe <- function(traindata, neighbors){
  library(themis)
  recipe(type~., data=traindata) |> 
    step_mutate_at(all_nominal_predictors(), fn = factor) |>
    step_lencode_mixed(all_nominal_predictors(), outcome = vars(type)) |>
    step_normalize(all_numeric_predictors()) |>
    step_pca(all_predictors(), threshold = .85) |> 
    step_smote(all_outcomes(), neighbors=neighbors)
}