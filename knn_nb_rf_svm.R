library(tidymodels)
library(vroom)
library(embed)

source("ggg_recipe.R")

ggg_train <- vroom("train.csv")
ggg_train$type <- as.factor(ggg_train$type)
ggg_test <- vroom("test.csv")

ggg_rec <- ggg_recipe(ggg_train)



# ### KNN
# 
# ggg_knn <- nearest_neighbor(neighbors=tune()) |> 
#   set_mode("classification") |> 
#   set_engine("kknn")
# 
# knn_grid <- grid_regular(
#   neighbors(),
#   levels = 10
# )
# 
# knn_wf <- workflow() |> 
#   add_model(ggg_knn) |> 
#   add_recipe(ggg_rec)
# 
# folds <- vfold_cv(ggg_train, v = 10, repeats = 1)
# 
# CV_results <- run_cv(knn_wf, folds, knn_grid, metric = metric_set(accuracy),
#                      cores = 7)
# 
# bestTune <- CV_results |> 
#   select_best(metric = "accuracy")
# 
# bestTune$neighbors # 10
# 
# final_wf <- knn_wf |> 
#   finalize_workflow(bestTune) |> 
#   fit(data=ggg_train)
# 
# knn_preds <- final_wf |> 
#   predict(new_data = ggg_test,
#           type = "class")
# 
# kaggle_submission <- knn_preds |> 
#   bind_cols(ggg_test) |> 
#   select(id, .pred_class) |> 
#   rename(type = .pred_class)
# 
# vroom_write(x=kaggle_submission, file="./results/knn.csv", delim = ",")



# ### Naive Bayes
# 
# library(naivebayes)
# library(discrim)
# 
# ggg_nbayes <- naive_Bayes(Laplace = tune(),
#                           smoothness = tune()) |> 
#   set_mode("classification") |> 
#   set_engine("naivebayes")
# 
# nbayes_wf <- workflow() |> 
#   add_recipe(ggg_rec) |> 
#   add_model(ggg_nbayes)
# 
# nbayes_grid <- grid_regular(
#   Laplace(),
#   smoothness(),
#   levels = 5)
# 
# folds <- vfold_cv(ggg_train, v = 5, repeats = 1)
# 
# CV_results <- run_cv(nbayes_wf, folds, nbayes_grid, metric = metric_set(accuracy),
#                      cores = 7)
# 
# bestTune <- CV_results |> 
#   select_best(metric = "accuracy")
# 
# bestTune$smoothness # .5
# bestTune$Laplace # 0
# 
# final_wf <- nbayes_wf |> 
#   finalize_workflow(bestTune) |> 
#   fit(data=ggg_train)
# 
# nbayes_preds <- final_wf |> 
#   predict(new_data = ggg_test,
#           type = "class")
# 
# kaggle_submission <- nbayes_preds |> 
#   bind_cols(ggg_test) |> 
#   select(id, .pred_class) |> 
#   rename(Id = id,
#          type = .pred_class)
# 
# vroom_write(x=kaggle_submission, file="./results/nbayes.csv", delim = ",")



# ### rforest
# 
# library(ranger)
# 
# ggg_forest <- rand_forest(mtry = tune(),
#                           min_n = tune(),
#                           trees = 1000) |> 
#   set_mode("classification") |> 
#   set_engine("ranger")
# 
# forest_grid <- grid_regular(
#   mtry(range = c(1, 50)),
#   min_n(),
#   levels = 5)
# 
# forest_wf <- workflow() |> 
#   add_model(ggg_forest) |> 
#   add_recipe(ggg_rec)
# 
# folds <- vfold_cv(ggg_train, v = 10, repeats = 1)
# 
# CV_results <- run_cv(forest_wf, folds, forest_grid, metric = metric_set(accuracy),
#                      cores = 8)
# 
# bestTune <- CV_results |> 
#   select_best(metric = "accuracy")
# 
# bestTune$mtry
# bestTune$min_n
# 
# final_wf <- forest_wf |> 
#   finalize_workflow(bestTune) |> 
#   fit(data=ggg_train)
# 
# forest_preds <- final_wf |> 
#   predict(new_data = ggg_test,
#           type = "class")
# 
# kaggle_submission <- forest_preds |> 
#   bind_cols(ggg_test) |> 
#   select(id, .pred_class) |> 
#   rename(Id = id,
#          type = .pred_class)
# 
# vroom_write(x=kaggle_submission, file="./results/rforest.csv", delim = ",")



### Support Vector Machine

library(kernlab)

# svmPoly <- svm_poly(degree=tune(), cost=tune()) %>% # set or tune
#   set_mode("classification") %>%
#   set_engine("kernlab")

svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

# svmLinear <- svm_linear(cost=tune()) %>% # set or tune
#   set_mode("classification") %>%
#   set_engine("kernlab")

## Fit or Tune Model HERE (radial)

svmRad_wf <- workflow() |> 
  add_recipe(ggg_rec) |> 
  add_model(svmRadial)

# svmPoly_grid <- grid_regular(
#   degree(),
#   cost(),
#   levels = 5)

svmRad_grid <- grid_regular(
  rbf_sigma(),
  cost(),
  levels = 5)

# svmLin_grid <- grid_regular(
#   cost(),
#   levels = 5)

folds <- vfold_cv(ggg_train, v = 10, repeats = 1)

CV_results <- run_cv(svmRad_wf, folds, svmRad_grid, metric = metric_set(accuracy),
                     cores = 7, parallel = T)

bestTune <- CV_results |> 
  select_best(metric = "accuracy")

bestTune$rbf_sigma # 0.003162278
bestTune$cost # 32

final_wf <- svmRad_wf |> 
  finalize_workflow(bestTune) |> 
  fit(data=ggg_train)

svmRad_preds <- final_wf |> 
  predict(new_data = ggg_test,
          type = "class")

kaggle_submission <- svmRad_preds |> 
  bind_cols(ggg_test) |> 
  select(id, .pred_class) |> 
  rename(type = .pred_class)

vroom_write(x=kaggle_submission, file="./results/svmRad.csv", delim = ",")
