library(tidymodels)
library(vroom)
library(embed)
library(stacks)

source("ggg_recipe.R")

ggg_train <- vroom("train.csv")
ggg_train$type <- as.factor(ggg_train$type)
ggg_test <- vroom("test.csv")

ggg_rec <- ggg_recipe(ggg_train)

untunedModel <- control_stack_grid()
tunedModel <- control_stack_resamples()

folds <- vfold_cv(ggg_train, v = 10, repeats = 1)

### Naive Bayes

library(naivebayes)
library(discrim)

ggg_nbayes <- naive_Bayes(Laplace = 0,
                          smoothness = 1) |>
  set_mode("classification") |>
  set_engine("naivebayes")

nbayes_wf <- workflow() |>
  add_recipe(ggg_rec) |>
  add_model(ggg_nbayes)

nbayes_model <- fit_resamples(nbayes_wf,
                              resamples=folds,
                              metrics=metric_set(accuracy, roc_auc),
                              control=tunedModel)



### Boosted Trees

library(bonsai)

ggg_boost <- boost_tree(tree_depth = 1,
                        trees = 1000,
                        learn_rate = 0.1) |> 
  set_engine("lightgbm") |> 
  set_mode("classification")

boost_wf <- workflow() |> 
  add_recipe(ggg_rec) |>
  add_model(ggg_boost)

boost_model <- fit_resamples(boost_wf,
                             resamples=folds,
                             metrics=metric_set(accuracy, roc_auc),
                             control=tunedModel)



### Support Vector Machine

library(kernlab)

svmRadial <- svm_rbf(rbf_sigma=0.003162278,
                     cost=2.378414) |>
  set_mode("classification")  |> 
  set_engine("kernlab")

svmRad_wf <- workflow() |>
  add_recipe(ggg_rec) |>
  add_model(svmRadial)

svmRad_model <- fit_resamples(svmRad_wf,
                              resamples=folds,
                              metrics=metric_set(accuracy, roc_auc),
                              control=tunedModel)



### Random Forest

library(ranger)

ggg_forest <- rand_forest(mtry = tune(),
                           min_n = tune(),
                           trees = 1000) |> 
  set_engine("ranger") |> 
  set_mode("classification")

rforest_wf <- workflow() |> 
  add_recipe(ggg_rec) |> 
  add_model(ggg_forest)

rforest_grid <- grid_regular(
  mtry(range = c(1, ncol(juice(prep(ggg_rec))))-1),
  min_n(),
  levels = 10)

rforest_models <- tune_grid(
  rforest_wf,
  resamples = folds,
  grid = rforest_grid,
  metrics = metric_set(accuracy, roc_auc),
  control = untunedModel
)



### Stack the Models

my_stack <- stacks() |> 
  add_candidates(nbayes_model) |> 
  add_candidates(boost_model) |> 
  add_candidates(svmRad_model) |> 
  add_candidates(rforest_models)

stack_model <- my_stack |> 
  blend_predictions() |> 
  fit_members()

stack_preds <- stack_model |> predict(new_data = ggg_test,
                                      type = "class")

kaggle_submission <- stack_preds |>
  bind_cols(ggg_test) |>
  select(id, .pred_class) |>
  rename(type = .pred_class)

vroom_write(x=kaggle_submission, file="./results/stack2.csv", delim = ",")
