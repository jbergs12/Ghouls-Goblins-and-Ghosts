library(tidymodels)
library(vroom)
library(embed)
library(bonsai)

source("ggg_recipe.R")

ggg_train <- vroom("train.csv")
ggg_train$type <- as.factor(ggg_train$type)
ggg_test <- vroom("test.csv")

ggg_rec <- ggg_recipe(ggg_train)


### Boosted Trees

ggg_boost <- boost_tree(tree_depth = tune(),
                        trees = tune(),
                        learn_rate = tune()) |> 
  set_engine("lightgbm") |> 
  set_mode("classification")

boost_wf <- workflow() |> 
  add_recipe(ggg_rec) |>
  add_model(ggg_boost)

boost_grid <- grid_regular(tree_depth(),
                           trees(),
                           learn_rate(),
                           levels = 5)

folds <- vfold_cv(ggg_train, v = 5, repeats = 1)

boost_CV <- run_cv(boost_wf, folds, boost_grid, metric = metric_set(roc_auc, accuracy),
                   parallel = F)

bestTune <- boost_CV |> 
  select_best(metric = "accuracy")

bestTune$tree_depth # 1
bestTune$trees # 1000
bestTune$learn_rate # 0.1

final_wf <- boost_wf |> 
  finalize_workflow(bestTune) |> 
  fit(data=ggg_train)

boost_preds <- final_wf |> 
  predict(new_data = ggg_test,
          type = "class")

kaggle_submission <- boost_preds |> 
  bind_cols(ggg_test) |> 
  select(id, .pred_class) |> 
  rename(type = .pred_class)

vroom_write(x=kaggle_submission, file="./results/boosted_trees.csv", delim = ",")



### BART

ggg_bart <- parsnip::bart(trees = tune()) |>
  set_engine("dbarts") |>
  set_mode("classification")

bart_wf <- workflow() |>
  add_recipe(ggg_rec) |>
  add_model(ggg_bart)

bart_grid <- grid_regular(trees(),
                          levels = 10)

folds <- vfold_cv(ggg_train, v = 10, repeats = 1)

bart_CV <- run_cv(bart_wf, folds, bart_grid, metric = metric_set(accuracy),
                   cores = 7)

bestTune_bart <- bart_CV |> 
  select_best(metric = "accuracy")

bestTune_bart$trees # 1

final_wf_bart <- bart_wf |> 
  finalize_workflow(bestTune_bart) |> 
  fit(data=ggg_train)

bart_preds <- final_wf_bart |> 
  predict(new_data = ggg_test,
          type = "class")

kaggle_submission_bart <- bart_preds |> 
  bind_cols(ggg_test) |> 
  select(id, .pred_class) |> 
  rename(type = .pred_class)

vroom_write(x=kaggle_submission_bart, file="./results/bart.csv", delim = ",")
