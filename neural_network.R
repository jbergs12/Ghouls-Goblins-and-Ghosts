library(tidymodels)
library(vroom)
library(embed)
# install.packages("remotes")
# remotes::install_github("rstudio/tensorflow")

source("ggg_recipe.R")

ggg_train <- vroom("train.csv")
ggg_train$type <- as.factor(ggg_train$type)
ggg_test <- vroom("test.csv")

ggg_rec <- recipe(type~., data=ggg_train) |> 
  # update_role(id, new_role="id") |> 
  step_mutate_at(all_nominal_predictors(), fn = factor) |> 
  step_dummy(color) |>
  step_range(all_numeric_predictors(), min = 0, max = 1)

ggg_nn <- mlp(hidden_units = tune(),
              epochs = 50) |> 
  set_engine("keras") |> 
  set_mode("classification")

nn_wf <- workflow() |> 
    add_recipe(ggg_rec) |>
    add_model(ggg_nn)

nn_grid <- grid_regular(hidden_units(range=c(1,15)),
                        levels = 5)

folds <- vfold_cv(ggg_train, v = 5, repeats = 1)

CV_results <- run_cv(nn_wf, folds, nn_grid, metric = metric_set(roc_auc, accuracy),
                     cores = 7)
  
CV_results |> 
  collect_metrics() |> 
  filter(.metric=='accuracy') |> 
  ggplot(aes(x=hidden_units, y=mean)) +
  geom_line()

bestTune <- CV_results |> 
  select_best(metric = "accuracy")

bestTune$hidden_units # 15

final_wf <- nn_wf |> 
  finalize_workflow(bestTune) |> 
  fit(data=ggg_train)

nn_preds <- final_wf |> 
  predict(new_data = ggg_test,
          type = "class")

kaggle_submission <- nn_preds |> 
  bind_cols(ggg_test) |> 
  select(id, .pred_class) |> 
  rename(type = .pred_class)

vroom_write(x=kaggle_submission, file="./results/neuralnet.csv", delim = ",")
