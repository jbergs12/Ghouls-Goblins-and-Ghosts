library(tidymodels)
library(vroom)

ggg_train <- vroom("./train.csv")
ggg_test <- vroom("./test.csv")
ggg_na <- vroom("./trainWithMissingValues.csv")

ggg_train$type <- factor(ggg_train$type)
ggg_na$type <- factor(ggg_na$type)

# Imputation

colMeans(is.na(ggg_na))

ggg_rec <- recipe(type~., data=ggg_na) |>
  step_mutate_at("color", fn = factor) |>
  step_impute_knn(hair_length, impute_with = imp_vars(has_soul, color),
                  neighbors = 7) |>
  step_impute_knn(rotting_flesh, impute_with = imp_vars(has_soul, color, hair_length),
                  neighbors = 7) |>
  step_impute_knn(rotting_flesh, impute_with = imp_vars(has_soul, color, hair_length, bone_length),
                  neighbors = 7) #|>
  # step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) |>
  # step_normalize(all_numeric_predictors())

baked_rec <- bake(prep(ggg_rec), new_data = ggg_na)

rmse_vec(ggg_train[is.na(ggg_na)], baked_rec[is.na(ggg_na)])
