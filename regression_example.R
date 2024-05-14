### Test of Regression Approach 


# libraries ---------------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(haven)
library(textrecipes)


# import and wrangle  --------------------------------------------------------------------

data = read_dta("tweet_classification/data/2013dataPSU.dta") |> 
  janitor::clean_names()



# eda ---------------------------------------------------------------------

data |> 
  distinct(name, ideology) |> 
  ggplot(aes(ideology))+
  geom_histogram()+
  theme_bw()


# preprocessing -----------------------------------------------------------

set.seed(0701)

split = initial_split(data)  

train = training(split)
test = testing(split)

recipe = recipe(ideology ~ text, data = train) |> 
  step_tokenize(text) |> 
  step_tokenfilter(text, max_tokens = 1000) |> 
  step_tfidf(text) |> 
  step_normalize(all_predictors())

prep = prep(recipe)
bake = bake(prep, new_data = NULL)

workflow = workflow() |> 
  add_recipe(recipe)

# set support vector machine model 
svm_spec = svm_linear() |> 
  set_mode("regression") |> 
  set_engine("LiblineaR")

# testing whether the specification is the issue 
svm_wf = workflow |> 
  add_model(svm_spec)


# support vector machine  -----------------------------------------------------------------

# create support vector machine fit 
svm_fit = workflow |> 
  add_model(svm_spec) |> 
  fit(data = train)

# model validation 
set.seed(815)

folds = vfold_cv(train, v = 10)

svm_validation = fit_resamples(workflow |> add_model(svm_spec), 
                               folds,
                               control = control_grid(save_pred = F, # cannot get this model to save predictions
                                                      verbose = T))
collect_metrics(svm_validation)


# compare to null ---------------------------------------------------------

null_regression <- null_model() %>%
  set_engine("parsnip") %>%
  set_mode("regression")

null_rs <- fit_resamples(
  workflow %>% add_model(null_regression),
  folds,
  metrics = metric_set(rmse)
)

null_rs

collect_metrics(null_rs)


# random forest -----------------------------------------------------------

rf_spec <- rand_forest(trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("regression")

rf_rs <- fit_resamples(
  workflow %>% add_model(rf_spec),
  folds,
  control = control_resamples(save_pred = TRUE,
                              verbose = T)
)

collect_metrics(rf_rs)
