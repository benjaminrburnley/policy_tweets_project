### Credit Claiming Classification Model v1 
## April 3, 2023


# libraries ---------------------------------------------------------------

library(tidyverse)
library(haven)
library(tidymodels)
library(textrecipes)
library(discrim)


# data --------------------------------------------------------------------

data = read_dta("data/2013dataPSU.dta")


# wrangle -----------------------------------------------------------------

# reclassify outcome variable 

data_cc = data |> 
  mutate(cc = if_else(CreditClaim == 1, "Credit Claiming", "Other"))


# preprocessing -----------------------------------------------------------

set.seed(0701)

# create initial split in data 
cc_split = initial_split(data_cc, strata = cc)

# create training set
cc_train = training(cc_split)

# create testing set 
cc_test = testing(cc_split)

# create recipe 
cc_rec = recipe(cc ~ Text, data = cc_train) |> 
  step_tokenize(Text) |> 
  step_tokenfilter(Text, max_tokens = 1000) |> 
  step_tfidf(Text)

# create workflow 
cc_wf = workflow() |> 
  add_recipe(cc_rec)


# naive bayes specification -----------------------------------------------

# create naive bayes model specification 
nb_spec = naive_Bayes() |> 
  set_mode("classification") |> 
  set_engine("naivebayes")

# fit model on training data
nb_fit = cc_wf |> 
  add_model(nb_spec) |> 
  fit(data = cc_train)

# validate model 
cc_folds = vfold_cv(cc_train)

# set validation wf 
nb_wf = workflow() |> 
  add_recipe(cc_rec) |> 
  add_model(nb_spec)

# resample 
nb_rs = fit_resamples(
  nb_wf,
  cc_folds,
  control = control_resamples(save_pred = T)
)

# metrics 
nb_rs_metrics = collect_metrics(nb_rs)
nb_rs_predictions = collect_predictions(nb_rs)

# plot ROCs
nb_rs_predictions %>%
  group_by(id) %>%
  roc_curve(truth = cc, `.pred_Credit Claiming`) %>%
  autoplot() +
  labs(
    color = NULL,
    title = "ROC curve for Credit Claiming in Tweets",
    subtitle = "Each resample fold is shown in a different color"
  )

# heat map 
conf_mat_resampled(nb_rs, tidy = FALSE) %>%
  autoplot(type = "heatmap")

