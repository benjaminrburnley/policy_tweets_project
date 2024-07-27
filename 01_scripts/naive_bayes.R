#### Naive Bayes Models
###
##
#

# setup -------------------------------------------------------------------

# libaries 
library(tidyverse)
library(haven)
library(tidymodels)
library(textrecipes)
library(discrim)
library(naivebayes)
library(glmnet)
library(hardhat)
library(vip)
library(here)
library(themis)
theme_set(theme_bw())


# data --------------------------------------------------------------------

# data 
data = read_rds("00_data/data_cleaned_2013.rds") |> 
  mutate(policy_content = as.factor(policy_content))
norts = data |> 
  filter(retweet == 0) 
  

# outcome variable 
data |> 
  group_by(major_topic) |> 
  count() |> 
  ggplot(aes(reorder(major_topic, -n), n))+
  geom_col(fill = "dodgerblue")+
  theme_bw()+
  labs(
    title = "Number of Tweets per Topic - 2013",
    x = NULL,
    y = "Total Number of Tweets"
  )+
  theme(
    axis.text.x = element_text(angle = 90)
  )

# preprocessing -----------------------------------------------------------
# set seed for replication 
set.seed(815)
split = initial_split(data, strata = major_topic)

# create testing set
test = testing(split)
# create training set 
train = training(split)

# create base recipe 
nb_base = recipe(major_topic ~ text, data = train) |> 
  step_tokenize(text) |> 
  step_tokenfilter(text, max_tokens = 1000) |> 
  step_tfidf(text) |> 
  step_normalize(all_predictors())
  
# create full recipe
full_recipe = recipe(major_topic ~ text, data = train) |> 
  step_tokenize(text, token = "ngrams", options = list(n = 2, n_min = 1)) |>
  step_stopwords(text, stopword_source = "snowball") |> 
  step_stem(text) |> 
  step_tokenfilter(text, max_tokens = 1000) |> 
  step_tfidf(text)

# add normalization
full_norm_recipe = recipe(major_topic ~ text, data = train) |> 
  step_tokenize(text, token = "ngrams", options = list(n = 2, n_min = 1)) |>
  step_stopwords(text, stopword_source = "snowball") |> 
  step_stem(text) |> 
  step_tokenfilter(text, max_tokens = 1000) |> 
  step_tfidf(text) |> 
  step_normalize(all_predictors())

# check prep
nb_prep = prep(nb_base)

# check bake
nb_bake = bake(nb_prep, new_data = NULL)


# workflows and specifications --------------------------------------------

# null model 
null_regression <- null_model() %>%
  set_engine("parsnip") %>%
  set_mode("regression")

# naive bayes model 
nb_model = naive_Bayes() |> 
  set_mode("classification") |> 
  set_engine("naivebayes")

# naive bayes workflow 
nb_wf = workflow() |> 
  add_recipe(nb_base) |> 
  add_model(nb_model)

nb_full_wf = workflow() |> 
  add_recipe(full_recipe) |> 
  add_model(nb_model)

nb_full_norm_wf = workflow() |> 
  add_recipe(full_norm_recipe) |> 
  add_model(nb_model)
# fit ---------------------------------------------------------------------

# fit model 
nb_fit = nb_wf |> 
  fit(train)

nb_full = nb_full_wf |> 
  fit(train)

nb_full_norm = nb_full_norm_wf |> 
  fit(train)

# cross validate ----------------------------------------------------------

# create folds 
folds = vfold_cv(train)

# resample process
nb_rs <- fit_resamples(
  nb_wf,
  folds,
  control = control_resamples(save_pred = TRUE, verbose = TRUE, parallel_over = "resamples")
)


# evaluate ---------------------------------------------------------------

# get predictions
nb_predictions = predict(nb_fit, train)

# predictions table 
metrics = tibble(truth = as.factor(train$major_topic), prediction = nb_predictions$.pred_class)

# fit accuracy
accuracy(metrics, truth = truth, estimate = prediction)
f_meas(metrics, truth = truth, estimate = prediction)

        
# resample metris
nb_rs_metrics = collect_metrics(nb_rs)
nb_rs_predictions = collect_predictions(nb_rs)

nb_rs_predictions %>%
  group_by(id) %>%
  roc_curve(truth = major_topic, .pred_Agriculture:.pred_Transportation) |> 
  autoplot() +
  labs(
    color = NULL,
    title = "ROC curve for Polict Tweets",
    subtitle = "Each resample fold is shown in a different color"
  )

# full predictions 
nb_full_prediction = predict(nb_full, train)

full_met = tibble(truth = as.factor(train$major_topic), estimate = nb_full_prediction$.pred_class)

accuracy(full_met, truth = truth, estimate = estimate)
f_meas(full_met, truth = truth, estimate = estimate)


# try with normalization
nb_full_norm_predictions = predict(nb_full_norm, train)
full_norm_met = tibble(truth = as.factor(train$major_topic), estimate = nb_full_norm_predictions$.pred_class)
accuracy(full_norm_met, truth, estimate)
f_meas(full_norm_met, truth, estimate)


# policy content ----------------------------------------------------------

# set seed for split 
set.seed(755)

policy_split = initial_split(data, strata = policy_content)
policy_train = training(policy_split)
policy_test = testing(policy_split)


policy_base = recipe(policy_content ~ text, data = policy_train) |> 
  step_tokenize(text) |> 
  step_tokenfilter(text, max_tokens = 1000) |> 
  step_tfidf(text) |> 
  step_normalize(all_predictors())

policy_full = recipe(policy_content ~ text, data = policy_train) |> 
  step_tokenize(text, token = "ngrams", options = list(n = 2, n_min = 1)) |>
  step_stopwords(text, stopword_source = "snowball") |> 
  step_stem(text) |> 
  step_tokenfilter(text, max_tokens = 1000) |> 
  step_tfidf(text)

policy_full_norm = recipe(policy_content ~ text, data = policy_train) |> 
  step_tokenize(text, token = "ngrams", options = list(n = 2, n_min = 1)) |>
  step_stopwords(text, stopword_source = "snowball") |> 
  step_stem(text) |> 
  step_tokenfilter(text, max_tokens = 1000) |> 
  step_tfidf(text) |> 
  step_normalize(all_predictors())

policy_wf = workflow() |> 
  add_recipe(policy_base) |> 
  add_model(nb_model)

policy_full = workflow() |> 
  add_recipe(policy_full) |> 
  add_model(nb_model)

policy_full_norm = workflow() |> 
  add_recipe(policy_full_norm) |> 
  add_model(nb_model)


# fit ---------------------------------------------------------------------

policy_fit = policy_wf |> 
  fit(data = policy_train)

policy_prediction = predict(policy_fit, policy_train)

policy_metrics = tibble(truth = policy_train$policy_content, estimate = policy_prediction$.pred_class)

accuracy(policy_metrics, truth = truth, estimate = estimate)
conf_mat(policy_metrics, truth = truth, estimate = estimate) |> 
  autoplot(type = "heatmap")
f_meas(policy_metrics, truth = truth, estimate = estimate)



policy_full_fit = policy_full |> 
  fit(policy_train)

policy_full_prediction = predict(policy_full_fit, policy_train)

policy_full_metrics = tibble(truth = policy_train$policy_content, estimate = policy_full_prediction$.pred_class)

accuracy(policy_full_metrics, truth = truth, estimate = estimate)
conf_mat(policy_full_metrics, truth = truth, estimate = estimate) |> 
  autoplot(type = "heatmap")
f_meas(policy_full_metrics, truth = truth, estimate = estimate)



policy_full_norm_fit = policy_full_norm |> 
  fit(policy_train)

policy_full_norm_prediction = predict(policy_full_norm_fit, policy_train)

policy_full_norm_metrics = tibble(truth = policy_train$policy_content, estimate = policy_full_norm_prediction$.pred_class)

accuracy(policy_full_norm_metrics, truth = truth, estimate = estimate)
conf_mat(policy_full_norm_metrics, truth = truth, estimate = estimate) |> 
  autoplot(type = "heatmap")
f_meas(policy_full_norm_metrics, truth = truth, estimate = estimate)


# policy area ----------------------------------------------------------

# set seed for split 
set.seed(755)

area_split = initial_split(data, strata = policy_area)
area_train = training(area_split)
area_test = testing(area_split)


area_base = recipe(policy_area ~ text, data = area_train) |> 
  step_tokenize(text) |> 
  step_tokenfilter(text, max_tokens = 1000) |> 
  step_tfidf(text)

area_full = recipe(policy_area ~ text, data = area_train) |> 
  step_tokenize(text, token = "ngrams", options = list(n = 2, n_min = 1)) |>
  step_stopwords(text, stopword_source = "snowball") |> 
  step_stem(text) |> 
  step_tokenfilter(text, max_tokens = 1000) |> 
  step_tfidf(text)

area_full_norm = recipe(policy_area ~ text, data = area_train) |> 
  step_tokenize(text, token = "ngrams", options = list(n = 2, n_min = 1)) |>
  step_stopwords(text, stopword_source = "snowball") |> 
  step_stem(text) |> 
  step_tokenfilter(text, max_tokens = 1000) |> 
  step_tfidf(text) |> 
  step_normalize(all_predictors())
  
area_wf = workflow() |> 
  add_recipe(area_base) |> 
  add_model(nb_model)

area_full_wf = workflow() |> 
  add_recipe(area_full) |> 
  add_model(nb_model)

area_full_norm_wf = workflow() |> 
  add_recipe(area_full_norm) |> 
  add_model(nb_model)

# fit ---------------------------------------------------------------------

area_fit = area_wf |> 
  fit(data = area_train)

area_prediction = predict(area_fit, area_train)

area_metrics = tibble(truth = as.factor(area_train$policy_area), estimate = area_prediction$.pred_class)

accuracy(area_metrics, truth = truth, estimate = estimate)
f_meas(area_metrics, truth = truth, estimate = estimate)
conf_mat(area_metrics, truth = truth, estimate = estimate) |> 
  autoplot(type = "heatmap")



area_full_fit = area_full_wf |> 
  fit(data = area_train)

area_full_prediction = predict(area_full_fit, area_train)
area_full_metrics = tibble(truth = as.factor(area_train$policy_area), estimate = area_full_prediction$.pred_class)

accuracy(area_full_metrics, truth, estimate)
f_meas(area_full_metrics, truth, estimate)
conf_mat(area_full_metrics, truth, estimate) |> 
  autoplot(type = "heatmap")


area_full_norm_fit = area_full_norm_wf |> 
  fit(data = area_train)

area_full_norm_prediction = predict(area_full_norm_fit, area_train)
area_full_norm_metrics = tibble(truth = as.factor(area_train$policy_area), estimate = area_full_norm_prediction$.pred_class)

accuracy(area_full_norm_metrics, truth, estimate)
f_meas(area_full_norm_metrics, truth, estimate)
conf_mat(area_full_metrics, truth, estimate) |> 
  autoplot(type = "heatmap")
