### Multinomial Regularization Approach 



# libraries  --------------------------------------------------------------

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


# data --------------------------------------------------------------------
data = read_rds("00_data/data_cleaned_2013.rds")

norts = data |> 
  filter(retweet == 0)


# create splits -----------------------------------------------------------

# create split
set.seed(0701)
split = initial_split(norts, strata = major_topic)

# training set
train = training(split)

# testing set 
test = testing(split)


# recipes -----------------------------------------------------------------

# base recipe with very few steps
base_recipe = recipe(major_topic ~ text, data = train) |> 
  step_tokenize(text) |> 
  step_tokenfilter(text, max_tokens = 3000) |> 
  step_tfidf(text)

# add ngrams, stop word removal, stemming 
full_recipe = recipe(major_topic ~ text, data = train) |> 
  step_tokenize(text, token = "ngrams", options = list(n = 2, n_min = 1)) |>
  step_stopwords(text, stopword_source = "snowball") |> 
  step_stem(text) |> 
  step_tokenfilter(text, max_tokens = 1000) |> 
  step_tfidf(text)

# same as full, but with down sampling on topic of tweet
downsample_recipe = recipe(major_topic ~ text, data = train) |> 
  step_tokenize(text, token = "ngrams", options = list(n = 2, n_min = 1)) |>
  step_stopwords(text, stopword_source = "snowball") |> 
  step_stem(text) |> 
  step_tokenfilter(text, max_tokens = 1000) |> 
  step_tfidf(text) |> 
  step_downsample(major_topic)

# create sparse bluepring - see HVITFELDT AND SILGE 2022 
sparse_bp <- default_recipe_blueprint(composition = "dgCMatrix")


# model specifications ----------------------------------------------------
multi_no_tune <- multinom_reg(penalty = 0.01, mixture = 1) %>%
  set_mode("classification") %>%
  set_engine("glmnet")

# workflows ---------------------------------------------------------------

base_wf = workflow() |> 
  add_recipe(base_recipe, blueprint = sparse_bp) |> 
  add_model(multi_no_tune)

full_wf = workflow() |> 
  add_recipe(full_recipe, blueprint = sparse_bp) |> 
  add_model(multi_no_tune)

downsample_wf = workflow() |> 
  add_recipe(downsample_recipe, blueprint = sparse_bp) |> 
  add_model(multi_no_tune)



# non-tuned fits ----------------------------------------------------------

# fit base 
mutli_base_fit = base_wf |> 
  fit(data = train)

# fit full 
multi_full_fit = full_wf |> 
  fit(data = train)

# fit downsampled 
multi_ds_fit = downsample_wf |> 
  fit(data = train)

### metrics 
mbf_preds = predict(mutli_base_fit, train)
mff_preds = predict(multi_full_fit, train) 
mdf_preds = predict(multi_ds_fit, train)

preds1 = tibble(truth = as.factor(train$major_topic), predicted = mbf_preds$.pred_class)
preds2 = tibble(truth = as.factor(train$major_topic), predicted = mff_preds$.pred_class)
preds3 = tibble(truth = as.factor(train$major_topic), predicted = mdf_preds$.pred_class)

# MODEL 10
accuracy(preds1, truth, predicted)
precision(preds1, truth = truth, estimate = predicted)
specificity(preds1, truth, predicted)
recall(preds1, truth, predicted)
f_meas(preds1, truth, predicted)

# MODEL 11
accuracy(preds2, truth, predicted)
precision(preds2, truth = truth, estimate = predicted)
specificity(preds2, truth, predicted)
recall(preds2, truth, predicted)
f_meas(preds2, truth, predicted)

# MODEL 12
accuracy(preds3, truth, predicted)
precision(preds3, truth = truth, estimate = predicted)
specificity(preds3, truth, predicted)
recall(preds3, truth, predicted)
f_meas(preds3, truth, predicted)

conf_mat(preds1, truth = truth, estimate = predicted) |> 
  autoplot(type = "heatmap")

conf_mat(preds2,
         truth = truth,
         estimate = predicted) |> 
  autoplot(type = "heatmap")


# try with additional variables  ------------------------------------------

full_recipe_cons = recipe(major_topic ~ text + name + ideology, data = train) |> 
  step_tokenize(text, token = "ngrams", options = list(n = 2, n_min = 1)) |>
  step_stopwords(text, stopword_source = "snowball") |> 
  step_stem(text) |> 
  step_tokenfilter(text, max_tokens = 1000) |> 
  step_tfidf(text) |> 
  step_dummy(name)

full_cons_wf = workflow() |> 
  add_recipe(full_recipe_cons, blueprint = sparse_bp) |> 
  add_model(multi_no_tune)

full_cons_fit = full_cons_wf |> 
  fit(data = train)

fcf_preds = predict(full_cons_fit, train) 
preds4 = tibble(truth = as.factor(train$major_topic), predicted = fcf_preds$.pred_class)

conf_mat(preds4,truth = truth, estimate = predicted)
accuracy(preds4, truth, predicted)
precision(preds4, truth = truth, estimate = predicted)
specificity(preds4, truth, predicted)
recall(preds4, truth, predicted)
f_meas(preds4, truth, predicted)


# tuning with cross validation --------------------------------------------

tuned_multi =multinom_reg(penalty = tune(), mixture = 1) %>%
  set_mode("classification") %>%
  set_engine("glmnet")



