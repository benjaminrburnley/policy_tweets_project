

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

# create split
set.seed(0701)
split = initial_split(norts, strata = policy_area)

# training set
train = training(split)

# testing set 
test = testing(split)

# create folds 
folds = vfold_cv(train)

# preprocessing -----------------------------------------------------------

# base recipe with very few steps
base_recipe = recipe(policy_area ~ text, data = train) |> 
  step_tokenize(text) |> 
  step_stopwords(text, stopword_source = "snowball") |>
  step_stem(text) |> 
  step_tokenfilter(text, max_tokens = 5000) |> 
  step_tfidf(text)

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

# fit ---------------------------------------------------------------------

# fit base 
mutli_base_fit = base_wf |> 
  fit(data = train)


# evaluate ----------------------------------------------------------------

### metrics 
mbf_preds = predict(mutli_base_fit, train)

preds1 = tibble(truth = as.factor(train$policy_area), predicted = mbf_preds$.pred_class)

accuracy(preds1, truth, predicted)
precision(preds1, truth = truth, estimate = predicted)
specificity(preds1, truth, predicted)
recall(preds1, truth, predicted)
f_meas(preds1, truth, predicted)

conf_mat(preds1, truth, predicted) |> 
  autoplot(type = "heatmap")

# validation - no tuning  -------------------------------------------------

set.seed(2020)
lasso_rs <- fit_resamples(
  base_wf,
  folds,
  control = control_resamples(save_pred = TRUE, verbose = TRUE)
)

base_metrics = collect_metrics(lasso_rs)
base_predictions = collect_predictions(lasso_rs)

base_metrics

base_predictions %>%
  group_by(id) %>%
  roc_curve(truth = policy_area, `.pred_Domestic Policy`:`.pred_Social Policy`) %>%
  autoplot() +
  labs(
    color = NULL,
    title = "ROC curve for Policy Topic",
    subtitle = "Each resample fold is shown in a different color"
  )


# tuning  -----------------------------------------------------------------

# set tuning specification 
multi_tune <- multinom_reg(penalty = tune(), mixture = 1) %>%
  set_mode("classification") %>%
  set_engine("glmnet")

# create grid 
lambda_grid <- grid_regular(penalty(), levels = 30)

# tune workflow 
tune_wf <- workflow() %>%
  add_recipe(base_recipe) %>%
  add_model(multi_tune)

# set.seed(2020)
# tune_rs <- tune_grid(
#   tune_wf,
#   folds,
#   grid = lambda_grid,
#   control = control_resamples(save_pred = TRUE, verbose = TRUE)
# )

tune_rs


collect_metrics(tune_rs)
autoplot(tune_rs)

show_best(tune_rs, metric = "roc_auc")


# check tuned -------------------------------------------------------------

multi_tuned <- multinom_reg(penalty = 0.00174 , mixture = 1) %>%
  set_mode("classification") %>%
  set_engine("glmnet")

# tune workflow 
start = Sys.time()
tuned_wf <- workflow() %>%
  add_recipe(base_recipe) %>%
  add_model(multi_tuned)

tuned_fit = tuned_wf |> 
  fit(train)

tuned_predictions = predict(tuned_fit, train)

tuned_preds = tibble(truth = as.factor(train$policy_area), predicted = tuned_predictions$.pred_class)

accuracy(tuned_preds, truth, predicted)
precision(tuned_preds, truth = truth, estimate = predicted)
specificity(tuned_preds, truth, predicted)
recall(tuned_preds, truth, predicted)
f_meas(tuned_preds, truth, predicted)

conf_mat(tuned_preds, truth = truth, estimate = predicted) |> 
  autoplot(type = "heatmap")
end = Sys.time()

diff = end - start
diff
tuned_predictions %>%
  group_by(id) %>%
  roc_curve(truth = policy_area, `.pred_Domestic Policy`:`.pred_Social Policy`) %>%
  autoplot() +
  labs(
    color = NULL,
    title = "ROC curve for Policy Topic",
    subtitle = "Each resample fold is shown in a different color"
  )



# cross validate ----------------------------------------------------------

tuned_cv = fit_resamples(
  tuned_wf, 
  folds,
  metrics = metric_set(accuracy, recall, precision, specificity, f_meas),
  control = control_resamples(verbose = TRUE)
)

collect_metrics(tuned_cv)

# quick plot --------------------------------------------------------------

plot = train |> 
  bind_cols(mbf_preds)

actual = plot |> 
  group_by(date) |> 
  count(policy_area)

predicted = plot |> 
  group_by(date) |> 
  count(.pred_class)

counts = actual |> 
  left_join(predicted, by = c("date", "policy_area" = ".pred_class")) |> 
  mutate(n.y = if_else(is.na(n.y), 0, n.y))

ggplot(counts, aes(y = n.x, x = date, color = policy_area))+
  geom_smooth(se = F)+
  geom_smooth(aes(y = n.y), linetype = "dashed", se = F)+
  theme_bw()+
  labs(
    title = "Classified vs. Predicted Policy Topic",
    subtitle = "Predicted policy topic shown by dashed line",
    x = "Date",
    y = "Daily Tweets About Topic",
    color = NULL
  )+
  theme(
    plot.title = element_text(face = "bold")
  )


# finalize workflow -------------------------------------------------------

finalize_workflow()
