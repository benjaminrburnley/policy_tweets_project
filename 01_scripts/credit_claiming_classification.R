### Credit Claiming Classification Model v1 
## April 3, 2023


# libraries ---------------------------------------------------------------

library(tidyverse)
library(haven)
library(tidymodels)
library(textrecipes)
library(discrim)
library(naivebayes)
library(glmnet)
library(hardhat)
library(vip)


# data --------------------------------------------------------------------

data = read_dta("00_data/2013dataPSU.dta") |> 
  janitor::clean_names()


# wrangle -----------------------------------------------------------------

# reclassify outcome variable 

data_cc = data |> 
  mutate(cc = if_else(credit_claim == 1, "Credit Claiming", "Other"))


# preprocessing -----------------------------------------------------------

set.seed(0701)

# create initial split in data 
cc_split = initial_split(data_cc, strata = cc)

# create training set
cc_train = training(cc_split)

# create testing set 
cc_test = testing(cc_split)

# create recipe 
cc_rec = recipe(cc ~ text, data = cc_train) |> 
  step_tokenize(text) |> 
  step_tokenfilter(text, max_tokens = 1000) |> 
  step_tfidf(text)

# create workflow 
cc_wf = workflow() |> 
  add_recipe(cc_rec)

# null model --------------------------------------------------------------

null_classification <- null_model() %>%
  set_engine("parsnip") %>%
  set_mode("classification")

null_rs <- workflow() %>%
  add_recipe(cc_rec) %>%
  add_model(null_classification) %>%
  fit_resamples(
    cc_folds
  )

collect_metrics(null_rs)


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
  control = control_resamples(save_pred = T,
                              verbose = T)
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



# lasso specification - no tuning -----------------------------------------------------

lasso_spec <- logistic_reg(penalty = 0.01, mixture = 1) %>%
  set_mode("classification") %>%
  set_engine("glmnet")

lasso_wf <- workflow() %>%
  add_recipe(cc_rec) %>%
  add_model(lasso_spec)

set.seed(2020)
lasso_rs <- fit_resamples(
  lasso_wf,
  cc_folds,
  control = control_resamples(save_pred = T,
                              verbose = T)
)

lasso_rs_metrics <- collect_metrics(lasso_rs)
lasso_rs_predictions <- collect_predictions(lasso_rs)

lasso_rs_predictions %>%
  group_by(id) %>%
  roc_curve(truth = cc, `.pred_Credit Claiming`) %>%
  autoplot() +
  labs(
    color = NULL,
    title = "ROC curve for Credit Claiming in Tweets",
    subtitle = "Each resample fold is shown in a different color"
  )

conf_mat_resampled(lasso_rs, tidy = FALSE) %>%
  autoplot(type = "heatmap")



# lasso specification - tuning parameters ---------------------------------

tune_spec <- logistic_reg(penalty = tune(), mixture = 1) %>%
  set_mode("classification") %>%
  set_engine("glmnet")

lambda_grid <- grid_regular(penalty(), levels = 30)

tune_wf <- workflow() %>%
  add_recipe(cc_rec) %>%
  add_model(tune_spec)

set.seed(2020)
tune_rs <- tune_grid(
  tune_wf,
  cc_folds,
  grid = lambda_grid,
  control = control_resamples(save_pred = TRUE,
                              verbose = TRUE)
)


autoplot(tune_rs) +
  labs(
    title = "Lasso model performance across regularization penalties",
    subtitle = "Performance metrics can be used to identity the best penalty"
  )

tune_rs %>%
  show_best(metric = "roc_auc")

chosen_auc <- tune_rs %>%
  select_by_one_std_err(metric = "roc_auc", -penalty)

final_lasso <- finalize_workflow(tune_wf, chosen_auc)

fitted_lasso <- fit(final_lasso, cc_train)

fitted_lasso %>%
  extract_fit_parsnip() %>%
  tidy() %>%
  arrange(-estimate)

fitted_lasso %>%
  extract_fit_parsnip() %>%
  tidy() %>%
  arrange(estimate)


# sparse encoding ----------------------------------------------------

sparse_bp <- default_recipe_blueprint(composition = "dgCMatrix")

sparse_wf <- workflow() %>%
  add_recipe(cc_rec, blueprint = sparse_bp) %>%
  add_model(tune_spec)

smaller_lambda <- grid_regular(penalty(range = c(-5, 0)), levels = 20)

set.seed(2020)
sparse_rs <- tune_grid(
  sparse_wf,
  cc_folds,
  grid = smaller_lambda,
  control = control_resamples(save_pred = T,
                              verbose = T)
)

sparse_rs %>%
  show_best(metric = "roc_auc")

# adding controls  --------------------------------------------------------

# create recipe 
cc_vars_rec = recipe(cc ~ text + date + com_leader + party_id + party_leader + total_output + gender + seniority + ideology + electoral_performance + age,
                     data = cc_train) |> 
  step_tokenize(text) |> 
  step_tokenfilter(text, max_tokens = 1000) |> 
  step_tfidf(text) |> 
  step_date(date, features = c("month", "dow"), role = "dates") |> 
  step_rm(date) |> 
  step_dummy(has_role("dates")) |> 
  step_normalize(all_predictors())

more_vars_wf <- workflow() %>%
  add_recipe(cc_vars_rec, blueprint = sparse_bp) %>%
  add_model(tune_spec)

set.seed(123)
more_vars_rs <- tune_grid(
  more_vars_wf,
  cc_folds,
  grid = smaller_lambda,
  control = control_resamples(save_pred = T,
                              verbose = T)
)

more_vars_rs %>%
  show_best(metric = "accuracy")

finalize_workflow(more_vars_wf, 
                  select_best(more_vars_rs, metric = "roc_auc")) %>%
  fit(cc_train) %>%
  extract_fit_parsnip() %>%
  tidy() %>% 
  arrange(-abs(estimate)) %>% 
  mutate(term_rank = row_number()) %>% 
  filter(!str_detect(term, "tfidf"))


# examine other metrics ---------------------------------------------------

check <- fit_resamples(
  lasso_wf,
  cc_folds,
  metrics = metric_set(recall, precision),
  control = control_resamples(verbose = T)
)



# finalize model ----------------------------------------------------------

cc_rec_2 = recipe(cc ~ text, data = cc_train) |> 
  step_tokenize(text, token = "ngrams", options = list(n = 2, n_min = 1)) |>
  step_stopwords(text, stopword_source = "snowball") |> 
  step_stem(text) |> 
  step_tokenfilter(text,
                   max_tokens = tune(), min_times = 100) |> 
  step_tfidf(text)

sparse_wf_v2 <- sparse_wf %>%
  update_recipe(cc_rec_2, blueprint = sparse_bp)


final_grid <- grid_regular(
  penalty(range = c(-4, 0)),
  max_tokens(range = c(1e3, 3e3)),
  levels = c(penalty = 20, max_tokens = 3)
)

set.seed(2020)

tune_rs <- tune_grid(
  sparse_wf_v2,
  cc_folds,
  grid = final_grid,
  metrics = metric_set(accuracy, sensitivity, specificity),
  control = control_resamples(verbose = T)
)

autoplot(tune_rs) +
  labs(
    color = "Number of tokens",
    title = "Model performance across regularization penalties and tokens",
    subtitle = paste("We can choose a simpler model with higher regularization")
  )

choose_acc <- tune_rs %>%
  select_by_pct_loss(metric = "accuracy")

final_wf <- finalize_workflow(sparse_wf_v2, choose_acc)

final_fitted <- last_fit(final_wf, cc_split)

collect_metrics(final_fitted)

collect_predictions(final_fitted) %>%
  conf_mat(truth = cc, estimate = .pred_class) %>%
  autoplot(type = "heatmap")

collect_predictions(final_fitted)  %>%
  roc_curve(truth = cc, `.pred_Credit Claiming`) %>%
  autoplot() +
  labs(
    color = NULL,
    title = "ROC curve for US Consumer Finance Complaints",
    subtitle = "With final tuned lasso regularized classifier on the test set"
  )


cc_imp <- extract_fit_parsnip(final_fitted$.workflow[[1]]) %>%
  vip::vi(lambda = choose_acc$penalty)

cc_imp %>%
  mutate(
    Sign = case_when(Sign == "POS" ~ "Less about credit claiming",
                     Sign == "NEG" ~ "More about credit claimin"),
    Importance = abs(Importance),
    Variable = str_remove_all(Variable, "tfidf_text_"),
    Variable = str_remove_all(Variable, "textfeature_text_copy_")
  ) %>%
  group_by(Sign) %>%
  dplyr::top_n(20, Importance) |> 
  ungroup %>%
  ggplot(aes(x = Importance,
             y = fct_reorder(Variable, Importance),
             fill = Sign)) +
  geom_col(show.legend = FALSE) +
  scale_x_continuous(expand = c(0, 0)) +
  facet_wrap(~Sign, scales = "free") +
  labs(
    y = NULL,
    title = "Variable importance for predicting the topic of a CFPB complaint",
    subtitle = paste0("These features are the most important in predicting\n",
                      "whether a complaint is about credit or not")
  )

ccs_bind <- collect_predictions(final_fitted) %>%
  bind_cols(cc_test %>% select(-cc))

ccs_bind %>%
  filter(cc == "Credit Claiming", `.pred_Credit Claiming` < 0.2) %>%
  select(text) %>%
  slice_sample(n = 10)


