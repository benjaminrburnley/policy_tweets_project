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
library(vetiver)


# data --------------------------------------------------------------------

data = read_rds("00_data/data_cleaned_2013.rds")

norts = data |> 
  filter(retweet == 0) |> 
  mutate(name_state = paste(name, state))

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

## base recipe with very few steps
base_recipe = recipe(policy_area ~ text + name_state + party_id + age + party_leader + ideology, data = train) |>
  step_tokenize(text) |>
  step_stopwords(text, stopword_source = "snowball") |>
  step_stem(text) |>
  step_tokenfilter(text, max_tokens = 5000) |>
  step_tfidf(text) |>
  step_dummy(name_state)

base_recipe = recipe(policy_area ~ text , data = train) |>
  step_tokenize(text) |>
  step_stopwords(text, stopword_source = "snowball") |>
  step_stem(text) |>
  step_tokenfilter(text, max_tokens = 5000) |>
  step_tfidf(text)


# create sparse bluepring - see HVITFELDT AND SILGE 2022 
sparse_bp <- default_recipe_blueprint(composition = "dgCMatrix")

multi_tuned <- multinom_reg(penalty = 0.00174 , mixture = 1) %>%
  set_mode("classification") %>%
  set_engine("glmnet")

# workflows ---------------------------------------------------------------

base_wf = workflow() |> 
  add_recipe(base_recipe, blueprint = sparse_bp) |> 
  add_model(multi_tuned)


# run model ---------------------------------------------------------------

fit = base_wf |> 
  fit(train)

preds = predict(fit, train)

output = train |> 
  mutate(truth = as.factor(policy_area)) |> 
  bind_cols(preds) |> 
  rename("estimate" = ".pred_class")

accuracy(output, truth, estimate)
specificity(output, truth, estimate)
recall(output, truth, estimate)
f_meas(output, truth, estimate)

conf_mat(output, truth, estimate) |> 
  autoplot(type = "heatmap")

# visualization -----------------------------------------------------------

policy = output |> 
  group_by(date) |> 
  count(truth)

prediction = output |> 
  group_by(date) |> 
  count(estimate)

ggplot(policy, aes(date, n, color = truth))+
  geom_smooth(se = F)+
  geom_smooth(data = prediction, aes(date, n, color = estimate), linetype = "dashed", se = F)+
  theme_bw()+
  labs(
    title = "Predicted vs. Coded Policy Content - Training Set",
    subtitle = "Dashed line shows predicted counts",
    color = NULL
  )+
  theme(
    legend.position = "bottom"
  )

ggsave("figures/2013_preds_vs_policy.png")



# cross validation --------------------------------------------------------

# tuned_cv = fit_resamples(
#   base_wf, 
#   folds,
#   metrics = metric_set(accuracy, recall, precision, specificity, f_meas),
#   control = control_resamples(verbose = TRUE)
# )
# 
# collect_metrics(tuned_cv)


# finalize workflow -------------------------------------------------------

fitted = last_fit(base_wf, split)

final_metrics = collect_metrics(fitted)
final_preds = collect_predictions(fitted)
reg_multnom_model = extract_workflow(fitted)

collect_predictions(fitted) %>%
  conf_mat(truth = policy_area, estimate = .pred_class) %>%
  autoplot(type = "heatmap")

ggsave("figures/2013_train_confmat.png")

collect_predictions(fitted) |> 
  roc_curve(truth = policy_area, `.pred_Domestic Policy`:`.pred_Social Policy`) %>%
  autoplot() +
  labs(
    color = NULL,
    title = "ROC curve for Policy Topics",
    subtitle = "With final tuned lasso regularized classifier on the test set"
  )

ggsave("figures/roc_curves.png")

test_output = test |>
  mutate(truth = as.factor(policy_area)) |> 
  bind_cols(final_preds) |> 
  rename("estimate" = ".pred_class")

accuracy(test_output, truth, estimate)
recall(test_output, truth, estimate)
specificity(test_output, truth, estimate)
f_meas(test_output, truth, estimate)

test_policy = test_output |> 
  group_by(date) |> 
  count(truth)

test_prediction = test_output |> 
  group_by(date) |> 
  count(estimate)

ggplot(test_policy, aes(date, n, color = truth))+
  geom_smooth(se = F)+
  geom_smooth(data = test_prediction, aes(date, n, color = estimate), linetype = "dashed", se = F)+
  theme_bw()+
  labs(
    title = "Predicted vs. Coded Policy Content - Test Set",
    subtitle = "Dashed line shows predicted counts",
    color = NULL
  )+
  theme(
    legend.position = "bottom"
  )

ggsave("figures/preds_v_policy_test.png")



# 2015 sample test --------------------------------------------------------

data_15 = read_rds("00_data/data_cleaned_2015.rds")

norts_15 = data_15 |> 
  filter(retweet == 0) |> 
  mutate(name_state = paste(name, state))

fit_15 = reg_multnom_model |> 
  fit(data = norts_15)

preds_15 = predict(fit_15, norts_15)

valid_output = norts_15 |>
  mutate(truth = as.factor(policy_area)) |> 
  bind_cols(preds_15) |> 
  rename("estimate" = ".pred_class")

accuracy(valid_output, truth, estimate)
recall(valid_output, truth, estimate)
specificity(valid_output, truth, estimate)
f_meas(valid_output, truth, estimate, estimator = "micro")

conf_mat(valid_output, truth, estimate) |> 
  autoplot(type = "heatmap")

ggsave("figures/2015_confmat.png")

valid_policy = valid_output |> 
  group_by(date) |> 
  count(truth)

valid_prediction = valid_output |> 
  group_by(date) |> 
  count(estimate)

ggplot(valid_policy, aes(date, n, color = truth))+
  geom_smooth(se = F)+
  geom_smooth(data = valid_prediction, aes(date, n, color = estimate), linetype = "dashed", se = F)+
  theme_bw()+
  labs(
    title = "Predicted vs. Coded Policy Content - 2015 data",
    subtitle = "Dashed line shows predicted counts",
    color = NULL
  )+
  theme(
    legend.position = "bottom"
  )

ggsave("figures/2015_validation.png")



## save this model for application 
write_rds(reg_multnom_model, "02_models/reg_multnom_model")
