### Test of Regression Approach 


# libraries ---------------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(haven)
library(textrecipes)


# import and wrangle  --------------------------------------------------------------------

data = read_dta("00_data/2013dataPSU.dta") |> 
  janitor::clean_names()



# eda ---------------------------------------------------------------------

data |> 
  distinct(name, ideology) |> 
  ggplot(aes(ideology))+
  geom_histogram()+
  theme_bw()


# preprocessing -----------------------------------------------------------

set.seed(0701)

# create split
split = initial_split(data)  

train = training(split)
test = testing(split)

# set recipe for preprocessing
recipe = recipe(ideology ~ text, data = train) |> 
  step_tokenize(text, token = "ngrams", options = list(n = 2, n_min = 1)) |> 
  step_stem(text) |> 
  step_stopwords(text, stopword_source = "snowball") |> 
  step_tokenfilter(text, max_tokens = tune()) |> 
  step_tfidf(text) |> 
  step_normalize(all_predictors())
  
# set support vector machine model 
svm_spec = svm_linear() |> 
    set_mode("regression") |> 
    set_engine("LiblineaR")

# create workflow 
workflow = workflow() |> 
  add_recipe(recipe) |> 
  add_model(svm_spec)

# create tuning grid 
final_grid <- grid_regular(
  max_tokens(range = c(1e3, 6e3)),
  levels = 6
)

# create cross validation folds 
folds = vfold_cv(train)

# tune grid  --------------------------------------------------------------

rs <- tune_grid(
  workflow,
  folds,
  grid = final_grid,
  metrics = metric_set(rmse, mae, mape),
  control = control_resamples(save_pred = TRUE,
                              verbose = T)
)

