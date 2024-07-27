## test on Congress Tweets 

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
library(jsonlite)


# import ------------------------------------------------------------------

# data
ct = read_rds("~/Desktop/Projects/dissertation/data/congresstweets.rds") |> 
  mutate(policy_area = "None")
ids = users = fromJSON("~/Desktop/Projects/dissertation/data/historical-users-filtered.json")

# model
model = read_rds("02_models/reg_multnom_model")


# wrangle -----------------------------------------------------------------

names = ids |> 
  filter(chamber %in% c("house", "senate")) |>  
  unnest_longer(col = accounts) |> 
  mutate(account_id = accounts$id, 
         screen_name = accounts$screen_name,
         account_type = accounts$account_type,
         prev_names = accounts$prev_names,
         bioguide = id$bioguide,
         gov_track = id$govtrack) |> 
  select(name:party, account_id:gov_track) |> 
  filter(type == "member")


# apply model -------------------------------------------------------------

ct_fit = model |> 
  fit(data = ct)
