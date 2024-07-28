####
### build congress data sets
##
#


# libraries  --------------------------------------------------------------

library(tidyverse)


# data --------------------------------------------------------------------

ct = read_rds("~/Desktop/Projects/dissertation/data/congresstweets.rds")
ids = read_rds("~/Desktop/Projects/dissertation/data/congress_tweets_user_info.rds")
dwnom = read_csv("00_data/dw_all.csv")


# wrangle -----------------------------------------------------------------

# join bio info
ct_ids = ct |> 
  left_join(ids, by = "screen_name")

# filter by senate 
ct_sen = ct_ids |> 
  filter(chamber == "senate")

# remove larger dataset 
rm(list = c("ct", "ct_ids"))

# check dwnom for duplicates
dwnom_distinct = dwnom |> 
  distinct(bioguide_id, congress, icpsr, nominate_dim1, party_code, state_abbrev) |> 
  filter(congress > 114) |> 
  filter(!(party_code == 328 & bioguide_id == "M001183")) # filter out Manchin becoming independent

# create time and congress variables 
data = ct_sen |>
  mutate(time = lubridate::as_datetime(time),
         year = year(time),
         congress = case_when(
           year == 2017 ~ 115,
           year == 2018 ~ 115,
           year == 2019 ~ 116,
           year == 2020 ~ 116,
           year == 2021 ~ 117,
           year == 2022 ~ 117,
           year == 2023 ~ 118,
           TRUE ~ NA
         )
  ) |> 
  left_join(dwnom_distinct, by = c("bioguide" = "bioguide_id", "congress")) |> 
  filter(!is.na(icpsr)) # checked: this filters out tweets that were collected *after* a MC left congress


## check missingness 
# Function to calculate the percentage of missing values in a column
percentage_missing <- function(column) {
  if (!is.vector(column)) {
    stop("Input must be a column (vector).")
  }
  
  # Calculate the number of missing values
  num_missing <- sum(is.na(column))
  
  # Calculate the total number of values
  total_values <- length(column)
  
  # Calculate the percentage of missing values
  percentage <- (num_missing / total_values) * 100
  
  return(percentage)
}

for(cols in colnames(data)){
  print(cols)
  print(sum(is.na(data[[cols]]))/length(data[[cols]]))
}


