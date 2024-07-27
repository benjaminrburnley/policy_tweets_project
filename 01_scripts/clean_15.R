
library(tidyverse)
library(janitor)
library(haven)

# data --------------------------------------------------------------------

data = read_dta("00_data/2015forPSU.dta") |> 
  janitor::clean_names() |> 
  mutate(major_topic = case_when(
    major_code == 0 ~ "None",
    major_code == 1 ~ "Economy",
    major_code == 2 ~ "Civil Rights",
    major_code == 3 ~ "Health",
    major_code == 4 ~ "Agriculture",
    major_code == 5 ~ "Labor",
    major_code == 6 ~ "Education",
    major_code == 7 ~ "Environment",
    major_code == 8 ~ "Energy",
    major_code == 9 ~ "Immigration",
    major_code == 10 ~ "Transportation",
    major_code == 12 ~ "Law",
    major_code == 13 ~ "Social Welfare",
    major_code == 14 ~ "Communities",
    major_code == 15 ~ "Finance",
    major_code == 16 ~ "Defense",
    major_code == 17 ~ "Science",
    major_code == 18 ~ "Trade",
    major_code == 19 ~ "International",
    major_code == 20 ~ "Operations",
    major_code == 21 ~ "Public Lands",
    major_code == 22 ~ "Other",
  )) |> 
  mutate(major_topic = if_else(policy_content == 1 & major_topic == "None", "Other", major_topic)) |> 
  mutate(policy_area = case_when(
    major_code == 0 ~ "None",
    major_code == 1 ~ "Economic Policy",
    major_code == 2 ~ "Social Policy",
    major_code == 3 ~ "Social Policy",
    major_code == 4 ~ "Domestic Policy",
    major_code == 5 ~ "Economic Policy",
    major_code == 6 ~ "Domestic Policy",
    major_code == 7 ~ "Domestic Policy",
    major_code == 8 ~ "Domestic Policy",
    major_code == 9 ~ "Social Policy",
    major_code == 10 ~ "Domestic Policy",
    major_code == 12 ~ "Social Policy",
    major_code == 13 ~ "Social Policy",
    major_code == 14 ~ "Social Policy",
    major_code == 15 ~ "Economic Policy",
    major_code == 16 ~ "Foreign Policy",
    major_code == 17 ~ "Domestic Policy",
    major_code == 18 ~ "Foreign Policy",
    major_code == 19 ~ "Foreign Policy",
    major_code == 20 ~ "Domestic Policy",
    major_code == 21 ~ "Domestic Policy",
    major_code == 22 ~ "Other",
  ))

# clean text for analysis 
data_cleaned = data |> 
  mutate(retweet = if_else(str_detect(pattern = "^RT", string = text), 1, 0), # create retweet binary
         text = str_to_lower(text), # to lower case
         text = str_remove(pattern = "[:punct:]", string = text))

write_rds(data_cleaned, "00_data/data_cleaned_2015.rds")
