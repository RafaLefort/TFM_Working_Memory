# Exporatory analysis
# Finished on the 12/03/2026 

setwd("C:\\Users\\lefor\\Desktop\\OneDriveBackupFiles\\Documentos\\Q4 MUEI\\TFM\\Base Dataset")

# Loading the needed libraries
library(dplyr)
library(tidyr)
library(glmnet)
library(caret)
library(ggplot2)
library(tibble)
library(doParallel)

# ----- 1. LOADING DATA -----

load("BSL_allChan.RData")
load("DELAY_allChan.RData")
load("SENSORY_allChan.RData")

# Function that assigns written categories for easy interpretation
assign_categories <- function(df){
  df %>%
    group_by(subjectID, trialID) %>%
    mutate(category = case_when(
      y == 1 ~ "visual",
      y == 3 ~ "spatial",
      y == 5 ~ "verbal",
      TRUE ~ NA_character_
    )) %>%
    ungroup()
}

BSL   <- assign_categories(BSL)
SENS  <- assign_categories(SENS)
DELAY <- assign_categories(DELAY)

# Deleting the repeated column
BSL   <- BSL   %>% select(-y)
SENS  <- SENS  %>% select(-y)
DELAY <- DELAY %>% select(-y)

# ----- 2. EXPLORATORY ANALYSIS -----

