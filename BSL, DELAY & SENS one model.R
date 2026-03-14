# Només un model per cada periode, té una accuracy decent. 1ª iteració.
# Acabat el 05/03/2026

setwd("C:\\Users\\lefor\\Desktop\\OneDriveBackupFiles\\Documentos\\Q4 MUEI\\TFM\\Base Dataset")

library(dplyr)
library(tidyr)
library(glmnet)
library(caret)
library(ggplot2)
library(tibble)

# ----- 1. LOAD AND PROCESS DATA -----

load("BSL_allChan.RData")
load("DELAY_allChan.RData")
load("SENSORY_allChan.RData")

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

BSL  <- assign_categories(BSL)
SENS <- assign_categories(SENS)
DELAY <- assign_categories(DELAY)

BSL  <- BSL  %>% select(-y)
SENS <- SENS %>% select(-y)
DELAY <- DELAY %>% select(-y)

rm(BSL_allChan, DELAY_allChan, SENSORY_allChan, Times)
gc()

# ----- 2. DATASET DOWNSAMPLING -----

process_dataset <- function(df) {
  
  results_df <- tibble(
    subjectID = numeric(),
    trialID   = numeric(),
    category  = character(),
    EEG       = list())
  
  for (subj in unique(df$subjectID)) {
    
    df_subj <- df %>% filter(subjectID == subj)
    subj_trials <- unique(df_subj$trialID)
    
    for (tr in subj_trials) {
      
      df_trial <- df_subj %>% filter(trialID == tr) %>% arrange(channel)
      
      ch_values <- numeric(64)
      
      for (chn in 1:64) {
        ts_vec <- as.numeric(df_trial[chn, 4][[1]])
        
        ch_values[chn] <- mean(ts_vec)
      }
      
      cat <- df_trial$category[1]
      
      trial_df <- tibble(
        subjectID = subj,
        trialID   = tr,
        category  = cat,
        EEG       = list(ch_values)   # 64 valors
      )
      
      results_df <- bind_rows(results_df, trial_df)
    }
  }
  
  return(results_df)
}

BSL_proc   <- process_dataset(BSL)
SENS_proc  <- process_dataset(SENS)
DELAY_proc <- process_dataset(DELAY)

rm(BSL, SENS, DELAY)
gc()

# ----- 3. MATRIX CREATION -----

make_X_matrix <- function(df_proc) {
  X <- t(sapply(df_proc$EEG, function(x) unlist(x)))
  return(X)
}

make_y_vector <- function(df_proc) df_proc$category

# ----- 4. SELECTION OF THE FIRST 1000 TRIALS -----

select_first_n_trials <- function(df_proc, n = 1000) {
  df_proc <- df_proc %>% arrange(subjectID, trialID)

  selected <- tibble()
  total <- 0

  for (subj in unique(df_proc$subjectID)) {
    df_subj <- df_proc %>% filter(subjectID == subj)
    n_subj <- nrow(df_subj)

    if (total + n_subj <= n) {
      selected <- bind_rows(selected, df_subj)
      total <- total + n_subj
    } else {
      break
    }
  }

  return(selected)
}

BSL_proc_1000   <- select_first_n_trials(BSL_proc,   1000)
SENS_proc_1000  <- select_first_n_trials(SENS_proc,  1000)
DELAY_proc_1000 <- select_first_n_trials(DELAY_proc, 1000)

rm(BSL_proc, SENS_proc, DELAY_proc)
gc()

X_BSL_1000   <- make_X_matrix(BSL_proc_1000)
y_BSL_1000   <- make_y_vector(BSL_proc_1000)

X_SENS_1000  <- make_X_matrix(SENS_proc_1000)
y_SENS_1000  <- make_y_vector(SENS_proc_1000)

X_DELAY_1000 <- make_X_matrix(DELAY_proc_1000)
y_DELAY_1000 <- make_y_vector(DELAY_proc_1000)



# ----- 5. LEAVE-ONE-SUBJECT-OUT CROSS-VALIDATION -----

train_glmnet <- function(X, y) {
  set.seed(2026)
  cv.glmnet(
    x = X,
    y = y,
    family = "multinomial",
    type.measure = "class",
    alpha = 1
  )
}

LOSO <- function(df_proc, X, y) {
  
  subjects <- unique(df_proc$subjectID)
  acc_list <- list()
  
  for (subj in subjects) {
    
    test_idx  <- which(df_proc$subjectID == subj)
    train_idx <- setdiff(1:nrow(df_proc), test_idx)
    
    model <- train_glmnet(X[train_idx, ], y[train_idx])
    
    pred <- predict(model, X[test_idx, ], type = "class")
    
    acc <- mean(pred == y[test_idx])
    
    acc_list[[as.character(subj)]] <- acc
  }
  
  tibble(subjectID = subjects, accuracy = unlist(acc_list))
}

acc_BSL   <- LOSO(BSL_proc_1000,   X_BSL_1000,   y_BSL_1000) %>% 
  mutate(dataset = "BSL")
acc_SENS  <- LOSO(SENS_proc_1000,  X_SENS_1000,  y_SENS_1000) %>% 
  mutate(dataset = "SENS")
acc_DELAY <- LOSO(DELAY_proc_1000, X_DELAY_1000, y_DELAY_1000) %>% 
  mutate(dataset = "DELAY")

df_violin <- bind_rows(acc_BSL, acc_SENS, acc_DELAY)

rm(acc_BSL, acc_SENS, acc_DELAY, BSL_proc_1000, SENS_proc_1000, DELAY_proc_1000)
gc()


# ----- 6. VIOLIN PLOT -----

ggplot(df_violin, aes(x = dataset, y = accuracy, fill = dataset)) +
  geom_violin(trim = FALSE, alpha = 0.6) +
  geom_jitter(width = 0.1, size = 2, alpha = 0.7) +
  stat_summary(fun = mean, geom = "crossbar", width = 0.5,
               fatten = 2, color = "black") +
  stat_summary(fun = median, geom = "point", size = 3, color = "white") +
  geom_hline(yintercept = 1/3, linetype = "dashed", color = "black") +
  theme_minimal(base_size = 14) +
  labs(title = "Classification accuracy per dataset (LOSO, 1000 trials)",
       y = "Accuracy",
       x = "Dataset")