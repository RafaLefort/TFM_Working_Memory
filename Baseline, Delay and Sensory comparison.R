# Baseline, Delay and Sensory comparison
# 16/02/2026

setwd("C:\\Users\\lefor\\Desktop\\OneDriveBackupFiles\\Documentos\\Q4 MUEI\\TFM\\Base Dataset")

library(dplyr)
library(tidyr)
library(glmnet)
library(caret)
library(ggplot2)
library(tibble)

# ----- 1. CARREGAR DADES -----

load("BSL_allChan.RData")
load("DELAY_allChan.RData")
load("SENSORY_allChan.RData")

t_BSL  <- Times[Times >= -300 & Times < 0 ]
t_SENS <- Times[Times >= 0 & Times < 1000 ]
t_DELAY <- Times[Times >= 1000 & Times < 2000]


# ----- 2. ASSIGNAR CATEGORIES -----

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

# ----- 3. ELIMINAR COLUMNA y -----

BSL  <- BSL  %>% select(-y)
SENS <- SENS %>% select(-y)
DELAY <- DELAY %>% select(-y)


# ----- 4. FUNCIÓ PER PROCESSAR DATASET -----

process_dataset <- function(df) {
  
  results_df <- tibble(
    subjectID = numeric(),
    trialID   = numeric(),
    category  = character(),
    EEG       = list()
  )
  
  for (subj in unique(df$subjectID)) {
    
    df_subj <- df %>% filter(subjectID == subj)
    subj_trials <- unique(df_subj$trialID)
    
    for (tr in subj_trials) {
      
      df_trial <- df_subj %>% filter(trialID == tr) %>% arrange(channel)
      
      # Llista on concatenarem tots els canals
      lst_channels <- list()
      
      for (chn in 1:64) {
        
        # extreure la sèrie temporal del canal
        ts_vec <- as.numeric(df_trial[chn, 4][[1]])
        
        # subsampling cada 5 punts
        ts_sub <- ts_vec[seq(1, length(ts_vec), by = 5)]
        
        # afegir a la llista
        lst_channels <- append(lst_channels, as.list(ts_sub))
      }
      
      # categoria del trial
      cat <- df_trial$category[1]
      
      # afegir fila al tibble
      trial_df <- tibble(
        subjectID = subj,
        trialID   = tr,
        category  = cat,
        EEG       = list(unlist(lst_channels))
      )
      
      results_df <- bind_rows(results_df, trial_df)
    }
  }
  
  return(results_df)
}


# ----- 5. PROCESSAR ELS TRES DATASETS -----

BSL_proc   <- process_dataset(BSL)     # 64 × 30 = 1920 features
SENS_proc  <- process_dataset(SENS)    # 64 × 100 = 6400 features
DELAY_proc <- process_dataset(DELAY)   # 64 × 100 = 6400 features


# ----- 6. CONSTRUIR X I y -----

make_X_matrix <- function(df_proc) {
  X <- t(sapply(df_proc$EEG, function(x) unlist(x)))
  return(X)
}

make_y_vector <- function(df_proc) {
  return(df_proc$category)
}

X_BSL   <- make_X_matrix(BSL_proc)
y_BSL   <- make_y_vector(BSL_proc)

X_SENS  <- make_X_matrix(SENS_proc)
y_SENS  <- make_y_vector(SENS_proc)

X_DELAY <- make_X_matrix(DELAY_proc)
y_DELAY <- make_y_vector(DELAY_proc)


# ----- 7. ENTRENAR MODELS GLMNET -----

train_glmnet <- function(X, y) {
  model <- cv.glmnet(
    x = X,
    y = y,
    family = "multinomial",
    type.measure = "class",
    alpha = 1)
  return(model)
}

model_BSL   <- train_glmnet(X_BSL, y_BSL)
model_SENS  <- train_glmnet(X_SENS, y_SENS)
model_DELAY <- train_glmnet(X_DELAY, y_DELAY)


# ----- 8. PREDICCIONS -----

pred_BSL   <- predict(model_BSL,   X_BSL,   type = "class")
pred_SENS  <- predict(model_SENS,  X_SENS,  type = "class")
pred_DELAY <- predict(model_DELAY, X_DELAY, type = "class")


# ----- 9. DATAFRAME PER VIOLIN PLOT -----

df_violin <- tibble(
  dataset = c(rep("BSL", length(pred_BSL)),
              rep("SENS", length(pred_SENS)),
              rep("DELAY", length(pred_DELAY))),
  predicted = c(pred_BSL, pred_SENS, pred_DELAY)
)


# ----- 10. VIOLIN PLOT -----

ggplot(df_violin, aes(x = dataset, y = predicted, fill = dataset)) +
  geom_violin(trim = FALSE) +
  theme_minimal() +
  labs(title = "Distribució de prediccions per dataset",
       y = "Categoria predita")







# ----- 4. EXPANDIR LA MATRIU EEG -----

BSL <- BSL %>% 
  bind_cols(as.data.frame(BSL$EEG)) %>% 
  select(-EEG)

SENS <- SENS %>% 
  bind_cols(as.data.frame(SENS$EEG)) %>% 
  select(-EEG)

DELAY <- DELAY %>% 
  bind_cols(as.data.frame(DELAY$EEG)) %>% 
  select(-EEG)


# ----- 5. FUNCIÓ PER CREAR 64 COLUMNES (C1...C64) -----

make_list_dataset <- function(df, n_timepoints){
  
  df %>%
    group_by(subjectID, trialID) %>%
    arrange(channel) %>%  # assegurar ordre consistent dels canals
    summarise(
      y = first(category),
      C1  = list(as.numeric(unlist(cur_data()[1,  paste0("V", 1:n_timepoints)]))),
      C2  = list(as.numeric(unlist(cur_data()[2,  paste0("V", 1:n_timepoints)]))),
      C3  = list(as.numeric(unlist(cur_data()[3,  paste0("V", 1:n_timepoints)]))),
      C4  = list(as.numeric(unlist(cur_data()[4,  paste0("V", 1:n_timepoints)]))),
      C5  = list(as.numeric(unlist(cur_data()[5,  paste0("V", 1:n_timepoints)]))),
      C6  = list(as.numeric(unlist(cur_data()[6,  paste0("V", 1:n_timepoints)]))),
      C7  = list(as.numeric(unlist(cur_data()[7,  paste0("V", 1:n_timepoints)]))),
      C8  = list(as.numeric(unlist(cur_data()[8,  paste0("V", 1:n_timepoints)]))),
      C9  = list(as.numeric(unlist(cur_data()[9,  paste0("V", 1:n_timepoints)]))),
      C10 = list(as.numeric(unlist(cur_data()[10, paste0("V", 1:n_timepoints)]))),
      C11 = list(as.numeric(unlist(cur_data()[11, paste0("V", 1:n_timepoints)]))),
      C12 = list(as.numeric(unlist(cur_data()[12, paste0("V", 1:n_timepoints)]))),
      C13 = list(as.numeric(unlist(cur_data()[13, paste0("V", 1:n_timepoints)]))),
      C14 = list(as.numeric(unlist(cur_data()[14, paste0("V", 1:n_timepoints)]))),
      C15 = list(as.numeric(unlist(cur_data()[15, paste0("V", 1:n_timepoints)]))),
      C16 = list(as.numeric(unlist(cur_data()[16, paste0("V", 1:n_timepoints)]))),
      C17 = list(as.numeric(unlist(cur_data()[17, paste0("V", 1:n_timepoints)]))),
      C18 = list(as.numeric(unlist(cur_data()[18, paste0("V", 1:n_timepoints)]))),
      C19 = list(as.numeric(unlist(cur_data()[19, paste0("V", 1:n_timepoints)]))),
      C20 = list(as.numeric(unlist(cur_data()[20, paste0("V", 1:n_timepoints)]))),
      C21 = list(as.numeric(unlist(cur_data()[21, paste0("V", 1:n_timepoints)]))),
      C22 = list(as.numeric(unlist(cur_data()[22, paste0("V", 1:n_timepoints)]))),
      C23 = list(as.numeric(unlist(cur_data()[23, paste0("V", 1:n_timepoints)]))),
      C24 = list(as.numeric(unlist(cur_data()[24, paste0("V", 1:n_timepoints)]))),
      C25 = list(as.numeric(unlist(cur_data()[25, paste0("V", 1:n_timepoints)]))),
      C26 = list(as.numeric(unlist(cur_data()[26, paste0("V", 1:n_timepoints)]))),
      C27 = list(as.numeric(unlist(cur_data()[27, paste0("V", 1:n_timepoints)]))),
      C28 = list(as.numeric(unlist(cur_data()[28, paste0("V", 1:n_timepoints)]))),
      C29 = list(as.numeric(unlist(cur_data()[29, paste0("V", 1:n_timepoints)]))),
      C30 = list(as.numeric(unlist(cur_data()[30, paste0("V", 1:n_timepoints)]))),
      C31 = list(as.numeric(unlist(cur_data()[31, paste0("V", 1:n_timepoints)]))),
      C32 = list(as.numeric(unlist(cur_data()[32, paste0("V", 1:n_timepoints)]))),
      C33 = list(as.numeric(unlist(cur_data()[33, paste0("V", 1:n_timepoints)]))),
      C34 = list(as.numeric(unlist(cur_data()[34, paste0("V", 1:n_timepoints)]))),
      C35 = list(as.numeric(unlist(cur_data()[35, paste0("V", 1:n_timepoints)]))),
      C36 = list(as.numeric(unlist(cur_data()[36, paste0("V", 1:n_timepoints)]))),
      C37 = list(as.numeric(unlist(cur_data()[37, paste0("V", 1:n_timepoints)]))),
      C38 = list(as.numeric(unlist(cur_data()[38, paste0("V", 1:n_timepoints)]))),
      C39 = list(as.numeric(unlist(cur_data()[39, paste0("V", 1:n_timepoints)]))),
      C40 = list(as.numeric(unlist(cur_data()[40, paste0("V", 1:n_timepoints)]))),
      C41 = list(as.numeric(unlist(cur_data()[41, paste0("V", 1:n_timepoints)]))),
      C42 = list(as.numeric(unlist(cur_data()[42, paste0("V", 1:n_timepoints)]))),
      C43 = list(as.numeric(unlist(cur_data()[43, paste0("V", 1:n_timepoints)]))),
      C44 = list(as.numeric(unlist(cur_data()[44, paste0("V", 1:n_timepoints)]))),
      C45 = list(as.numeric(unlist(cur_data()[45, paste0("V", 1:n_timepoints)]))),
      C46 = list(as.numeric(unlist(cur_data()[46, paste0("V", 1:n_timepoints)]))),
      C47 = list(as.numeric(unlist(cur_data()[47, paste0("V", 1:n_timepoints)]))),
      C48 = list(as.numeric(unlist(cur_data()[48, paste0("V", 1:n_timepoints)]))),
      C49 = list(as.numeric(unlist(cur_data()[49, paste0("V", 1:n_timepoints)]))),
      C50 = list(as.numeric(unlist(cur_data()[50, paste0("V", 1:n_timepoints)]))),
      C51 = list(as.numeric(unlist(cur_data()[51, paste0("V", 1:n_timepoints)]))),
      C52 = list(as.numeric(unlist(cur_data()[52, paste0("V", 1:n_timepoints)]))),
      C53 = list(as.numeric(unlist(cur_data()[53, paste0("V", 1:n_timepoints)]))),
      C54 = list(as.numeric(unlist(cur_data()[54, paste0("V", 1:n_timepoints)]))),
      C55 = list(as.numeric(unlist(cur_data()[55, paste0("V", 1:n_timepoints)]))),
      C56 = list(as.numeric(unlist(cur_data()[56, paste0("V", 1:n_timepoints)]))),
      C57 = list(as.numeric(unlist(cur_data()[57, paste0("V", 1:n_timepoints)]))),
      C58 = list(as.numeric(unlist(cur_data()[58, paste0("V", 1:n_timepoints)]))),
      C59 = list(as.numeric(unlist(cur_data()[59, paste0("V", 1:n_timepoints)]))),
      C60 = list(as.numeric(unlist(cur_data()[60, paste0("V", 1:n_timepoints)]))),
      C61 = list(as.numeric(unlist(cur_data()[61, paste0("V", 1:n_timepoints)]))),
      C62 = list(as.numeric(unlist(cur_data()[62, paste0("V", 1:n_timepoints)]))),
      C63 = list(as.numeric(unlist(cur_data()[63, paste0("V", 1:n_timepoints)]))),
      C64 = list(as.numeric(unlist(cur_data()[64, paste0("V", 1:n_timepoints)])))
    ) %>%
    ungroup()
}


# ----- 6. CREAR DATASETS FINALS -----

BSL_new   <- make_list_dataset(BSL,   150)
SENS_new  <- make_list_dataset(SENS,  500)
DELAY_new <- make_list_dataset(DELAY, 500)








# ----- 3. AFEGIR COLUMNA TIME A BSL -----

# t_BSL són els 150 temps del baseline
# Repetim aquests temps per cada subjecte, trial i canal

BSL <- BSL %>%
  group_by(subjectID, trialID, channel) %>%
  mutate(time = t_BSL) %>%
  ungroup()


# ----- 4. CONVERTIR BSL A FORMAT WIDE -----

BSL <- BSL %>%
  arrange(subjectID, trialID, channel, time) %>%
  mutate(time_index = paste0("V", match(time, t_BSL))) %>%
  select(-time) %>%
  pivot_wider(names_from = time_index, values_from = EEG)


# ----- 6. DETECTAR COLUMNES EEG -----

get_eeg_cols <- function(df){
  grep("^V[0-9]+$", names(df), value = TRUE)
}

eeg_BSL   <- get_eeg_cols(BSL)
eeg_SENS  <- get_eeg_cols(SENS)
eeg_DELAY <- get_eeg_cols(DELAY)


# ----- 7. FUNCIÓ GLMNET MULTICLASS -----

run_glmnet <- function(df, eeg_cols){
  
  # Si no hi ha prou categories → retornem NA
  if(n_distinct(df$category) < 2) return(NA_real_)
  
  # Si no hi ha prou trials → retornem NA
  if(nrow(df) < 10) return(NA_real_)
  
  # Si no hi ha prou columnes EEG → retornem NA
  if(length(eeg_cols) < 2) return(NA_real_)
  
  # Eliminar columnes EEG buides
  eeg_cols_valid <- eeg_cols[colSums(!is.na(df[, eeg_cols])) > 0]
  if(length(eeg_cols_valid) < 2) return(NA_real_)
  
  set.seed(123)
  idx <- createDataPartition(df$category, p = .7, list = FALSE)
  
  train <- df[idx, ]
  test  <- df[-idx, ]
  
  x_train <- as.matrix(train[, eeg_cols_valid])
  x_test  <- as.matrix(test[, eeg_cols_valid])
  y_train <- train$category
  
  model <- cv.glmnet(
    x_train,
    y_train,
    family = "multinomial",
    alpha = 0.5,
    type.measure = "class"
  )
  
  preds <- predict(model, x_test, s = "lambda.min", type = "class")
  accuracy <- mean(preds == test$category)
  
  return(accuracy)
}


# ----- 8. ENTRENAR PER PERÍODE -----

acc_BSL   <- BSL   %>% group_by(subjectID) %>% summarise(acc = run_glmnet(cur_data(), eeg_BSL))
acc_SENS  <- SENS  %>% group_by(subjectID) %>% summarise(acc = run_glmnet(cur_data(), eeg_SENS))
acc_DELAY <- DELAY %>% group_by(subjectID) %>% summarise(acc = run_glmnet(cur_data(), eeg_DELAY))

acc_BSL$period   <- "Baseline"
acc_SENS$period  <- "Sensory"
acc_DELAY$period <- "Delay"

results <- bind_rows(acc_BSL, acc_SENS, acc_DELAY)

print(results)


# ----- 9. VIOLIN PLOT -----

ggplot(results, aes(x = period, y = acc, fill = period)) +
  geom_violin(trim = FALSE, alpha = 0.6) +
  geom_jitter(width = 0.05, size = 1.5) +
  stat_summary(fun = mean, geom = "point", size = 4, color = "black") +
  stat_summary(fun = mean, geom = "text",
               aes(label = sprintf("%.2f", ..y..)),
               vjust = -1.5, color = "black") +
  geom_hline(yintercept = 1/3, linetype = "dashed") +
  theme_minimal(base_size = 14) +
  labs(title = "GLMNET Classification Accuracy by Period",
       y = "Accuracy",
       x = "")