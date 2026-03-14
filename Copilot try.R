# Copilot try

setwd("C:\\Users\\lefor\\Desktop\\OneDriveBackupFiles\\Documentos\\Q4 MUEI\\TFM\\Base Dataset")
# ---------------------------------------------------------
# COMPARACIÓ BASELINE vs SENSORY vs DELAY AMB GLM
# ---------------------------------------------------------

library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)
library(purrr)

# ---------------------------------------------------------
# 1. CARREGAR DADES
# ---------------------------------------------------------

load("BSL_allChan.RData")
load("DELAY_allChan.RData")
load("SENSORY_allChan.RData")

BSL$subjectID   <- as.factor(BSL$subjectID)
DELAY$subjectID <- as.factor(DELAY$subjectID)
SENS$subjectID  <- as.factor(SENS$subjectID)

# ---------------------------------------------------------
# 2. ANÀLISI DESCRIPTIVA
# ---------------------------------------------------------

# cat("\n--- DESCRIPTIVES ---\n")
# summary(BSL)
# summary(SENS)
# summary(DELAY)

# cat("\nTrials per subject (Baseline):\n")
# table(BSL$subjectID)

# ---------------------------------------------------------
# 2. CREAR COLUMNA CATEGORY (visual / spatial / verbal)
# ---------------------------------------------------------

assign_categories <- function(df){
  df %>%
    arrange(subjectID, trialID) %>%
    group_by(subjectID) %>%
    mutate(category = case_when(
      row_number() <= 128 ~ "visual",
      row_number() <= 256 ~ "spatial",
      TRUE ~ "verbal"
    )) %>%
    ungroup()
}

BSL  <- assign_categories(BSL)
SENS <- assign_categories(SENS)
DELAY <- assign_categories(DELAY)

# ---------------------------------------------------------
# 3. FUNCIÓ PER PREPARAR DADES (long → wide)
# ---------------------------------------------------------

prep_data <- function(df, label_name){
  df %>%
    group_by(subjectID, trialID, category, channel) %>%
    summarise(EEG = mean(EEG), .groups = "drop") %>%
    pivot_wider(names_from = channel, values_from = EEG) %>%
    mutate(period = label_name)
}

BSL_cat   <- prep_data(BSL, "Baseline")
SENS_cat  <- prep_data(SENS, "Sensory")
DELAY_cat <- prep_data(DELAY, "Delay")

all_data <- bind_rows(BSL_cat, SENS_cat, DELAY_cat)

# ---------------------------------------------------------
# 4. FUNCIÓ PER ENTRENAR GLM PER SUBJECTE I PERÍODE
# ---------------------------------------------------------

run_glm <- function(df){

  # Si no hi ha prou categories → retornem NA
  if(n_distinct(df$category) < 2){
    return(NA_real_)
  }
  
  # Si no hi ha prou trials → retornem NA
  if(nrow(df) < 10){
    return(NA_real_)
  }

  # Seleccionar només columnes EEG
  eeg_cols <- grep("^\\d+$|^EEG", names(df), value = TRUE)

  set.seed(123)
  idx <- createDataPartition(df$category, p = .7, list = FALSE)

  train <- df[idx, ]
  test  <- df[-idx, ]

  model <- train(
    x = train[, eeg_cols],
    y = train$category,
    method = "multinom",
    trace = FALSE
  )

  preds <- predict(model, test[, eeg_cols])
  accuracy <- mean(preds == test$category)

  return(accuracy)
}


# ---------------------------------------------------------
# 5. APLICAR GLM PER SUBJECTE I PERÍODE
# ---------------------------------------------------------

results <- all_data %>%
  group_by(subjectID, period) %>%
  group_modify(~ tibble(accuracy = run_glm(.x))) %>%
  ungroup()

print(results)

# ---------------------------------------------------------
# 6. VIOLIN PLOT ESTIL ARTICLE
# ---------------------------------------------------------

ggplot(results, aes(x = period, y = accuracy, fill = period)) +
  geom_violin