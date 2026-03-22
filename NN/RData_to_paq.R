library(arrow)
library(tidyverse)

# Set working directory
setwd("C:\\Users\\lefor\\Desktop\\OneDriveBackupFiles\\Documentos\\Q4 MUEI\\TFM\\GitHub")

library(data.table)
library(dplyr)
library(arrow)

# Load RData
load("BSL_allChan.RData")
load("DELAY_allChan.RData")
load("SENSORY_allChan.RData")

# Function to expand EEG matrix
expand_eeg_columns <- function(df, list_col = "EEG") {
  
  eeg_matrix <- df[[list_col]]
  colnames(eeg_matrix) <- paste0("EEG.V", seq_len(ncol(eeg_matrix)))

  df_out <- cbind(df, eeg_matrix)
  df_out <- select(df_out, -EEG)
  
  return(df_out)
}

# Expand
BSL_exp   <- expand_eeg_columns(BSL, "EEG")
SENS_exp  <- expand_eeg_columns(SENS, "EEG")
DELAY_exp <- expand_eeg_columns(DELAY, "EEG")

# Save as PARQUET
write_parquet(BSL_exp, "NN/BSL_allChan.parquet", compression = "snappy")
write_parquet(SENS_exp, "NN/SENS_allChan.parquet", compression = "snappy")
write_parquet(DELAY_exp, "NN/DELAY_allChan.parquet", compression = "snappy")