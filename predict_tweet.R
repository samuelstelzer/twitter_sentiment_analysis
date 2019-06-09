#!/usr/bin/ Rscript
options(warn=-1)
suppressMessages(library('caret'))
suppressMessages(library('dplyr'))
suppressMessages(library('tidytext'))
suppressMessages(library('data.table'))
suppressMessages(library('stringr'))
source("./R/feature_creation.R")

text <- commandArgs(trailingOnly=TRUE)
model <- readRDS("./models/model.rds")
scaler <- readRDS("./models/scaler.rds")
df <- data.table(text, stringsAsFactors = FALSE)
df <- create_features(df)
df <- predict(scaler, df)
df <- df[,2:ncol(df)]
result <- predict(model, df)

print(paste("Prediction for tweet:", text, " - ", as.character(result)))
