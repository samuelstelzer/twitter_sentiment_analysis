#!/usr/bin/env Rscript
options(warn=-1)
suppressMessages(library(dplyr))
suppressMessages(library(tidytext))
suppressMessages(library(data.table))
suppressMessages(library(stringr))
source("./R/feature_creation.R")

args <- commandArgs(trailingOnly=TRUE)
model <- readRDS("./models/model.RDS")
scaler <- readRDS("./models/scaler.rds")
text <- args
df <- data.frame(text, stringsAsFactors = FALSE)
df <- create_features(df)
df <- predict(scaler, df)
df <- df[,2:ncol(df)]
result <- predict(model, df)

print(paste("Prediction for Tweet:", text, " - ", as.character(result)))