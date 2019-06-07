#!/usr/bin/ Rscript
options(warn=-1)
suppressMessages(library(caret))
suppressMessages(library(dplyr))
suppressMessages(library(tidytext))
suppressMessages(library(data.table))
suppressMessages(library(stringr))
source("./R/feature_creation.R")

print(paste("WD:", getwd()))
args <- commandArgs(trailingOnly=TRUE)
print(paste("Predicting tweet \'", args, "\'"))
print("Reading model...")
model <- readRDS("./models/model.rds")
print("Reading scaler...")
scaler <- readRDS("./models/scaler.rds")
text <- args
df <- data.frame(text, stringsAsFactors = FALSE)
print("Creating features scaler...")
df <- create_features(df)
print("Scaling features...")
df <- predict(scaler, df)
df <- df[,2:ncol(df)]
print("Predicting...")
result <- predict(model, df)

print(paste("Prediction for tweet:", text, " - ", as.character(result)))
