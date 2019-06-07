#!/usr/bin/ Rscript
# I. load libraries and source functions
library('caret')
library('dplyr')
library('tidytext')
library('data.table')
library('stringr')

source("./R/feature_creation.R")

# II. read data
# read tweet data
train_raw_df <- read.table("./data/germeval2018.training.txt", sep = "\t", header = FALSE, fill = TRUE, comment.char = "",
                           quote = NULL, stringsAsFactors = FALSE, na.strings = "", encoding = "UTF-8")
test_raw_df  <- read.table("./data/germeval2018.test.txt", sep = "\t", header = FALSE, fill = TRUE, comment.char = "",
                           quote = NULL, stringsAsFactors = FALSE, na.strings = "", encoding = "UTF-8")

# read sentiment dictionaries
negative_dictionary <- read.table("./data/GermanPolarityClues-Negative-21042012.tsv", sep = "\t", header = FALSE, fill = TRUE, comment.char = "",
                                  quote = NULL, stringsAsFactors = FALSE, na.strings = "", encoding = "UTF-8")
positive_dictionary <- read.table("./data/GermanPolarityClues-Positive-21042012.tsv", sep = "\t", header = FALSE, fill = TRUE, comment.char = "",
                                  quote = NULL, stringsAsFactors = FALSE, na.strings = "", encoding = "UTF-8")
neutral_dictionary <- read.table("./data/GermanPolarityClues-Neutral-21042012.tsv", sep = "\t", header = FALSE, fill = TRUE, comment.char = "",
                                  quote = NULL, stringsAsFactors = FALSE, na.strings = "", encoding = "UTF-8")

negative_words <- tolower(negative_dictionary$V1)
positive_words <- tolower(negative_dictionary$V1)
neutral_words <- tolower(neutral_dictionary$V1)

saveRDS(negative_words, "./data/negative_words.rds")
saveRDS(positive_words, "./data/positive_words.rds")
saveRDS(neutral_words, "./data/neutral_words.rds")

# combine data
total_df <- rbind(train_raw_df, test_raw_df)

# III. preprocess text
# 1. rename columns
names(total_df) <- c("text", "sentiment", "offense_type")

# 2. factor variables
total_df$sentiment <- as.factor(total_df$sentiment)
total_df$offense_type <- as.factor(total_df$offense_type)

# 4. remove duplicates
total_df <- total_df[!duplicated(total_df$text), ]
print(nrow(total_df))

# 5. remove rows where sentiment and offense_type is NA
total_df <- total_df[!(is.na(total_df$sentiment) & is.na(total_df$offense_type)), ]
print(nrow(total_df))

# 6. split into training and test data
train_index <- createDataPartition(total_df$sentiment, p=0.8, list=FALSE)
train_df <- total_df[train_index,]
test_df <- total_df[-train_index,]

# IV. create sentiment tfidf dictionaries
sentiment_words <- train_df %>%
  unnest_tokens(word, text) %>%
  count(sentiment, word, sort = TRUE) %>%
  bind_tf_idf(word, sentiment, n) %>%
  mutate(tf_idf = ifelse(sentiment == "OTHER", tf_idf, tf_idf * -1)) %>%
  arrange(tf_idf)

offense_type_words <- train_df %>%
  unnest_tokens(word, text) %>%
  count(offense_type, word, sort = TRUE) %>%
  bind_tf_idf(word, offense_type, n) %>%
  mutate(tf_idf = ifelse(offense_type == "OTHER", tf_idf, tf_idf * -1)) %>%
  arrange(tf_idf)

saveRDS(offense_type_words, "./data/sentiment_words.rds")
saveRDS(offense_type_words, "./data/offense_type_words.rds")

# V. feature engineering
train_df <- create_features(df = train_df)
test_df <- create_features(df = test_df)

# VI. model training
# 1. remove text and offense_type
train <- train_df[,!(colnames(train_df) %in% c("text", "offense_type"))]
test <- test_df[,!(colnames(test_df) %in% c("text", "offense_type"))]

# 2. scale and center training data
scaler <- preProcess(train, method = c("center", "scale"))
train <- predict(scaler, train)
test <- predict(scaler, test)

saveRDS(scaler, "./models/scaler.rds")

# 3. define train control and tune grid
train_control <- trainControl(method = "cv",
                              number = 5,
                              classProbs = TRUE)

tune_grid <- expand.grid(nrounds = c(30, 50, 75),
                         max_depth = c(4, 5, 7),
                         eta = c(0.08, 0.1, 0.12, 0.2),
                         gamma = c(0, 0.01, 0.05),
                         colsample_bytree = c(0.5, 0.75, 1),
                         min_child_weight = c(0, 0.3, 1),
                         subsample = c(0.7, 0.8, 0.9))

# 4. fit model
fit <- train(sentiment ~ .,
             data = train,
             method = "xgbTree",
             trControl = train_control,
             tuneGrid = tune_grid,
             tuneLength = 10)

# 5. testing
test_predict <- predict(fit, test)

# 6. print metrics
prec <- precision(test_predict, reference = test$sentiment)
rec <- recall(test_predict, reference = test$sentiment)
f1 <- F_meas(test_predict, reference = test$sentiment)
conf_matrix <- confusionMatrix(test_predict, reference = test$sentiment)
print(paste("precision:", round(prec, digits = 3), " | ",
            "recall:", round(rec, digits = 3), " | ",
            "F1:", round(f1, digits = 3), " | ",
            "accuracy:", round(conf_matrix$overall["Accuracy"], digits = 3)))

# 7. save model
saveRDS(fit, paste0("./models/model_accuracy_", round(temp["Accuracy"], digits = 3),".rds"))
saveRDS(fit, paste0("./models/model.rds"))
