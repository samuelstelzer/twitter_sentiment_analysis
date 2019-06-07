offense_type_path <-ifelse(Sys.info()[['sysname']]=="Windows",
                           "./data/offense_type_words.rds",
                           "\\\\data\\offense_type_words.rds")
offense_type_words <- readRDS(offense_type_path)

sentiment_words_path <- ifelse(Sys.info()[['sysname']]=="Windows",
                               "./data/sentiment_words.rds",
                               "\\\\data\\sentiment_words.rds")
sentiment_words <- readRDS("./data/sentiment_words.rds")

negative_words_path <- ifelse(Sys.info()[['sysname']]=="Windows",
                              "./data/negative_words.rds",
                              "\\\\data\\negative_words.rds")
negative_words <- readRDS(negative_words_path)

positive_words_path <- ifelse(Sys.info()[['sysname']]=="Windows",
                              "./data/positive_words.rds",
                              "\\\\data\\positive_words.rds")
positive_words <- readRDS(positive_words_path)

neutral_words_path <- ifelse(Sys.info()[['sysname']]=="Windows",
                              "./data/neutral_words.rds",
                              "\\\\data\\neutral_words.rds")
neutral_words <- readRDS(neutral_words_path)

create_tf_idf_df <- function(df) {
  tf_idf_df <-
    df %>%
    unnest_tokens(word, text, drop=FALSE) %>%
    merge(sentiment_words)
  return(tf_idf_df)
}
max_ <- function(..., def=0, na.rm=FALSE) {
  if(!is.infinite(x <- suppressWarnings(max(..., na.rm=na.rm)))) x else def
}
min_ <- function(..., def=0, na.rm=FALSE) {
  if(!is.infinite(x <- suppressWarnings(max(..., na.rm=na.rm)))) x else def
}
mean_ <- function(..., def=0, na.rm=FALSE) {
  if(!is.na(x <- suppressWarnings(mean(..., na.rm=na.rm)))) x else def
}
max_positive_sentiment_tfidf <- function(tf_idf_df) {
  tf_idf_df <-
    tf_idf_df %>%
    group_by(text) %>%
    summarise(tf_idf_max = max_(tf_idf[tf_idf > 0], na.rm = TRUE))
  return(tf_idf_df$tf_idf_max)
}
min_positive_sentiment_tfidf <- function(tf_idf_df) {
  tf_idf_df <-
    tf_idf_df %>%
    group_by(text) %>%
    summarise(tf_idf_min = min_(tf_idf[tf_idf > 0], na.rm = TRUE))
  return(tf_idf_df$tf_idf_min)
}
avg_positive_sentiment_tfidf <- function(tf_idf_df) {
  tf_idf_df <-
    tf_idf_df %>%
    group_by(text) %>%
    summarise(tf_idf_avg = mean_(tf_idf[tf_idf > 0], na.rm = TRUE))
  return(tf_idf_df$tf_idf_avg)
}
max_negative_sentiment <- function(tf_idf_df) {
  tf_idf_df <-
    tf_idf_df %>%
    group_by(text) %>%
    summarise(tf_idf_max = max_(tf_idf[tf_idf < 0], na.rm = TRUE))
  return(tf_idf_df$tf_idf_max)
}
min_negative_sentiment <- function(tf_idf_df) {
  tf_idf_df <-
    tf_idf_df %>%
    group_by(text) %>%
    summarise(tf_idf_min = min_(tf_idf[tf_idf < 0], na.rm = TRUE))
  return(tf_idf_df$tf_idf_min)
}
avg_negative_sentiment_tfidf <- function(tf_idf_df) {
  tf_idf_df <-
    tf_idf_df %>%
    group_by(text) %>%
    summarise(tf_idf_avg = mean_(tf_idf[tf_idf < 0], na.rm = TRUE))
  return(tf_idf_df$tf_idf_avg)
}
avg_sentiment_tfidf <- function(tf_idf_df) {
  tf_idf_df <-
    tf_idf_df %>%
    group_by(text) %>%
    summarise(tf_idf_avg = mean(tf_idf, na.rm = TRUE))
  return(tf_idf_df$tf_idf_avg)
}
number_of_negatives_tfidf <- function(tf_idf_df) {
  tf_idf_df <-
    tf_idf_df %>%
    group_by(text) %>%
    summarise(negative_count = sum(tf_idf < 0))
  return(tf_idf_df$negative_count)
}
number_of_positives_tfidf <- function(tf_idf_df) {
  tf_idf_df <-
    tf_idf_df %>%
    group_by(text) %>%
    summarise(positive_count = sum(tf_idf > 0))
  return(tf_idf_df$positive_count)
}

number_of_positives_tfidf <- function(tf_idf_df) {
  tf_idf_df <-
    tf_idf_df %>%
    group_by(text) %>%
    summarise(positive_count = sum(tf_idf > 0))
  return(tf_idf_df$positive_count)
}
number_of_negatives <- function(tf_idf_df) {
  tf_idf_df <-
    tf_idf_df %>%
    group_by(text) %>%
    summarise(negative_count = sum(word %in% negative_words))
  return(tf_idf_df$negative_count)
}
number_of_positives <- function(tf_idf_df) {
  tf_idf_df <-
    tf_idf_df %>%
    group_by(text) %>%
    summarise(positive_count = sum(word %in% positive_words))
  return(tf_idf_df$positive_count)
}
number_of_neutrals <- function(tf_idf_df) {
  tf_idf_df <-
    tf_idf_df %>%
    group_by(text) %>%
    summarise(neutral_count = sum(word %in% neutral_words))
  return(tf_idf_df$neutral_count)
}

create_features <- function(df) {
  df$chars <- nchar(df$text)
  df$words <- str_count(df$text, pattern = "[^ ]+")
  df$avg_chars_per_word <- df$chars / df$words
  df$text <- gsub("@[a-zäöüß_0-9]+", "@MENTION", df$text, ignore.case = TRUE)
  df$mentions <- str_count(df$text, pattern = "@MENTION") / df$words
  df$text <- gsub("(^| )@MENTION", "", df$text)
  df$linebreaks <- str_count(df$text, pattern = " \\|LBR\\| ") / df$chars
  df$text <- gsub(" \\|LBR\\| ", " ", df$text, ignore.case = TRUE)
  df$consecutive_punctuation <- str_count(df$text, pattern = "([[:punct:]])\\1") / df$chars
  df$quotes <- str_count(df$text, pattern = "([\"']).+?\\1")
  df$punctuation_marks <- str_count(df$text, pattern = "[[:punct:],^#]")  / df$chars
  df$hashtags <- str_count(df$text, pattern = "#") / df$words
  df$all_caps_words <- str_count(df$text, pattern = "(?:^| )[A-ZÄÖÜ]{4,}") / df$words
  df$ws_before_punctuation <- str_count(df$text, pattern = " [;,!\\.\\?] ?[[:alpha:][;,!\\.\\?]]+(?:$| )")
  df$non_ascii <- str_count(df$text, pattern = "(?![äöüß])[^[:ascii:]]") / df$words
  tf_idf_df <- create_tf_idf_df(df)
  df$max_positive_sentiment_tfidf <- ifelse((nrow(tf_idf_df) == 0), 0, max_positive_sentiment_tfidf(tf_idf_df))
  df$min_positive_sentiment_tfidf <- ifelse((nrow(tf_idf_df) == 0), 0, min_positive_sentiment_tfidf(tf_idf_df))
  df$avg_positive_sentiment_tfidf <- ifelse((nrow(tf_idf_df) == 0), 0, avg_positive_sentiment_tfidf(tf_idf_df))
  df$max_negative_sentiment <- ifelse((nrow(tf_idf_df) == 0), 0, max_negative_sentiment(tf_idf_df))
  df$min_negative_sentiment <- ifelse((nrow(tf_idf_df) == 0), 0, min_negative_sentiment(tf_idf_df))
  df$avg_negative_sentiment_tfidf <- ifelse((nrow(tf_idf_df) == 0), 0, avg_negative_sentiment_tfidf(tf_idf_df))
  df$avg_sentiment_tfidf <- ifelse((nrow(tf_idf_df) == 0), 0, avg_sentiment_tfidf(tf_idf_df))
  df$negatives_tfidf <- ifelse((nrow(tf_idf_df) == 0), 0, number_of_negatives_tfidf(tf_idf_df)/df$words)
  df$positives_tfidf <- ifelse((nrow(tf_idf_df) == 0), 0, number_of_positives_tfidf(tf_idf_df)/df$words)
  df$negatives <- ifelse((nrow(tf_idf_df) == 0), 0, number_of_negatives(tf_idf_df)/df$words)
  df$positives <- ifelse((nrow(tf_idf_df) == 0), 0, number_of_positives(tf_idf_df)/df$words)
  df$negatives <- ifelse((nrow(tf_idf_df) == 0), 0, number_of_neutrals(tf_idf_df)/df$words)
  return(df)
}