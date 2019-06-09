offense_type_words <- readRDS("./data/offense_type_words.rds")
sentiment_words <- readRDS("./data/sentiment_words.rds")
negative_words <- readRDS("./data/negative_words.rds")
positive_words <- readRDS("./data/positive_words.rds")
neutral_words <- readRDS("./data/neutral_words.rds")

create_tf_idf_df <- function(df) {
  tf_idf_df <-
    df %>%
    unnest_tokens(word, text, drop = FALSE) %>%
    merge(sentiment_words)
  return(tf_idf_df)
}
max_ <- function(..., default=0, na.rm=FALSE) {
  if(!is.infinite(x <- suppressWarnings(max(..., na.rm = na.rm)))) x else default
}
min_ <- function(..., default=0, na.rm=FALSE) {
  if(!is.infinite(x <- suppressWarnings(min(..., na.rm = na.rm)))) x else default
}
mean_ <- function(..., default=0, na.rm=FALSE) {
  if(!is.na(x <- suppressWarnings(mean(..., na.rm=na.rm)))) x else default
}
max_positive_tfidf <- function(tf_idf_df, df, default=0) {
  if (nrow(tf_idf_df) == 0) {
    result_df <- df[, max_positive_tfidf := default]
  } else {
    temp <-
      tf_idf_df %>%
      group_by(text) %>%
      summarise(max_positive_tfidf = max_(tf_idf[tf_idf > 0], na.rm = TRUE)) %>%
      as.data.table()
    result_df <- df[temp, on="text"]
    result_df[, max_positive_tfidf := max_positive_tfidf/words]
  }
  return(result_df)
}
min_positive_tfidf <- function(tf_idf_df, df, default=0) {
  if (nrow(tf_idf_df) == 0) {
    result_df <- df[, min_positive_tfidf := default]
  } else {
    temp <-
      tf_idf_df %>%
      group_by(text) %>%
      summarise(min_positive_tfidf = min_(tf_idf[tf_idf > 0], na.rm = TRUE)) %>%
      as.data.table()
    result_df <- df[temp, on="text"]
    result_df[, min_positive_tfidf := min_positive_tfidf/words]
  }
  return(result_df)
}
avg_positive_tfidf <- function(tf_idf_df, df, default=0) {
  if (nrow(tf_idf_df) == 0) {
    result_df <- df[, avg_positive_tfidf := default]
  } else {
    temp <-
      tf_idf_df %>%
      group_by(text) %>%
      summarise(avg_positive_tfidf = mean_(tf_idf[tf_idf > 0], na.rm = TRUE)) %>%
      as.data.table()
    result_df <- df[temp, on="text"]
    result_df[, avg_positive_tfidf := avg_positive_tfidf/words]
  }
  return(result_df)
}
max_negative_tfidf <- function(tf_idf_df, df, default=0) {
  if (nrow(tf_idf_df) == 0) {
    result_df <- df[, max_negative_tfidf := default]
  } else {
    temp <-
      tf_idf_df %>%
      group_by(text) %>%
      summarise(max_negative_tfidf = max_(tf_idf[tf_idf < 0], na.rm = TRUE)) %>%
      as.data.table()
    result_df <- df[temp, on="text"]
    result_df[, max_negative_tfidf := max_negative_tfidf/words]
  }
  return(result_df)
}
min_negative_tfidf <- function(tf_idf_df, df, default=0) {
  if (nrow(tf_idf_df) == 0) {
    result_df <- df[, min_negative_tfidf := default]
  } else {
    temp <-
      tf_idf_df %>%
      group_by(text) %>%
      summarise(min_negative_tfidf = min_(tf_idf[tf_idf < 0], na.rm = TRUE)) %>%
      as.data.table()
    result_df <- df[temp, on="text"]
    result_df[, min_negative_tfidf := min_negative_tfidf/words]
  }
  return(result_df)
}
avg_negative_tfidf <- function(tf_idf_df, df, default=0) {
  if (nrow(tf_idf_df) == 0) {
    result_df <- df[, avg_negative_tfidf := default]
  } else {
    temp <-
      tf_idf_df %>%
      group_by(text) %>%
      summarise(avg_negative_tfidf = mean_(tf_idf[tf_idf < 0], na.rm = TRUE)) %>%
      as.data.table()
    result_df <- df[temp, on="text"]
    result_df[, avg_negative_tfidf := avg_negative_tfidf/words]
  }
  return(result_df)
}
avg_tfidf <- function(tf_idf_df, df, default=0) {
  if (nrow(tf_idf_df) == 0) {
    result_df <- df[, avg_tfidf := default]
  } else {
    temp <-
      tf_idf_df %>%
      group_by(text) %>%
      summarise(avg_tfidf = mean(tf_idf, na.rm = TRUE)) %>%
      as.data.table()
    result_df <- df[temp, on="text"]
    result_df[, avg_tfidf := avg_tfidf/words]
  }
  return(result_df)
}
negative_tfidf_count <- function(tf_idf_df, df, default=0) {
  if (nrow(tf_idf_df) == 0) {
    result_df <- df[, negative_tfidf_count := default]
  } else {
    temp <-
      tf_idf_df %>%
      group_by(text) %>%
      summarise(negative_tfidf_count = sum(tf_idf < 0)) %>%
      as.data.table()
    result_df <- df[temp, on="text"]
    result_df[, negative_tfidf_count := negative_tfidf_count/words]
  }
  return(result_df)
}
positive_tfidf_count <- function(tf_idf_df, df, default=0) {
  if (nrow(tf_idf_df) == 0) {
    result_df <- df[, positive_tfidf_count := default]
  } else {
    temp <-
      tf_idf_df %>%
      group_by(text) %>%
      summarise(positive_tfidf_count = sum(tf_idf > 0)) %>%
      as.data.table()
    result_df <- df[temp, on="text"]
    result_df[, positive_tfidf_count := positive_tfidf_count/words]
  }
  return(result_df)
}
negative_count <- function(tf_idf_df, df, default=0) {
  if (nrow(tf_idf_df) == 0) {
    result_df <- df[, negative_count := default]
  } else {
    temp <-
      tf_idf_df %>%
      group_by(text) %>%
      summarise(negative_count = sum(word %in% negative_words)) %>%
      as.data.table()
    result_df <- df[temp, on="text"]
    result_df[, negative_count := negative_count/words]
  }
  return(result_df)
}
positive_count <- function(tf_idf_df, df, default=0) {
  if (nrow(tf_idf_df) == 0) {
    result_df <- df[, positive_count := default]
  } else {
    temp <-
      tf_idf_df %>%
      group_by(text) %>%
      summarise(positive_count = sum(word %in% positive_words)) %>%
      as.data.table()
    result_df <- df[temp, on="text"]
    result_df[, positive_count := positive_count/words]
  }
  return(result_df)
}
neutral_count <- function(tf_idf_df, df, default=0) {
  if (nrow(tf_idf_df) == 0) {
    result_df <- df[, neutral_count := default]
  } else {
    temp <-
      tf_idf_df %>%
      group_by(text) %>%
      summarise(neutral_count = sum(word %in% neutral_words)) %>%
      as.data.table()
    result_df <- df[temp, on="text"]
    result_df[, neutral_count := neutral_count/words]
  }
  return(result_df)
}

create_features <- function(df) {
  df <- copy(df)
  df[, chars := nchar(text)]
  df[, words := str_count(text, pattern = "[^ ]+")]
  df[, avg_chars_per_word := chars / words]
  df[, text := gsub("@[a-zäöüß_0-9]+", "@MENTION", text, ignore.case = TRUE)]
  df[, mentions := str_count(text, pattern = "@MENTION") / words]
  df[, text := gsub("(^| )@MENTION", "", text)]
  df[, linebreaks := str_count(text, pattern = " \\|LBR\\| ") / chars]
  df[, text := gsub(" \\|LBR\\| ", " ", text, ignore.case = TRUE)]
  df[, consecutive_punctuation := str_count(text, pattern = "([[:punct:]])\\1") / chars]
  df[, quotes := str_count(text, pattern = "([\"']).+?\\1") / words]
  df[, punctuation_marks := str_count(text, pattern = "[[:punct:],^#]")  / chars]
  df[, hashtags := str_count(text, pattern = "#") / words]
  df[, all_caps_words := str_count(text, pattern = "(?:^| )[A-ZÄÖÜ]{4,}") / words]
  df[, ws_before_punctuation := str_count(text, pattern = " [;,!\\.\\?] ?[[:alpha:][;,!\\.\\?]]+(?:$| )")]
  df[, non_ascii := str_count(text, pattern = regex("(?![äöüß])[^[:ascii:]]", ignore.case = TRUE)) / words]

  tf_idf_df <- create_tf_idf_df(df)
  df <- max_positive_tfidf(tf_idf_df, df)
  df <- min_positive_tfidf(tf_idf_df, df)
  df <- avg_positive_tfidf(tf_idf_df, df)
  df <- max_negative_tfidf(tf_idf_df, df)
  df <- min_negative_tfidf(tf_idf_df, df)
  df <- avg_negative_tfidf(tf_idf_df, df)
  df <- avg_tfidf(tf_idf_df, df)
  df <- negative_tfidf_count(tf_idf_df, df)
  df <- positive_tfidf_count(tf_idf_df, df)
  df <- negative_count(tf_idf_df, df)
  df <- positive_count(tf_idf_df, df)
  df <- neutral_count(tf_idf_df, df)
  return(df)
}