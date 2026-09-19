# Data preparation for min-char-rnn.

read_text_file <- function(path) {
  if (!file.exists(path)) {
    stop(sprintf("Input file does not exist: %s", path))
  }

  size <- file.info(path)$size
  if (is.na(size) || size < 1) {
    stop("Input file is empty.")
  }

  raw <- readBin(path, what = "raw", n = size)
  text <- rawToChar(raw)
  Encoding(text) <- "UTF-8"

  validated <- iconv(text, from = "UTF-8", to = "UTF-8", sub = NA)
  if (is.na(validated)) {
    stop("Input must be valid UTF-8 text.")
  }

  text
}

text_to_characters <- function(text) {
  chars <- strsplit(text, split = "", fixed = TRUE)[[1L]]
  if (length(chars) < 2L) {
    stop("Input text must contain at least two characters.")
  }
  chars
}

build_vocab <- function(text) {
  chars_in_text <- text_to_characters(text)

  # Sorting gives a deterministic mapping. R indices are deliberately one-based.
  chars <- sort(unique(chars_in_text), method = "radix")
  char_to_ix <- seq_along(chars)
  names(char_to_ix) <- chars

  list(
    chars = chars,
    char_to_ix = char_to_ix,
    ix_to_char = chars,
    size = length(chars)
  )
}

encode_text <- function(text, vocab) {
  chars <- text_to_characters(text)
  ids <- unname(vocab$char_to_ix[chars])

  if (anyNA(ids)) {
    missing_chars <- unique(chars[is.na(ids)])
    stop(sprintf(
      "Text contains characters absent from the vocabulary: %s",
      paste(missing_chars, collapse = " ")
    ))
  }

  as.integer(ids)
}

decode_indices <- function(indices, vocab) {
  if (length(indices) == 0L) {
    return("")
  }

  if (any(indices < 1L | indices > vocab$size)) {
    stop("Indices are outside the one-based vocabulary range.")
  }

  paste0(vocab$ix_to_char[indices], collapse = "")
}

split_sequence <- function(ids, validation_fraction = 0.1, seq_length = 25L) {
  if (validation_fraction <= 0 || validation_fraction >= 0.5) {
    stop("validation_fraction must be greater than 0 and less than 0.5.")
  }

  n <- length(ids)
  train_n <- floor(n * (1 - validation_fraction))

  train <- ids[seq_len(train_n)]
  validation <- ids[seq.int(train_n + 1L, n)]

  minimum <- seq_length + 1L
  if (length(train) < minimum || length(validation) < minimum) {
    stop(sprintf(
      paste0(
        "Dataset is too small for seq_length=%d and validation_fraction=%.3f. ",
        "Need at least %d characters in both splits."
      ),
      seq_length,
      validation_fraction,
      minimum
    ))
  }

  list(train = train, validation = validation)
}

input_checksum <- function(path) {
  unname(tools::md5sum(path))
}

save_vocab <- function(vocab, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  saveRDS(vocab, path)
  invisible(path)
}

load_vocab <- function(path) {
  vocab <- readRDS(path)

  required <- c("chars", "char_to_ix", "ix_to_char", "size")
  if (!all(required %in% names(vocab))) {
    stop("Vocabulary file is missing required fields.")
  }

  vocab
}

