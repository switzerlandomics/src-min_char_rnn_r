#!/usr/bin/env Rscript

script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)

  if (length(file_arg) == 0L) {
    return(normalizePath("tests/test_model.R", mustWork = FALSE))
  }

  normalizePath(sub("^--file=", "", file_arg[1L]), mustWork = FALSE)
}

PROJECT_ROOT <- normalizePath(
  file.path(dirname(script_path()), ".."),
  mustWork = TRUE
)
setwd(PROJECT_ROOT)

source(file.path("R", "model.R"))
source(file.path("R", "data.R"))

run_test <- function(name, fn) {
  started <- proc.time()[["elapsed"]]

  tryCatch(
    {
      fn()
      elapsed <- proc.time()[["elapsed"]] - started
      cat(sprintf("[PASS] %-36s %.3fs\n", name, elapsed))
    },
    error = function(e) {
      cat(sprintf("[FAIL] %s\n", name))
      cat(sprintf("       %s\n", conditionMessage(e)))
      quit(status = 1L)
    }
  )
}

cat("min-char-rnn test suite\n\n")

run_test("encode/decode round trip", function() {
  text <- "hello\nRNN!"
  vocab <- build_vocab(text)
  ids <- encode_text(text, vocab)
  stopifnot(identical(decode_indices(ids, vocab), text))
  stopifnot(min(ids) == 1L)
  stopifnot(max(ids) == vocab$size)
})

run_test("model parameter dimensions", function() {
  set.seed(1)
  model <- initialise_model(vocab_size = 5L, hidden_size = 4L)

  stopifnot(identical(dim(model$Wxh), c(4L, 5L)))
  stopifnot(identical(dim(model$Whh), c(4L, 4L)))
  stopifnot(identical(dim(model$Why), c(5L, 4L)))
  stopifnot(identical(dim(model$bh), c(4L, 1L)))
  stopifnot(identical(dim(model$by), c(5L, 1L)))
  stopifnot(model_parameter_count(model) == 65L)
})

run_test("stable softmax", function() {
  y <- matrix(c(1000, 1001, 999), ncol = 1L)
  p <- stable_softmax(y)

  stopifnot(all(is.finite(p)))
  stopifnot(abs(sum(p) - 1) < 1e-12)
  stopifnot(all(p > 0))
})

run_test("forward/backward dimensions", function() {
  set.seed(2)
  model <- initialise_model(vocab_size = 4L, hidden_size = 3L)
  inputs <- c(1L, 2L, 3L, 2L)
  targets <- c(2L, 3L, 2L, 4L)

  result <- loss_fun(model, inputs, targets, clip_value = 5)

  stopifnot(is.finite(result$loss))
  stopifnot(identical(dim(result$grads$Wxh), dim(model$Wxh)))
  stopifnot(identical(dim(result$grads$Whh), dim(model$Whh)))
  stopifnot(identical(dim(result$grads$Why), dim(model$Why)))
  stopifnot(identical(dim(result$grads$bh), dim(model$bh)))
  stopifnot(identical(dim(result$grads$by), dim(model$by)))
  stopifnot(identical(dim(result$last_h), c(3L, 1L)))
})

run_test("finite-difference gradients", function() {
  set.seed(3)
  model <- initialise_model(vocab_size = 4L, hidden_size = 3L)
  inputs <- c(1L, 2L, 3L, 2L)
  targets <- c(2L, 3L, 2L, 4L)

  check <- gradient_check(
    model,
    inputs,
    targets,
    epsilon = 1e-5,
    max_checks_per_parameter = Inf
  )

  max_error <- max(check$relative_error)
  cat(sprintf("       max relative error: %.3e\n", max_error))
  stopifnot(is.finite(max_error))
  stopifnot(max_error < 1e-5)
})

run_test("AdaGrad changes parameters", function() {
  set.seed(4)
  model <- initialise_model(vocab_size = 4L, hidden_size = 3L)
  original <- model
  state <- new_adagrad_state(model)

  result <- loss_fun(
    model,
    c(1L, 2L, 3L),
    c(2L, 3L, 4L),
    clip_value = 5
  )
  updated <- adagrad_update(
    model,
    result$grads,
    state,
    learning_rate = 0.1
  )

  changed <- vapply(
    c("Wxh", "Whh", "Why", "bh", "by"),
    function(name) !isTRUE(all.equal(original[[name]], updated$model[[name]])),
    logical(1L)
  )
  stopifnot(all(changed))
})

run_test("sampling returns valid indices", function() {
  set.seed(5)
  model <- initialise_model(vocab_size = 4L, hidden_size = 3L)
  ids <- sample_indices(
    model,
    h = zero_hidden_state(model),
    seed_ix = 1L,
    n = 25L
  )

  stopifnot(length(ids) == 25L)
  stopifnot(all(ids >= 1L & ids <= 4L))
})

run_test("save/load model and vocabulary", function() {
  set.seed(6)
  model <- initialise_model(vocab_size = 4L, hidden_size = 3L)
  vocab <- build_vocab("abcdabcd")

  model_path <- tempfile(fileext = ".rds")
  vocab_path <- tempfile(fileext = ".rds")
  on.exit(unlink(c(model_path, vocab_path)), add = TRUE)

  save_model(model, model_path)
  save_vocab(vocab, vocab_path)

  restored_model <- load_model(model_path)
  restored_vocab <- load_vocab(vocab_path)

  stopifnot(identical(model, restored_model))
  stopifnot(identical(vocab, restored_vocab))
})

cat("\nAll tests passed.\n")
