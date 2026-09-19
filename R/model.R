# min-char-rnn: vanilla character-level RNN in base R
# Educational R translation of the mathematics in Andrej Karpathy's min-char-rnn.
#
# Parameter shapes:
#   Wxh: hidden_size x vocab_size
#   Whh: hidden_size x hidden_size
#   Why: vocab_size x hidden_size
#   bh:  hidden_size x 1
#   by:  vocab_size x 1

stable_softmax <- function(y) {
  stopifnot(is.matrix(y), ncol(y) == 1L)
  shifted <- y - max(y)
  exp_shifted <- exp(shifted)
  exp_shifted / sum(exp_shifted)
}

initialise_model <- function(vocab_size, hidden_size, weight_sd = 0.01) {
  stopifnot(vocab_size >= 1L, hidden_size >= 1L, weight_sd > 0)

  list(
    Wxh = matrix(
      rnorm(hidden_size * vocab_size, mean = 0, sd = weight_sd),
      nrow = hidden_size,
      ncol = vocab_size
    ),
    Whh = matrix(
      rnorm(hidden_size * hidden_size, mean = 0, sd = weight_sd),
      nrow = hidden_size,
      ncol = hidden_size
    ),
    Why = matrix(
      rnorm(vocab_size * hidden_size, mean = 0, sd = weight_sd),
      nrow = vocab_size,
      ncol = hidden_size
    ),
    bh = matrix(0, nrow = hidden_size, ncol = 1L),
    by = matrix(0, nrow = vocab_size, ncol = 1L)
  )
}

model_parameter_count <- function(model) {
  sum(vapply(model[c("Wxh", "Whh", "Why", "bh", "by")], length, integer(1L)))
}

zero_hidden_state <- function(model) {
  matrix(0, nrow = nrow(model$Whh), ncol = 1L)
}

one_hot <- function(index, vocab_size) {
  if (length(index) != 1L || index < 1L || index > vocab_size) {
    stop("Character index is outside the one-based vocabulary range.")
  }

  x <- matrix(0, nrow = vocab_size, ncol = 1L)
  x[index, 1L] <- 1
  x
}

sequence_loss <- function(model, inputs, targets, hprev = zero_hidden_state(model)) {
  if (length(inputs) != length(targets) || length(inputs) < 1L) {
    stop("inputs and targets must have the same positive length.")
  }

  vocab_size <- nrow(model$Why)
  h <- hprev
  loss <- 0

  for (t in seq_along(inputs)) {
    x <- one_hot(inputs[t], vocab_size)

    # h_t = tanh(Wxh x_t + Whh h_(t-1) + bh)
    h <- tanh(model$Wxh %*% x + model$Whh %*% h + model$bh)

    # y_t = Why h_t + by
    y <- model$Why %*% h + model$by

    # Stable cross-entropy:
    # -log softmax(y)[target] = log(sum(exp(y))) - y[target].
    y_max <- max(y)
    log_sum_exp <- y_max + log(sum(exp(y - y_max)))
    loss <- loss + log_sum_exp - y[targets[t], 1L]
  }

  list(loss = as.numeric(loss), last_h = h)
}

loss_fun <- function(model,
                     inputs,
                     targets,
                     hprev = zero_hidden_state(model),
                     clip_value = 5) {
  if (length(inputs) != length(targets) || length(inputs) < 1L) {
    stop("inputs and targets must have the same positive length.")
  }

  vocab_size <- nrow(model$Why)
  time_steps <- length(inputs)

  xs <- vector("list", time_steps)
  hs <- vector("list", time_steps + 1L)
  ys <- vector("list", time_steps)
  ps <- vector("list", time_steps)

  hs[[1L]] <- hprev
  loss <- 0

  # Forward pass.
  for (t in seq_len(time_steps)) {
    xs[[t]] <- one_hot(inputs[t], vocab_size)

    hs[[t + 1L]] <- tanh(
      model$Wxh %*% xs[[t]] +
        model$Whh %*% hs[[t]] +
        model$bh
    )

    ys[[t]] <- model$Why %*% hs[[t + 1L]] + model$by
    ps[[t]] <- stable_softmax(ys[[t]])

    y_max <- max(ys[[t]])
    log_sum_exp <- y_max + log(sum(exp(ys[[t]] - y_max)))
    loss <- loss + log_sum_exp - ys[[t]][targets[t], 1L]
  }

  # Backward pass through time.
  grads <- list(
    Wxh = matrix(0, nrow = nrow(model$Wxh), ncol = ncol(model$Wxh)),
    Whh = matrix(0, nrow = nrow(model$Whh), ncol = ncol(model$Whh)),
    Why = matrix(0, nrow = nrow(model$Why), ncol = ncol(model$Why)),
    bh = matrix(0, nrow = nrow(model$bh), ncol = 1L),
    by = matrix(0, nrow = nrow(model$by), ncol = 1L)
  )

  dhnext <- matrix(0, nrow = nrow(model$Whh), ncol = 1L)

  for (t in seq.int(time_steps, 1L)) {
    # For softmax + cross-entropy, dL/dy = p - one_hot(target).
    dy <- ps[[t]]
    dy[targets[t], 1L] <- dy[targets[t], 1L] - 1

    grads$Why <- grads$Why + dy %*% t(hs[[t + 1L]])
    grads$by <- grads$by + dy

    dh <- t(model$Why) %*% dy + dhnext
    dhraw <- (1 - hs[[t + 1L]]^2) * dh

    grads$bh <- grads$bh + dhraw
    grads$Wxh <- grads$Wxh + dhraw %*% t(xs[[t]])
    grads$Whh <- grads$Whh + dhraw %*% t(hs[[t]])

    dhnext <- t(model$Whh) %*% dhraw
  }

  # Karpathy's original element-wise gradient clipping.
  if (is.finite(clip_value)) {
    for (name in names(grads)) {
      gradient <- grads[[name]]
      gradient[gradient > clip_value] <- clip_value
      gradient[gradient < -clip_value] <- -clip_value
      grads[[name]] <- gradient
    }
  }

  list(
    loss = as.numeric(loss),
    grads = grads,
    last_h = hs[[time_steps + 1L]]
  )
}

new_adagrad_state <- function(model) {
  lapply(model[c("Wxh", "Whh", "Why", "bh", "by")], function(x) {
    matrix(0, nrow = nrow(x), ncol = ncol(x))
  })
}

adagrad_update <- function(model, grads, state, learning_rate, epsilon = 1e-8) {
  stopifnot(learning_rate > 0, epsilon > 0)
  parameter_names <- c("Wxh", "Whh", "Why", "bh", "by")

  for (name in parameter_names) {
    state[[name]] <- state[[name]] + grads[[name]]^2
    model[[name]] <- model[[name]] -
      learning_rate * grads[[name]] / sqrt(state[[name]] + epsilon)
  }

  list(model = model, state = state)
}

sample_indices <- function(model,
                           h,
                           seed_ix,
                           n,
                           temperature = 1.0) {
  if (n < 0L || temperature <= 0) {
    stop("n must be non-negative and temperature must be positive.")
  }

  vocab_size <- nrow(model$Why)
  x <- one_hot(seed_ix, vocab_size)
  indices <- integer(n)

  if (n == 0L) {
    return(indices)
  }

  for (t in seq_len(n)) {
    h <- tanh(model$Wxh %*% x + model$Whh %*% h + model$bh)
    y <- (model$Why %*% h + model$by) / temperature
    p <- stable_softmax(y)

    ix <- sample.int(vocab_size, size = 1L, prob = as.numeric(p))
    indices[t] <- ix

    x <- one_hot(ix, vocab_size)
  }

  indices
}

assert_model_finite <- function(model) {
  parameter_names <- c("Wxh", "Whh", "Why", "bh", "by")
  ok <- all(vapply(parameter_names, function(name) {
    all(is.finite(model[[name]]))
  }, logical(1L)))

  if (!ok) {
    stop("Model contains a non-finite parameter.")
  }

  invisible(TRUE)
}

save_model <- function(model, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  saveRDS(model, path)
  invisible(path)
}

load_model <- function(path) {
  model <- readRDS(path)
  assert_model_finite(model)
  model
}

gradient_check <- function(model,
                           inputs,
                           targets,
                           hprev = zero_hidden_state(model),
                           epsilon = 1e-5,
                           max_checks_per_parameter = Inf) {
  if (epsilon <= 0) {
    stop("epsilon must be positive.")
  }

  analytic <- loss_fun(
    model,
    inputs,
    targets,
    hprev = hprev,
    clip_value = Inf
  )$grads

  rows <- list()
  row_id <- 1L

  for (name in c("Wxh", "Whh", "Why", "bh", "by")) {
    n_values <- length(model[[name]])

    if (is.finite(max_checks_per_parameter) &&
        max_checks_per_parameter < n_values) {
      check_indices <- unique(as.integer(round(seq(
        1,
        n_values,
        length.out = max_checks_per_parameter
      ))))
    } else {
      check_indices <- seq_len(n_values)
    }

    for (index in check_indices) {
      plus <- model
      minus <- model

      plus[[name]][index] <- plus[[name]][index] + epsilon
      minus[[name]][index] <- minus[[name]][index] - epsilon

      loss_plus <- sequence_loss(plus, inputs, targets, hprev)$loss
      loss_minus <- sequence_loss(minus, inputs, targets, hprev)$loss
      numerical <- (loss_plus - loss_minus) / (2 * epsilon)
      analytical <- analytic[[name]][index]

      denominator <- max(1e-8, abs(numerical) + abs(analytical))
      relative_error <- abs(numerical - analytical) / denominator

      rows[[row_id]] <- data.frame(
        parameter = name,
        index = index,
        analytical = analytical,
        numerical = numerical,
        relative_error = relative_error,
        stringsAsFactors = FALSE
      )
      row_id <- row_id + 1L
    }
  }

  do.call(rbind, rows)
}

