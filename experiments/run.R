#!/usr/bin/env Rscript
# An experiment wrapper for the minimal RNN. All neural-network mathematics
# remains in R/model.R; this file handles configuration, I/O and monitoring.

script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (!length(file_arg)) return(normalizePath("experiments/run.R", mustWork = FALSE))
  normalizePath(sub("^--file=", "", file_arg[1L]), mustWork = FALSE)
}
PROJECT_ROOT <- normalizePath(file.path(dirname(script_path()), ".."), mustWork = TRUE)
setwd(PROJECT_ROOT)
source(file.path("R", "model.R"))
source(file.path("R", "data.R"))
source(file.path("R", "progress.R"))
source(file.path("R", "plots.R"))

REFERENCE_INPUT_URL <- paste0(
  "https://raw.githubusercontent.com/karpathy/char-rnn/",
  "master/data/tinyshakespeare/input.txt"
)

# Intervals are measured in parameter updates; timings are measured in seconds.
default_config <- function() {
  list(input = file.path("data", "tiny_shakespeare.txt"), output = "output",
       hidden_size = 100L, seq_length = 25L, lr = 0.1,
       iterations = 50000L, seed = 42L, log_interval = 500L,
       validation_interval = 1000L, plot_interval = 1000L,
       sample_interval = 5000L, sample_length = 200L,
       checkpoint_interval = 5000L, validation_fraction = 0.10,
       validation_chars = 2000L, plot = TRUE, smoke = FALSE, resume = "")
}

print_help <- function() {
  cat(paste(c(
    "min-char-rnn: CPU-only, manually differentiated character RNN",
    "Usage: Rscript experiments/run.R [options]",
    "", "--input=PATH                 Plain text corpus",
    "--output=DIR                 Experiment output parent directory",
    "--iterations=N               Total target updates, including resumed updates",
    "--hidden-size=N              Hidden units (new runs only)",
    "--seq-length=N               Unroll length (new runs only)",
    "--lr=NUMBER                 AdaGrad learning rate (new runs only)",
    "--seed=N                     RNG seed (new runs only)",
    "--log-interval=N             Console progress frequency",
    "--validation-interval=N      Validation and metrics frequency",
    "--plot-interval=N            Redraw image frequency; multiple of validation interval",
    "--sample-interval=N          Generated text frequency",
    "--sample-length=N            Generated character count",
    "--checkpoint-interval=N      Save complete training state frequency",
    "--validation-fraction=NUMBER Contiguous held-out fraction",
    "--validation-chars=N         Fixed held-out transitions evaluated per checkpoint",
    "--no-plot                    Disable ggplot2 graphics (training still works)",
    "--plot                       Enable graphics (default)",
    "--resume=DIR                 Continue from DIR/latest_checkpoint.rds",
    "--smoke                      Short run using bundled data/input.txt",
    "--help", "",
    "During training, create a file named STOP in the experiment directory to",
    "stop cleanly at the next progress check. Remove STOP before resuming.",
    "The existing model.rds is final weights, not a resumable checkpoint."
  ), collapse = "\n"), "\n")
}

parse_args <- function(args, config) {
  supplied <- character()
  for (arg in args) {
    if (arg == "--help") { print_help(); quit(status = 0L) }
    if (arg %in% c("--smoke", "--no-plot", "--plot")) {
      key <- if (arg == "--smoke") "smoke" else "plot"
      config[[key]] <- arg != "--no-plot"
      supplied <- c(supplied, key)
      next
    }
    parts <- regmatches(arg, regexec("^--([^=]+)=(.*)$", arg))[[1L]]
    if (length(parts) != 3L) stop("Unknown argument: ", arg)
    key <- gsub("-", "_", parts[2L], fixed = TRUE)
    if (!key %in% names(config) || key %in% c("smoke", "plot")) {
      stop("Unknown option: --", parts[2L])
    }
    current <- config[[key]]
    value <- parts[3L]
    if (is.integer(current)) {
      value <- suppressWarnings(as.integer(value))
      if (is.na(value)) stop(arg, " requires an integer.")
    } else if (is.numeric(current)) {
      value <- suppressWarnings(as.numeric(value))
      if (!is.finite(value)) stop(arg, " requires a finite number.")
    }
    config[[key]] <- value
    supplied <- c(supplied, key)
  }
  list(config = config, supplied = unique(supplied))
}

apply_smoke_defaults <- function(config, supplied) {
  if (!isTRUE(config$smoke)) return(config)
  smoke <- list(input = file.path("data", "input.txt"), hidden_size = 16L,
                seq_length = 12L, iterations = 30L, log_interval = 10L,
                validation_interval = 10L, plot_interval = 10L,
                checkpoint_interval = 15L, sample_interval = 15L,
                sample_length = 120L, validation_chars = 120L)
  for (key in names(smoke)) if (!key %in% supplied) config[[key]] <- smoke[[key]]
  config
}

validate_config <- function(config) {
  keys <- c("hidden_size", "seq_length", "iterations", "log_interval",
            "validation_interval", "plot_interval", "checkpoint_interval",
            "sample_interval", "sample_length", "validation_chars")
  for (key in keys) if (is.na(config[[key]]) || config[[key]] < 1L) {
    stop(key, " must be a positive integer.")
  }
  if (!is.finite(config$lr) || config$lr <= 0) stop("lr must be positive.")
  if (!(config$validation_fraction > 0 && config$validation_fraction < 0.5)) {
    stop("validation_fraction must be strictly between 0 and 0.5.")
  }
  if (config$plot_interval %% config$validation_interval != 0L) {
    stop("plot_interval must be a multiple of validation_interval.")
  }
  invisible(TRUE)
}

ensure_input_file <- function(config, supplied) {
  if (file.exists(config$input)) return(invisible(config$input))
  default <- file.path("data", "tiny_shakespeare.txt")
  if (!"input" %in% supplied && identical(config$input, default)) {
    dir.create(dirname(default), recursive = TRUE, showWarnings = FALSE)
    message("[setup] Downloading the reference training corpus ...")
    status <- tryCatch(utils::download.file(REFERENCE_INPUT_URL, default,
                                            mode = "wb", quiet = TRUE),
                       error = function(e) stop("Download failed: ", conditionMessage(e)))
    if (status != 0L || !file.exists(default)) stop("Could not download corpus.")
    return(invisible(default))
  }
  stop("Input file does not exist: ", config$input)
}

# Evaluate exactly the same contiguous held-out prefix at every checkpoint.
# The RNN is reset once at the validation boundary; no gradient is calculated.
evaluate_validation <- function(model, validation_ids, max_chars) {
  n <- min(length(validation_ids) - 1L, max_chars)
  result <- sequence_loss(model, validation_ids[seq_len(n)],
                          validation_ids[seq.int(2L, n + 1L)],
                          hprev = zero_hidden_state(model))
  result$loss / n
}

# A simple add-one-smoothed bigram baseline, fitted on training text only.
# This is evaluated on the same validation transitions as the RNN.
bigram_baseline <- function(training_ids, validation_ids, vocab_size, max_chars) {
  n <- length(training_ids) - 1L
  prev <- training_ids[seq_len(n)]
  next_ix <- training_ids[seq.int(2L, n + 1L)]
  counts <- matrix(tabulate((prev - 1L) * vocab_size + next_ix,
                            nbins = vocab_size * vocab_size),
                   nrow = vocab_size, byrow = TRUE)
  totals <- rowSums(counts)
  k <- min(length(validation_ids) - 1L, max_chars)
  prev <- validation_ids[seq_len(k)]
  next_ix <- validation_ids[seq.int(2L, k + 1L)]
  -mean(log((counts[cbind(prev, next_ix)] + 1) / (totals[prev] + vocab_size)))
}

sample_deterministically <- function(model, vocab, seed_ix, n, seed) {
  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had_seed) old_seed <- get(".Random.seed", envir = .GlobalEnv)
  on.exit({
    if (had_seed) assign(".Random.seed", old_seed, envir = .GlobalEnv)
    else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)
  set.seed(seed)
  decode_indices(sample_indices(model, h = zero_hidden_state(model),
                                seed_ix = seed_ix, n = n), vocab)
}

append_sample <- function(path, iteration, val_loss, seed_char, text) {
  cat(sprintf("\n=== iteration %d | validation %.6f | seed %s ===\n",
              iteration, val_loss, encodeString(seed_char, quote = '"')),
      text, "\n", file = path, append = TRUE, sep = "")
}

# Write to a sibling temporary file. Rename is atomic on normal local POSIX
# filesystems, so readers need not see a half-written CSV or checkpoint.
atomic_write <- function(path, writer) {
  tmp <- tempfile(".writing-", tmpdir = dirname(path))
  on.exit(if (file.exists(tmp)) unlink(tmp), add = TRUE)
  writer(tmp)
  # First attempt an atomic replacement. Windows may require unlinking the
  # destination before rename; fall back only when the first attempt fails.
  replaced <- file.rename(tmp, path)
  if (!replaced) {
    if (file.exists(path)) unlink(path)
    replaced <- file.rename(tmp, path)
  }
  if (!replaced && !file.copy(tmp, path, overwrite = TRUE)) {
    stop("Could not write: ", path)
  }
  invisible(path)
}
write_rds <- function(object, path) atomic_write(path, function(tmp) saveRDS(object, tmp))
write_metrics <- function(metrics, path) {
  atomic_write(path, function(tmp) utils::write.csv(metrics, tmp, row.names = FALSE))
}

make_metrics_row <- function(iteration, train_loss, val_loss, elapsed, train_size, seq_length) {
  data.frame(iteration = as.integer(iteration),
             train_loss_nats_per_char = as.numeric(train_loss),
             validation_loss_nats_per_char = as.numeric(val_loss),
             elapsed_seconds = as.numeric(elapsed),
             iterations_per_second = if (elapsed > 0) iteration / elapsed else NA_real_,
             epochs = (as.double(iteration) * seq_length) / max(1, train_size - 1L))
}

status_line <- function(iteration, target, elapsed, active_elapsed, active_updates) {
  fraction <- min(iteration / target, 1)
  filled <- floor(24 * fraction)
  bar <- paste0("[", paste0(rep("=", filled), collapse = ""),
                paste0(rep(".", 24 - filled), collapse = ""), "]")
  eta <- if (active_updates > 0 && active_elapsed > 0) {
    (target - iteration) / (active_updates / active_elapsed)
  } else NA_real_
  sprintf("%s %5.1f%% | %d/%d | elapsed %s | ETA %s",
          bar, 100 * fraction, iteration, target, format_duration(elapsed),
          if (is.finite(eta)) format_duration(eta) else "calculating")
}

unique_experiment_dir <- function(output, seed) {
  stamp <- paste0(format(Sys.time(), "%Y%m%d_%H%M%S"), "_seed", seed)
  candidate <- file.path(output, stamp)
  suffix <- 1L
  while (file.exists(candidate)) {
    candidate <- file.path(output, paste0(stamp, "_", suffix))
    suffix <- suffix + 1L
  }
  dir.create(candidate, recursive = TRUE, showWarnings = FALSE)
  candidate
}

parsed <- parse_args(commandArgs(trailingOnly = TRUE), default_config())
config <- apply_smoke_defaults(parsed$config, parsed$supplied)
resume <- nzchar(config$resume)
checkpoint <- NULL
if (resume) {
  checkpoint_path <- if (dir.exists(config$resume)) {
    file.path(config$resume, "latest_checkpoint.rds")
  } else config$resume
  if (!file.exists(checkpoint_path)) stop("Checkpoint not found: ", checkpoint_path)
  checkpoint <- readRDS(checkpoint_path)
  if (!identical(checkpoint$version, 1L)) stop("Unsupported checkpoint version.")
  original <- checkpoint$config
  allowed <- c("iterations", "log_interval", "validation_interval", "plot_interval",
               "sample_interval", "sample_length", "checkpoint_interval", "plot", "resume")
  prohibited <- setdiff(parsed$supplied, allowed)
  if (length(prohibited)) stop("Do not change these settings on resume: ",
                               paste(prohibited, collapse = ", "))
  for (key in intersect(parsed$supplied, allowed)) original[[key]] <- config[[key]]
  config <- original
  config$resume <- checkpoint_path
  if (config$iterations < checkpoint$iteration) {
    stop("iterations must be at least the checkpoint's completed iteration.")
  }
}
validate_config(config)
ensure_input_file(config, if (resume) "input" else parsed$supplied)
if (isTRUE(config$plot) && !requireNamespace("ggplot2", quietly = TRUE)) {
  stop("Live plots require ggplot2. Run install.packages('ggplot2') or pass --no-plot.")
}

if (resume) {
  experiment_dir <- dirname(normalizePath(checkpoint_path, mustWork = TRUE))
  if (file.exists(file.path(experiment_dir, "STOP"))) {
    stop("Remove ", file.path(experiment_dir, "STOP"), " before resuming.")
  }
  # A previous clean exit leaves FINISHED; clear it before the next run so
  # external --watch viewers do not interpret resumed training as completed.
  if (file.exists(file.path(experiment_dir, "FINISHED"))) {
    unlink(file.path(experiment_dir, "FINISHED"))
  }
} else {
  set.seed(config$seed)
  experiment_dir <- unique_experiment_dir(config$output, config$seed)
}
path <- function(name) file.path(experiment_dir, name)
log_msg <- make_logger(path("experiment.log"))
run_start <- Sys.time()
log_msg("START", "%s | base R | CPU-only | %s", "min-char-rnn",
        if (resume) "resuming" else "new experiment")
log_msg("CONFIG", "R=%s | input=%s | md5=%s", R.version.string,
        config$input, input_checksum(config$input))
log_msg("CONFIG", "hidden=%d | seq=%d | lr=%.6g | target=%d | seed=%d | plot=%s",
        config$hidden_size, config$seq_length, config$lr,
        config$iterations, config$seed, config$plot)
log_msg("STAGE", "[1/5] Preparing data")
text <- read_text_file(config$input)
vocab <- build_vocab(text)
ids <- encode_text(text, vocab)
splits <- split_sequence(ids, validation_fraction = config$validation_fraction,
                         seq_length = config$seq_length)
train_size <- length(splits$train)
validation_size <- length(splits$validation)
validation_n <- min(validation_size - 1L, config$validation_chars)
log_msg("DATA", "characters=%d | vocabulary=%d | train=%d | validation=%d",
        length(ids), vocab$size, train_size, validation_size)
log_msg("DATA", "fixed validation window=%d transitions | approximate updates/pass=%.0f",
        validation_n, (train_size - 1) / config$seq_length)

if (resume) {
  if (!identical(checkpoint$input_md5, input_checksum(config$input)) ||
      !identical(checkpoint$vocab_chars, vocab$chars) ||
      checkpoint$train_size != train_size) {
    stop("Input data or vocabulary differs from the saved training checkpoint.")
  }
  model <- checkpoint$model
  state <- checkpoint$adagrad_state
  hprev <- checkpoint$hprev
  pointer <- checkpoint$pointer
  smooth_loss <- checkpoint$smooth_loss
  iteration_start <- checkpoint$iteration
  elapsed_before <- checkpoint$elapsed_training_seconds
  metrics <- checkpoint$metrics
  best_loss <- checkpoint$best_validation_loss
  best_iteration <- checkpoint$best_iteration
  initial_loss <- checkpoint$initial_validation_loss
  uniform_baseline <- checkpoint$uniform_baseline
  bigram_loss <- checkpoint$bigram_loss
  last_validation <- tail(metrics$validation_loss_nats_per_char, 1L)
  assign(".Random.seed", checkpoint$rng_state, envir = .GlobalEnv)
  log_msg("RESUME", "loaded iteration %d | previous training time %s",
          iteration_start, format_duration(elapsed_before))
} else {
  log_msg("STAGE", "[2/5] Initialising model")
  model <- initialise_model(vocab$size, config$hidden_size)
  state <- new_adagrad_state(model)
  hprev <- zero_hidden_state(model)
  pointer <- 1L
  iteration_start <- 0L
  elapsed_before <- 0
  smooth_loss <- log(vocab$size) * config$seq_length
  uniform_baseline <- log(vocab$size)
  initial_loss <- evaluate_validation(model, splits$validation, config$validation_chars)
  bigram_loss <- bigram_baseline(splits$train, splits$validation,
                                vocab$size, config$validation_chars)
  best_loss <- initial_loss
  best_iteration <- 0L
  last_validation <- initial_loss
  metrics <- make_metrics_row(0L, NA_real_, initial_loss, 0, train_size, config$seq_length)
  write_metrics(metrics, path("metrics.csv"))
  save_model(model, path("best_model.rds"))
  save_vocab(vocab, path("vocab.rds"))
  cat("", file = path("samples.txt"))
  log_msg("MODEL", "Vanilla RNN | parameters=%d | optimiser=AdaGrad | clip=[-5,5]",
          model_parameter_count(model))
  log_msg("MODEL", "uniform=%.6f | bigram=%.6f | initial val=%.6f nats/char",
          uniform_baseline, bigram_loss, initial_loss)
  initial_sample <- sample_deterministically(model, vocab, splits$train[1L],
                                             config$sample_length, config$seed + 100000L)
  append_sample(path("samples.txt"), 0L, initial_loss,
                decode_indices(splits$train[1L], vocab), initial_sample)
}

metadata <- if (resume && file.exists(path("metadata.rds"))) readRDS(path("metadata.rds")) else list()
if (is.null(metadata$created_at)) metadata$created_at <- as.character(Sys.time())
metadata$experiment_id <- basename(experiment_dir)
metadata$R_version <- R.version.string
metadata$RNG_kind <- RNGkind()
metadata$seed <- config$seed
metadata$input_path <- config$input
metadata$input_md5 <- input_checksum(config$input)
metadata$config <- config
metadata$vocabulary_size <- vocab$size
metadata$train_characters <- train_size
metadata$validation_characters <- validation_size
metadata$validation_transitions_evaluated <- validation_n
metadata$parameter_count <- model_parameter_count(model)
metadata$uniform_baseline_nats_per_char <- uniform_baseline
metadata$bigram_baseline_nats_per_char <- bigram_loss
metadata$initial_validation_loss_nats_per_char <- initial_loss
metadata$best_validation_loss_nats_per_char <- best_loss
metadata$best_iteration <- best_iteration
write_rds(metadata, path("metadata.rds"))

render_plots <- function() {
  if (!isTRUE(config$plot)) return(invisible(FALSE))
  # Plotting is an optional observer. Rendering errors should never destroy
  # a long-running training experiment or its resumable checkpoint.
  tryCatch({
    plot_experiment(experiment_dir)
    TRUE
  }, error = function(e) {
    log_msg("PLOT", "render failed: %s", conditionMessage(e))
    FALSE
  })
}
if (isTRUE(config$plot) && isTRUE(render_plots())) {
  log_msg("PLOT", "open %s in your browser to monitor training",
          normalizePath(path("training.html"), winslash = "/", mustWork = FALSE))
}

training_start <- Sys.time()
last_completed <- iteration_start
stopped_by_request <- FALSE

persist <- function(iteration, elapsed, save_checkpoint = FALSE, update_plots = FALSE) {
  write_metrics(metrics, path("metrics.csv"))
  metadata$best_validation_loss_nats_per_char <<- best_loss
  metadata$best_iteration <<- best_iteration
  metadata$completed_iterations <<- iteration
  metadata$training_elapsed_seconds <<- elapsed
  metadata$updated_at <<- as.character(Sys.time())
  write_rds(metadata, path("metadata.rds"))
  if (save_checkpoint) {
    # RNG state, AdaGrad accumulators and the recurrent state are all needed
    # for a faithful continuation; model.rds alone is not sufficient.
    snapshot <- list(version = 1L, config = config, iteration = as.integer(iteration),
                     model = model, adagrad_state = state, hprev = hprev,
                     pointer = pointer, smooth_loss = smooth_loss,
                     metrics = metrics, best_validation_loss = best_loss,
                     best_iteration = best_iteration, initial_validation_loss = initial_loss,
                     uniform_baseline = uniform_baseline, bigram_loss = bigram_loss,
                     elapsed_training_seconds = elapsed, rng_state = .Random.seed,
                     input_md5 = input_checksum(config$input),
                     vocab_chars = vocab$chars, train_size = train_size)
    write_rds(snapshot, path("latest_checkpoint.rds"))
  }
  if (update_plots) render_plots()
  invisible(NULL)
}

if (!resume) persist(0L, 0, save_checkpoint = TRUE, update_plots = FALSE)
log_msg("STAGE", "[3/5] Training | progress every %d | validation every %d | checkpoint every %d",
        config$log_interval, config$validation_interval, config$checkpoint_interval)

if (iteration_start < config$iterations) {
  for (iteration in seq.int(iteration_start + 1L, config$iterations)) {
    if (pointer + config$seq_length > train_size) {
      pointer <- 1L
      hprev <- zero_hidden_state(model)
    }
    inputs <- splits$train[seq.int(pointer, pointer + config$seq_length - 1L)]
    targets <- splits$train[seq.int(pointer + 1L, pointer + config$seq_length)]
    step <- loss_fun(model, inputs, targets, hprev = hprev, clip_value = 5)
    if (!is.finite(step$loss)) stop("Non-finite loss at iteration ", iteration)
    if (!all(vapply(step$grads, function(g) all(is.finite(g)), logical(1L)))) {
      stop("Non-finite gradient at iteration ", iteration)
    }
    updated <- adagrad_update(model, step$grads, state, learning_rate = config$lr)
    model <- updated$model
    state <- updated$state
    hprev <- step$last_h
    smooth_loss <- 0.999 * smooth_loss + 0.001 * step$loss
    pointer <- pointer + config$seq_length
    last_completed <- iteration

    is_final <- iteration == config$iterations
    log_due <- iteration %% config$log_interval == 0L || is_final
    validation_due <- iteration %% config$validation_interval == 0L || is_final
    sample_due <- iteration %% config$sample_interval == 0L || is_final
    checkpoint_due <- iteration %% config$checkpoint_interval == 0L || is_final
    stop_requested <- (log_due || validation_due) && file.exists(path("STOP"))
    if (stop_requested) validation_due <- TRUE
    elapsed_active <- as.numeric(difftime(Sys.time(), training_start, units = "secs"))
    elapsed <- elapsed_before + elapsed_active

    if (validation_due) {
      assert_model_finite(model)
      last_validation <- evaluate_validation(model, splits$validation, config$validation_chars)
      if (!is.finite(last_validation)) stop("Non-finite validation loss at iteration ", iteration)
      metrics <- rbind(metrics, make_metrics_row(iteration,
                      smooth_loss / config$seq_length, last_validation,
                      elapsed, train_size, config$seq_length))
      if (last_validation < best_loss) {
        best_loss <- last_validation
        best_iteration <- iteration
        save_model(model, path("best_model.rds"))
        log_msg("BEST", "iteration=%d | validation=%.6f nats/char", iteration, best_loss)
      }
    }
    if (log_due || stop_requested) {
      speed <- (iteration - iteration_start) / max(elapsed_active, 1e-9)
      log_msg("TRAIN", "%s | train=%.6f | val(last)=%.6f | %.1f iter/s",
              status_line(iteration, config$iterations, elapsed,
                          elapsed_active, iteration - iteration_start),
              smooth_loss / config$seq_length, last_validation, speed)
    }
    if (sample_due || stop_requested) {
      sample <- sample_deterministically(model, vocab, splits$train[1L],
                                          config$sample_length, config$seed + 100000L)
      append_sample(path("samples.txt"), iteration, last_validation,
                    decode_indices(splits$train[1L], vocab), sample)
      log_msg("SAMPLE", "iteration=%d | validation=%.6f\n%s", iteration,
              last_validation, sample)
    }
    if (validation_due || checkpoint_due || stop_requested) {
      persist(iteration, elapsed,
              save_checkpoint = checkpoint_due || stop_requested,
              update_plots = (iteration %% config$plot_interval == 0L ||
                              is_final || stop_requested))
    }
    if (stop_requested) {
      stopped_by_request <- TRUE
      log_msg("STOP", "STOP marker found; saved checkpoint at iteration %d", iteration)
      break
    }
  }
}

log_msg("STAGE", "[4/5] Final evaluation")
final_validation <- evaluate_validation(model, splits$validation, config$validation_chars)
if (!is.finite(final_validation)) stop("Non-finite final validation loss.")
if (final_validation < best_loss) {
  best_loss <- final_validation
  best_iteration <- last_completed
  save_model(model, path("best_model.rds"))
}
log_msg("EVAL", "initial=%.6f | current=%.6f | best=%.6f at %d | bigram=%.6f nats/char",
        initial_loss, final_validation, best_loss, best_iteration, bigram_loss)
log_msg("STAGE", "[5/5] Saving results")
save_model(model, path("model.rds"))
save_vocab(vocab, path("vocab.rds"))
elapsed <- elapsed_before + as.numeric(difftime(Sys.time(), training_start, units = "secs"))
metadata$final_validation_loss_nats_per_char <- final_validation
metadata$total_elapsed_seconds <- elapsed
metadata$status <- if (stopped_by_request) "stopped" else "completed"
metadata$completed_iterations <- last_completed
persist(last_completed, elapsed, save_checkpoint = TRUE, update_plots = TRUE)
writeLines(metadata$status, path("FINISHED"))
log_msg("SAVE", "model=%s | best=%s | checkpoint=%s", path("model.rds"),
        path("best_model.rds"), path("latest_checkpoint.rds"))
log_msg("SAVE", "metrics=%s | plots=%s | samples=%s", path("metrics.csv"),
        if (isTRUE(config$plot)) path("training.html") else "disabled (--no-plot)",
        path("samples.txt"))
log_msg("DONE", "status=%s | iterations=%d | total training time %s",
        metadata$status, last_completed, format_duration(elapsed))
