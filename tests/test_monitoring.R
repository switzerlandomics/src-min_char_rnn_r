#!/usr/bin/env Rscript
# Run from the project root: Rscript tests/test_monitoring.R
# No model training is performed by these tests.
args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script <- if (length(file_arg)) sub("^--file=", "", file_arg[1L]) else "tests/test_monitoring.R"
root <- normalizePath(file.path(dirname(normalizePath(script, mustWork = TRUE)), ".."), mustWork = TRUE)
source(file.path(root, "R", "plots.R"))

experiment_dir <- tempfile("monitor-test-")
dir.create(experiment_dir)
metrics <- data.frame(
  iteration = c(0L, 100L, 200L, 300L),
  train_loss_nats_per_char = c(NA_real_, 3.4, 2.9, 2.7),
  validation_loss_nats_per_char = c(4.17, 3.5, 3.0, 3.1),
  elapsed_seconds = c(0, 2, 4, 6),
  iterations_per_second = c(NA, 50, 50, 50),
  epochs = c(0, 0.2, 0.4, 0.6)
)
utils::write.csv(metrics, file.path(experiment_dir, "metrics.csv"), row.names = FALSE)
saveRDS(list(uniform_baseline_nats_per_char = 4.17),
        file.path(experiment_dir, "metadata.rds"))
loaded <- read_experiment_metrics(experiment_dir)
stopifnot(nrow(loaded) == 4L,
          best_validation_row(loaded)$iteration == 200L,
          identical(names(loaded), names(metrics)))

# Historical metrics (without the newer epochs column) remain plottable.
utils::write.csv(metrics[, names(metrics) != "epochs"],
                 file.path(experiment_dir, "metrics.csv"), row.names = FALSE)
stopifnot(is.null(best_validation_row(data.frame(
  iteration = 1L, validation_loss_nats_per_char = NA_real_
))))
if (requireNamespace("ggplot2", quietly = TRUE)) {
  result <- plot_experiment(experiment_dir)
  stopifnot(result$best_iteration == 200L,
            length(result$files) == 3L,
            all(file.exists(result$files)),
            all(file.info(result$files)$size > 0))
  message("PASS: metrics parsing, best checkpoint, historical compatibility, PNG and live HTML")
} else {
  message("PASS: metrics parsing and historical compatibility (plot rendering skipped: install ggplot2)")
}
unlink(experiment_dir, recursive = TRUE)
