#!/usr/bin/env Rscript
# Read-only plot generation for completed or in-progress experiments.
args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script <- if (length(file_arg)) sub("^--file=", "", file_arg[1L]) else "experiments/plot_results.R"
root <- normalizePath(file.path(dirname(normalizePath(script, mustWork = TRUE)), ".."), mustWork = TRUE)
source(file.path(root, "R", "plots.R"))

argv <- commandArgs(trailingOnly = TRUE)
if (!length(argv) || any(argv %in% c("--help", "-h"))) {
  cat("Usage: Rscript experiments/plot_results.R EXPERIMENT_DIR [--watch] [--refresh=SECONDS]\n",
      "Generates training.png, validation_detail.png and training.html.\n",
      "--watch polls for changes and refreshes plots until FINISHED appears.\n",
      "In a browser, open training.html for automatic 10-second refreshes.\n", sep = "")
  quit(status = if (!length(argv)) 1L else 0L)
}
experiment_dir <- normalizePath(argv[1L], mustWork = TRUE)
watch <- "--watch" %in% argv
refresh_arg <- grep("^--refresh=", argv, value = TRUE)
refresh <- if (length(refresh_arg)) as.numeric(sub("^--refresh=", "", refresh_arg[1L])) else 10
if (!is.finite(refresh) || refresh < 1) stop("--refresh must be at least 1 second.")
if (length(setdiff(argv[-1L], c("--watch", refresh_arg)))) stop("Unknown option.")

render <- function() {
  result <- plot_experiment(experiment_dir)
  cat(sprintf("[PLOT] best validation %.6f at iteration %d | %s\n",
              result$best_validation_loss, result$best_iteration,
              file.path(experiment_dir, "training.html")))
}
render()
if (watch) {
  path <- file.path(experiment_dir, "metrics.csv")
  previous <- file.info(path)$mtime
  repeat {
    if (file.exists(file.path(experiment_dir, "FINISHED"))) break
    Sys.sleep(refresh)
    current <- file.info(path)$mtime
    if (!is.na(current) && !identical(current, previous)) {
      tryCatch(render(), error = function(e) message("[PLOT] ", conditionMessage(e)))
      previous <- current
    }
  }
  cat("[PLOT] Experiment finished.\n")
}
