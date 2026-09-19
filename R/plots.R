# Monitoring functions. Training itself requires only base R; plotting uses ggplot2.
# The functions consume generic experiment metrics, not a particular corpus.

read_experiment_metrics <- function(experiment_dir) {
  path <- file.path(experiment_dir, "metrics.csv")
  if (!file.exists(path)) stop("No metrics.csv found in: ", experiment_dir)
  metrics <- utils::read.csv(path, stringsAsFactors = FALSE)
  required <- c("iteration", "train_loss_nats_per_char", "validation_loss_nats_per_char")
  if (!all(required %in% names(metrics))) {
    stop("metrics.csv is missing required columns: ", paste(setdiff(required, names(metrics)), collapse = ", "))
  }
  metrics <- metrics[order(metrics$iteration), , drop = FALSE]
  metrics <- metrics[!duplicated(metrics$iteration, fromLast = TRUE), , drop = FALSE]
  if (!nrow(metrics)) stop("metrics.csv has no observations.")
  metrics
}

read_experiment_metadata <- function(experiment_dir) {
  path <- file.path(experiment_dir, "metadata.rds")
  if (!file.exists(path)) return(list())
  readRDS(path)
}

best_validation_row <- function(metrics) {
  valid <- which(is.finite(metrics$validation_loss_nats_per_char))
  if (!length(valid)) return(NULL)
  metrics[valid[which.min(metrics$validation_loss_nats_per_char[valid])], , drop = FALSE]
}

# A 10 x 6 inch canvas at 120 dpi yields a roughly 1,200 px wide image.
# At the monitor's 1,100 px CSS width, 13 pt labels remain legible without
# oversized text or a giant blank canvas. All metrics are in nats/character.
plot_style <- function(base_size = 13) {
  ggplot2::theme_bw(base_size = base_size) +
    ggplot2::theme(
      panel.grid.minor = ggplot2::element_blank(),
      panel.grid.major.x = ggplot2::element_blank(),
      panel.grid.major.y = ggplot2::element_line(colour = "#E7EBEE", linewidth = 0.28),
      panel.border = ggplot2::element_rect(colour = "#C9D0D5", linewidth = 0.45),
      plot.title.position = "plot",
      plot.title = ggplot2::element_text(face = "bold", colour = "#23323C"),
      plot.subtitle = ggplot2::element_text(colour = "#50626E", margin = ggplot2::margin(b = 10)),
      plot.caption = ggplot2::element_text(hjust = 0, colour = "#657780", size = 10),
      plot.caption.position = "plot",
      axis.text = ggplot2::element_text(colour = "#455760"),
      axis.title = ggplot2::element_text(colour = "#23323C"),
      legend.position = "top",
      legend.justification = "left",
      legend.title = ggplot2::element_blank(),
      legend.text = ggplot2::element_text(colour = "#344751"),
      plot.margin = ggplot2::margin(14, 20, 14, 15)
    )
}

axis_number <- function(x) format(x, big.mark = ",", trim = TRUE, scientific = FALSE)

plot_learning_curves <- function(metrics, metadata = list()) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Plotting requires ggplot2: install.packages('ggplot2')")
  }
  train <- metrics[is.finite(metrics$train_loss_nats_per_char), , drop = FALSE]
  val <- metrics[is.finite(metrics$validation_loss_nats_per_char), , drop = FALSE]
  plot_data <- rbind(
    data.frame(iteration = train$iteration, loss = train$train_loss_nats_per_char,
               series = "Training (smoothed)"),
    data.frame(iteration = val$iteration, loss = val$validation_loss_nats_per_char,
               series = "Validation (held out)")
  )
  if (!nrow(plot_data) || !nrow(val)) stop("No finite validation loss values in metrics.csv.")
  plot_data$series <- factor(plot_data$series,
                             levels = c("Training (smoothed)", "Validation (held out)"))
  best <- best_validation_row(metrics)
  current <- val[nrow(val), , drop = FALSE]
  last <- metrics[nrow(metrics), , drop = FALSE]
  passes <- if ("epochs" %in% names(last) && is.finite(last$epochs)) {
    sprintf("  |  %.2f approximate corpus passes", last$epochs)
  } else ""
  subtitle <- sprintf("%s updates%s  |  current validation %.3f nats/character",
                      axis_number(last$iteration), passes,
                      current$validation_loss_nats_per_char)
  p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = iteration, y = loss, colour = series)) +
    ggplot2::scale_colour_manual(
      values = c("Training (smoothed)" = "#677883",
                 "Validation (held out)" = "#087A92"), drop = FALSE
    ) +
    ggplot2::scale_x_continuous(labels = axis_number,
                                 expand = ggplot2::expansion(mult = c(0.012, 0.035))) +
    ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0.06, 0.13))) +
    ggplot2::labs(
      title = "Training and validation",
      subtitle = subtitle,
      x = "Parameter updates", y = "Cross-entropy loss (nats / character)",
      caption = paste("Training = exponentially smoothed batch loss;",
                      "validation = fixed held-out sequence. Lower is better.")
    ) + plot_style()
  # At iteration zero there is just one validation point: do not try to draw a
  # line through a single observation (which produces a ggplot2 warning).
  if (nrow(val) > 1L) {
    p <- p + ggplot2::geom_line(linewidth = 0.9, na.rm = TRUE)
  }
  p <- p + ggplot2::geom_point(
    data = val, ggplot2::aes(x = iteration, y = validation_loss_nats_per_char),
    inherit.aes = FALSE, colour = "#087A92", size = 1.2
  )
  uniform <- metadata$uniform_baseline_nats_per_char
  if (length(uniform) == 1L && is.finite(uniform)) {
    p <- p + ggplot2::geom_hline(yintercept = uniform, colour = "#B6BEC3",
                                 linetype = "dashed", linewidth = 0.45)
    p <- p + ggplot2::labs(
      caption = paste("Dashed line = uniform predictor. Training = smoothed batch loss;",
                      "validation = fixed held-out sequence. Lower is better.")
    )
  }
  if (!is.null(best)) {
    best_point <- data.frame(iteration = best$iteration,
                             loss = best$validation_loss_nats_per_char)
    p <- p + ggplot2::geom_point(
      data = best_point, ggplot2::aes(x = iteration, y = loss),
      inherit.aes = FALSE, colour = "#B54E32", size = 2.8
    ) + ggplot2::labs(
      caption = sprintf("Orange point = best validation %.3f at %s updates. %s",
                        best$validation_loss_nats_per_char,
                        axis_number(best$iteration),
                        "Training is smoothed; validation uses fixed held-out text.")
    )
  }
  p
}

plot_validation_detail <- function(metrics) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Plotting requires ggplot2: install.packages('ggplot2')")
  }
  val <- metrics[is.finite(metrics$validation_loss_nats_per_char), , drop = FALSE]
  if (!nrow(val)) stop("No finite validation loss values in metrics.csv.")
  best <- best_validation_row(metrics)
  # Recent history is a focused view: keep at least 20 checks, or the final
  # third of a long run. Short experiments show every checkpoint.
  n_recent <- min(nrow(val), max(20L, ceiling(nrow(val) / 3)))
  recent <- utils::tail(val, n_recent)
  recent$best_so_far <- cummin(val$validation_loss_nats_per_char)[
    seq.int(nrow(val) - n_recent + 1L, nrow(val))
  ]
  recent_long <- rbind(
    data.frame(iteration = recent$iteration,
               loss = recent$validation_loss_nats_per_char,
               series = "Validation"),
    data.frame(iteration = recent$iteration, loss = recent$best_so_far,
               series = "Best so far")
  )
  recent_long$series <- factor(recent_long$series,
                               levels = c("Validation", "Best so far"))
  subtitle <- sprintf("Last %s checks  |  global best %.3f at %s updates",
                      axis_number(n_recent), best$validation_loss_nats_per_char,
                      axis_number(best$iteration))
  p <- ggplot2::ggplot(recent_long,
                       ggplot2::aes(x = iteration, y = loss, colour = series)) +
    ggplot2::geom_point(data = recent, mapping = ggplot2::aes(
      x = iteration, y = validation_loss_nats_per_char),
      inherit.aes = FALSE, colour = "#087A92", size = 1.15) +
    ggplot2::scale_colour_manual(
      values = c("Validation" = "#087A92", "Best so far" = "#B54E32"),
      drop = FALSE
    ) +
    ggplot2::scale_x_continuous(labels = axis_number,
                                 expand = ggplot2::expansion(mult = c(0.015, 0.04))) +
    ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0.13, 0.16))) +
    ggplot2::labs(
      title = "Recent validation performance", subtitle = subtitle,
      x = "Parameter updates", y = "Validation loss (nats / character)",
      caption = "Orange = lowest measured validation loss so far. Look for trends across multiple checks."
    ) + plot_style()
  if (nrow(recent) > 1L) p <- p + ggplot2::geom_line(linewidth = 0.9)
  if (best$iteration %in% recent$iteration) {
    p <- p + ggplot2::geom_point(
      data = best, ggplot2::aes(x = iteration, y = validation_loss_nats_per_char),
      inherit.aes = FALSE, colour = "#B54E32", size = 2.9
    )
  }
  p
}

write_png_safely <- function(plot, path, width, height, dpi = 150) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  tmp <- tempfile(pattern = ".plot-", tmpdir = dirname(path), fileext = ".png")
  on.exit(if (file.exists(tmp)) unlink(tmp), add = TRUE)
  ggplot2::ggsave(filename = tmp, plot = plot, width = width, height = height,
                  units = "in", dpi = dpi, bg = "white", limitsize = TRUE)
  # On Windows, renaming over an existing file may fail; fall back to overwrite-copy.
  replaced <- file.rename(tmp, path)
  if (!replaced) {
    if (file.exists(path)) unlink(path)  # Windows-compatible fallback.
    replaced <- file.rename(tmp, path)
  }
  if (!replaced && !file.copy(tmp, path, overwrite = TRUE)) {
    stop("Could not write plot: ", path)
  }
  invisible(path)
}

plot_experiment <- function(experiment_dir) {
  metrics <- read_experiment_metrics(experiment_dir)
  metadata <- read_experiment_metadata(experiment_dir)
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Install ggplot2 for plotting: install.packages('ggplot2')")
  }
  write_png_safely(plot_learning_curves(metrics, metadata),
                   file.path(experiment_dir, "training.png"), width = 10, height = 6, dpi = 120)
  write_png_safely(plot_validation_detail(metrics),
                   file.path(experiment_dir, "validation_detail.png"), width = 10, height = 4.9, dpi = 120)
  # Browsers opened on this local HTML page automatically refresh both images.
  refresh_key <- format(Sys.time(), "%Y%m%d%H%M%S")
  html <- c(
    '<!doctype html><html lang="en"><head><meta charset="utf-8">',
    '<meta http-equiv="refresh" content="10">',
    '<meta name="viewport" content="width=device-width, initial-scale=1">',
    '<title>Training monitor</title>',
    '<style>body{font:16px system-ui,sans-serif;color:#25313a;background:#fff;',
    'max-width:1100px;margin:26px auto;padding:0 18px}h1{font-size:1.4rem}',
    'img{width:100%;height:auto;display:block;margin:12px 0 24px}',
    'small{color:#64717a}</style></head><body>',
    '<h1>Training monitor</h1><small>Refreshes every 10 seconds when this page remains open.</small>',
    sprintf('<img src="training.png?v=%s" alt="Training and validation learning curves">', refresh_key),
    sprintf('<img src="validation_detail.png?v=%s" alt="Recent validation loss">', refresh_key),
    '</body></html>'
  )
  writeLines(html, file.path(experiment_dir, "training.html"), useBytes = TRUE)
  best <- best_validation_row(metrics)
  invisible(list(best_iteration = if (is.null(best)) NA_integer_ else best$iteration,
                 best_validation_loss = if (is.null(best)) NA_real_ else best$validation_loss_nats_per_char,
                 files = file.path(experiment_dir,
                                   c("training.png", "validation_detail.png", "training.html"))))
}
