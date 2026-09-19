# Human-readable logging, timers, and progress display.

format_duration <- function(seconds) {
  seconds <- max(0, as.numeric(seconds))
  hours <- floor(seconds / 3600)
  minutes <- floor((seconds %% 3600) / 60)
  secs <- floor(seconds %% 60)

  if (hours > 0) {
    sprintf("%02d:%02d:%02d", hours, minutes, secs)
  } else {
    sprintf("%02d:%02d", minutes, secs)
  }
}

make_logger <- function(log_path) {
  dir.create(dirname(log_path), recursive = TRUE, showWarnings = FALSE)
  cat("", file = log_path)

  function(level, fmt, ...) {
    message_text <- if (length(list(...)) > 0L) {
      sprintf(fmt, ...)
    } else {
      fmt
    }

    line <- sprintf(
      "%s | %-8s | %s",
      format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
      level,
      message_text
    )

    cat(line, "\n", sep = "")
    cat(line, "\n", sep = "", file = log_path, append = TRUE)
    invisible(line)
  }
}

progress_line <- function(current, total, start_time, width = 24L) {
  if (total < 1L || current < 0L || current > total) {
    stop("Invalid progress values.")
  }

  elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  fraction <- if (total == 0L) 1 else current / total
  filled <- floor(width * fraction)

  bar <- paste0(
    "[",
    paste(rep("=", filled), collapse = ""),
    paste(rep(".", width - filled), collapse = ""),
    "]"
  )

  eta <- if (current > 0L) {
    elapsed * (total - current) / current
  } else {
    NA_real_
  }

  sprintf(
    "%s %5.1f%% | %d/%d | elapsed %s | ETA %s",
    bar,
    100 * fraction,
    current,
    total,
    format_duration(elapsed),
    if (is.finite(eta)) format_duration(eta) else "--:--"
  )
}
