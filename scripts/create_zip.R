#!/usr/bin/env Rscript

script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)

  if (length(file_arg) == 0L) {
    return(normalizePath("scripts/create_zip.R", mustWork = FALSE))
  }

  normalizePath(sub("^--file=", "", file_arg[1L]), mustWork = FALSE)
}

project_root <- normalizePath(
  file.path(dirname(script_path()), ".."),
  mustWork = TRUE
)
project_name <- basename(project_root)
parent_dir <- dirname(project_root)
zip_path <- file.path(parent_dir, paste0(project_name, ".zip"))

zip_command <- Sys.which("zip")
if (!nzchar(zip_command)) {
  stop(
    paste(
      "A system 'zip' executable is required by utils::zip().",
      "Install zip, then rerun this script."
    )
  )
}

relative_files <- c(
  file.path(project_name, "R", "model.R"),
  file.path(project_name, "R", "data.R"),
  file.path(project_name, "R", "progress.R"),
  file.path(project_name, "experiments", "run.R"),
  file.path(project_name, "tests", "test_model.R"),
  file.path(project_name, "scripts", "create_zip.R"),
  file.path(project_name, "data", "input.txt"),
  file.path(project_name, "README.md")
)

missing <- relative_files[!file.exists(file.path(parent_dir, relative_files))]
if (length(missing) > 0L) {
  stop(sprintf(
    "Cannot create archive; missing files: %s",
    paste(missing, collapse = ", ")
  ))
}

if (file.exists(zip_path)) {
  unlink(zip_path)
}

old_wd <- setwd(parent_dir)
on.exit(setwd(old_wd), add = TRUE)

utils::zip(
  zipfile = zip_path,
  files = relative_files,
  flags = "-9X"
)

cat(sprintf("Created: %s\n", normalizePath(zip_path, mustWork = TRUE)))
