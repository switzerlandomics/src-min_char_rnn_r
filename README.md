# min-char-rnn in base R

A small CPU-only educational implementation of Andrej Karpathy's minimal
character-level vanilla RNN. The training code uses base R matrix operations
only: no torch, TensorFlow, Keras, Python bridge, GPU code, or automatic
differentiation.

The project keeps Karpathy's core mathematics and parameter names:

- `Wxh`: input-to-hidden weights
- `Whh`: hidden-to-hidden recurrent weights
- `Why`: hidden-to-output weights
- `bh`: hidden bias
- `by`: output bias

It adds deterministic one-based R vocabulary indexing, a numerically stable
softmax/cross-entropy calculation, a held-out validation split, concise logging,
timing, periodic samples, checkpoint files, and finite-difference gradient
tests.

## Quick test

These three scripts test and run the project.

```sh
Rscript tests/test_model.R
Rscript experiments/run.R --smoke
Rscript experiments/run.R
```

## Project layout

| Path | Purpose |
|---|---|
| `R/model.R` | Forward pass, BPTT, clipping, AdaGrad, sampling, gradient checker |
| `R/data.R` | UTF-8 input, vocabulary, one-based encoding, split, checksum |
| `R/progress.R` | Logging, elapsed time, ETA, progress bar |
| `experiments/run.R` | CLI wrapper and complete training experiment |
| `tests/test_model.R` | Base R tests, including finite-difference gradients |
| `data/input.txt` | Tiny bundled corpus for smoke tests |
| `scripts/create_zip.R` | Rebuild the distributable zip |
| `README.md` | Usage and design notes |

## Requirements

- R with `Rscript` available on the command line
- Base R and the standard `tools` package
- CPU only
- Internet access only if the default Tiny Shakespeare file has not yet been
  downloaded

No R packages need to be installed.

## Mathematical model

For a one-hot character vector \(x_t\) and previous hidden state \(h_{t-1}\):

```text
h_t = tanh(Wxh x_t + Whh h_(t-1) + bh)
y_t = Why h_t + by
p_t = softmax(y_t)
loss = -sum_t log p_t[target_t]
```

Backpropagation through time computes gradients for all five learned parameter
objects. Each gradient element is clipped to `[-5, 5]`, matching the original
minimal implementation. Parameters are updated with AdaGrad:

```text
memory = memory + gradient^2
parameter = parameter - learning_rate * gradient / sqrt(memory + 1e-8)
```

R arrays are one-based, so vocabulary indices are `1:vocab_size`. The mapping is
sorted before training so it is deterministic.

## Default configuration

| Setting | Default |
|---|---:|
| Hidden size | 100 |
| Sequence length | 25 |
| Learning rate | 0.1 |
| Iterations | 5,000 |
| Random seed | 42 |
| Log interval | 100 |
| Sample interval | 500 |
| Sample length | 200 |
| Validation fraction | 0.10 |
| Validation characters per evaluation | 2,000 |
| Gradient clipping | `[-5, 5]` |
| Weight initialisation SD | 0.01 |

The default input is `data/tiny_shakespeare.txt`. If it is absent, the runner
downloads the Tiny Shakespeare corpus from Karpathy's `char-rnn` repository.
The exact downloaded file is recorded by MD5 checksum in the experiment log and
metadata.

## Run the tests

From the project root:

```sh
Rscript tests/test_model.R
```

The test suite checks:

1. text encoding and decoding
2. parameter dimensions
3. numerical stability of softmax
4. forward and backward output dimensions
5. analytical gradients against central finite differences
6. AdaGrad parameter updates
7. valid sampling
8. model and vocabulary save/load round trips

A gradient-check failure exits with a non-zero status.

## Run the bundled smoke test

```sh
Rscript experiments/run.R --smoke
```

Smoke mode uses `data/input.txt`, a hidden size of 16, sequence length 12, and
30 updates. It exercises the full pipeline without downloading a dataset.

## Run Tiny Shakespeare

```sh
Rscript experiments/run.R
```

If `data/tiny_shakespeare.txt` is missing, it is downloaded automatically using
base R's `download.file()`.

To provide your own corpus:

```sh
Rscript experiments/run.R \
  --input=data/my_input.txt \
  --hidden-size=100 \
  --seq-length=25 \
  --lr=0.1 \
  --iterations=5000 \
  --seed=42 \
  --log-interval=100 \
  --sample-interval=500 \
  --sample-length=200
```

The input file must be valid UTF-8 plain text. Spaces, punctuation, case, and
newlines are preserved as learnable characters.

## Output

Each run creates its own timestamped directory under `output/`:

```text
output/20260919_143000_seed42/
├── experiment.log
├── metadata.rds
├── metrics.csv
├── model.rds
├── samples.txt
└── vocab.rds
```

`experiment.log` records R version, seed, input checksum, configuration, data
sizes, model size, periodic training and validation loss, progress, ETA, and
saved paths.

`samples.txt` stores generated text at iteration 0 and each sample interval,
together with the corresponding validation loss.

`model.rds` and `vocab.rds` can be restored with:

```r
source("R/model.R")
source("R/data.R")

model <- load_model("output/.../model.rds")
vocab <- load_vocab("output/.../vocab.rds")
```

## Reading the reported losses

Training and validation losses are reported in nats per character. A uniform
predictor over a vocabulary of size `V` has loss:

```text
log(V)
```

The training display uses Karpathy's exponentially smoothed sequence loss,
divided by sequence length so that it is expressed per character.

Periodic validation uses a fixed prefix of the held-out validation split. This
keeps validation useful and deterministic without making validation much more
expensive than training. `--validation-chars=N` controls its size.

## Runtime expectations

Runtime depends strongly on CPU, R build, and BLAS implementation. For a modern
single-core laptop, sensible planning ranges are:

| Run | Approximate planning range |
|---|---|
| Tests | under a few seconds |
| `--smoke` | a few seconds |
| 5,000 iterations, hidden size 100 | roughly 1 to 10 minutes |
| 20,000 iterations, hidden size 100 | roughly 5 to 40 minutes |

These are planning ranges, not benchmarks. Use the live iteration rate and ETA
printed by the runner as the measurement for your machine.

## Reproducibility

The runner records:

- `R.version.string`
- RNG kind
- explicit seed
- input path
- MD5 input checksum
- all experiment settings
- vocabulary size
- parameter count
- initial and final validation loss
- elapsed time

For the same R version, RNG behaviour, input bytes, seed, and configuration, the
run is designed to be reproducible. Sampling uses temporary deterministic seeds
that are restored afterwards, so periodic text generation does not alter model
training.

## Create a zip

```sh
Rscript scripts/create_zip.R
```

This writes `min-char-rnn.zip` next to the project directory. The packaging
script uses R's `utils::zip()` and therefore expects a system `zip` executable.

Equivalent shell command from the directory containing `min-char-rnn/`:

```sh
zip -r min-char-rnn.zip min-char-rnn \
  -x 'min-char-rnn/output/*' \
     'min-char-rnn/data/tiny_shakespeare.txt'
```


---


# Monitoring and resumable training

This update adds live learning-curve images and complete resumable checkpoints to
an existing `min-char-rnn` project. **Keep your existing `R/model.R`, `R/data.R`,
`R/progress.R`, `tests/test_model.R` and training data**. The update does not
replace or reimplement the working network mathematics.

## Install

Copy the supplied files into the corresponding paths at your project root.
`experiments/run.R` replaces the previous runner; the other files are new.

Plotting uses ggplot2 3.4 or later. The neural network is still written in base R.

```r
install.packages("ggplot2")
```

If you cannot install ggplot2, add `--no-plot` to run training without graphics.

## Start a run

```sh
Rscript tests/test_model.R
Rscript tests/test_monitoring.R
Rscript experiments/run.R --smoke
Rscript experiments/run.R --iterations=50000
```

The 50,000-iteration setting is a **total** training budget, not a recommended
minimum or an assurance that text quality will improve throughout training.
The log prints the experiment directory at startup. Open its `training.html`
file in a browser while training continues. It refreshes every 10 seconds and
shows the newest `training.png` and `validation_detail.png`. The images are
updated after each `--plot-interval` checkpoint (default: every 1,000 updates).
No browser or graphics display is required for training to continue.

To render plots retrospectively:

```sh
Rscript experiments/plot_results.R output/YOUR_EXPERIMENT_DIRECTORY
```

To monitor an experiment started with `--no-plot` using a separate process:

```sh
Rscript experiments/plot_results.R output/YOUR_EXPERIMENT_DIRECTORY --watch
```

`--watch` polls the CSV every 10 seconds by default. When monitoring a run
without a `FINISHED` marker (for example, an old experiment), press Ctrl+C to
exit the viewer. The viewer **does not** control or stop training.

## Stop and resume safely

To request a clean stop, open a second terminal in the project root and create
an empty marker file inside the active run's output directory:

```sh
touch output/YOUR_EXPERIMENT_DIRECTORY/STOP
```

The runner checks for the marker at the next console/validation interval,
validates and saves the current state, updates plots, then exits. This is not
an immediate operating-system interrupt. **Do not use Ctrl+C as a substitute**
for clean stop when you need a recent resumable checkpoint.

Remove the marker and resume to a new *total iteration target*:

```sh
rm output/YOUR_EXPERIMENT_DIRECTORY/STOP
Rscript experiments/run.R --resume=output/YOUR_EXPERIMENT_DIRECTORY --iterations=100000
```

On Windows, create or delete the `STOP` file with your file manager or the
PowerShell commands `New-Item` and `Remove-Item`.

Changing the input, vocabulary, architecture, learning rate or sequence length
while resuming is intentionally disallowed. You can change monitoring intervals
and increase the total iteration target. Resume expects
`latest_checkpoint.rds`, not the old `model.rds` file, and verifies the input
file checksum. **Pre-update experiments cannot be resumed**, because their
saved `model.rds` does not include AdaGrad accumulators, hidden state, cursor or
RNG state.

## Output files

Each new experiment directory contains:

- `metrics.csv`: measured losses, elapsed training seconds and approximate
  corpus passes; written **during** training at validation intervals.
- `training.png`: full training and validation learning curves, 10 x 6 inches,
  120 dpi, ggplot2 `theme_bw()` and legible default-size typography.
- `validation_detail.png`: final third of validation checkpoints, 10 x 4.9 inches, 120 dpi.
- `training.html`: lightweight local viewer, reloading every 10 seconds.
- `best_model.rds`: lowest measured validation loss so far, including iteration 0.
- `model.rds`: model weights at the end of the latest run.
- `latest_checkpoint.rds`: complete state for resuming training, including
  AdaGrad accumulators, text cursor, recurrent state, RNG state and metrics.
- `vocab.rds`, `samples.txt`, `experiment.log`, `metadata.rds`.
- `FINISHED`: `completed` or `stopped` after a clean exit.

A training image is a monitor, not a control surface. The `STOP` file is the
explicit control mechanism. A plot being open or closed has no effect on the
training process.

## What the lines mean

Training loss is an exponentially smoothed, recent-mini-sequence estimate.
Validation loss is measured on the same fixed held-out prefix, from a reset
hidden state at each evaluation. The default validation window is the **first
2,000 held-out transitions**, not the complete held-out split. The displayed
training and validation losses are therefore not identically sampled and their
gap must be interpreted with care.

`epochs` is an *approximate* count of corpus passes: completed iterations times
sequence length divided by train-set character transitions. The final short
remainder at the end of each pass is skipped, as in the original loop.

The light-grey dashed horizontal line is the uniform-character cross-entropy
baseline, `log(vocabulary size)`. It is not a target the network should be
expected to reach. The bigram reference is an add-one-smoothed next-character
model fitted on training text alone; it is logged numerically and is evaluated
on exactly the same held-out transitions as the RNN.

An isolated validation increase is normal. Assess the trend over multiple
checkpoints, compare generated samples using a constant sampling RNG seed, and
keep the best-validation weights. Improving per-character loss does not by
itself guarantee long-range coherent prose.

## General design

Plot rendering failures are logged without aborting the training loop; the metrics
and resumable checkpoint remain the sources of truth.

The plotting functions accept generic `metrics.csv` columns and metadata; they
contain no hard-coded dataset name, training duration or output experiment ID.
The image size provides space for normal-sized type, while restrained colour,
a light background, minimal gridlines and an explicit best-checkpoint marker
prioritise information over ornament. All plots can be recreated later from the
saved files, without retraining.

## Caveats

The integration assumes the existing public functions and return structures
shown in your supplied `experiments/run.R`: `initialise_model`, `loss_fun`,
`adagrad_update`, `sequence_loss`, `sample_indices`, `save_model`, `save_vocab`,
`make_logger`, `format_duration`, `read_text_file`, `build_vocab`, `encode_text`,
`decode_indices`, `split_sequence`, `input_checksum` and related helpers.

The three original implementation modules were not provided with the runner, so
this update deliberately preserves them. Run your existing `test_model.R` and
the short smoke test before beginning a long experiment.



## Reference

The implementation follows the equations and training logic of Andrej
Karpathy's `min-char-rnn.py` Gist and the accompanying 2015 article, *The
Unreasonable Effectiveness of Recurrent Neural Networks*. Karpathy's minimal
script uses one-hot character inputs, a tanh recurrent state, softmax
cross-entropy, backpropagation through time, element-wise gradient clipping at
5, AdaGrad, and periodic autoregressive sampling.

Tiny Shakespeare is the example corpus distributed in Karpathy's `char-rnn`
repository.

