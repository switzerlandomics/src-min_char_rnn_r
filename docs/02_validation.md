# How validation works

A character-level recurrent neural network (RNN) learns to predict the next character in a sequence. During training, it reads text one character at a time, predicts which character comes next, and compares its prediction with the actual next character. It then adjusts its model parameters to improve future predictions.

For example, given the training text `The king`, the model receives the following inputs and targets:

| Input | Correct next character |
| ----- | ---------------------- |
| `T`   | `h`                    |
| `h`   | `e`                    |
| `e`   | space                  |
| space | `k`                    |
| `k`   | `i`                    |
| `i`   | `n`                    |
| `n`   | `g`                    |

For each input, the model assigns a probability to every possible next character. If the actual next character is `h` and the model assigns it a probability of 0.8, the loss for that prediction is `-log(0.8)`. A higher probability for the correct character produces a lower loss.

The RNN does not consider each character in isolation. It carries a hidden state through the sequence, allowing its predictions to depend on preceding characters as well as the current input. Consequently, it can predict different continuations for the same character in different contexts.

## Training versus validation

The dataset is divided into **training text** and **held-out validation text**. Both are evaluated using the same next-character prediction task, but only training updates the model parameters.

During validation, the model reads a passage it was not trained on. It predicts the next character at each position, compares its probability with the actual character, and calculates the average loss across the evaluated sequence. The actual character is then supplied as the next input, so each prediction is assessed against the real text rather than against the model's previously generated output.

Validation loss is reported in **nats per character**. A lower value means that, on average, the model assigned higher probabilities to the correct next characters in the held-out text. Comparing training and validation loss helps us judge whether the model is learning patterns that also apply to unseen text.

## Validation versus text generation

Validation always has a known correct continuation against which predictions can be measured. During text generation, there is no supplied correct continuation: the model starts with a seed character, samples a character from its predicted probabilities, and feeds that generated character back into the network to produce the next one.

**Validation measures next-character prediction on held-out text; generation shows what happens when the model must produce its own continuation.**

