# What happened during the 50,000-iteration run?

Your graph reveals something important: the model continued learning the training text, but its ability to predict the held-out validation text became unstable. The sharp jumps are much more informative than the final loss alone.

Best validation loss

# 2.246

At 41,000 updates

Final validation loss

# 3.314

At 50,000 updates

The final model predicts the measured validation text substantially less accurately than the best checkpoint. The training curve, meanwhile, remains near 2.0 nats per character.

## 1. Reading the graph

0–29,000 updates

The model learns and generalises increasingly well.

Training loss falls from approximately 4.2 to 2.0. Validation loss also falls, reaching approximately 2.3. This is the behaviour we hoped to observe: the network is learning patterns that help it predict both its training text and unseen text.

Around 30,000 updates

Validation loss suddenly jumps from approximately 2.3 to 3.6.

Training loss does not show a corresponding increase. The model remains effective at predicting the recent training text, but its predictions on the validation passage deteriorate sharply.

Around 41,000 updates

Validation recovers and reaches its best recorded value of 2.246.

This is particularly interesting. The earlier deterioration was not necessarily permanent: subsequent training produced parameters that predicted the validation passage well again.

42,000–50,000 updates

Validation deteriorates again while training loss remains relatively low.

The run ends with validation loss at 3.314. The lowest-loss checkpoint at 41,000 updates is therefore more informative than the final model when assessing this run's best observed validation performance.

This is not a simple case of training loss and validation loss gradually separating. There are two distinct, abrupt changes in validation performance, followed by substantially different behaviour in the training curve.

## 2. Why does validation jump so dramatically?

There are several plausible explanations, and the graph alone cannot establish which is responsible.

### A. The model encounters different sections of the training text

Your training process reads consecutive 25-character sequences from the corpus rather than randomly mixing the text at every update.

With approximately one million characters in the training split, the first complete pass occurs near 40,000 updates.

That means the first validation jump occurs when the model has processed roughly 750,000 training-character positions. The sharp recovery near 41,000 updates occurs shortly after it returns to the beginning of the training corpus.

This timing is worth investigating.

A small RNN trained sequentially can adjust strongly to patterns in the section of text it has encountered most recently. If those patterns differ from the validation passage, its validation performance may deteriorate even as its recent training loss improves.

However, the alignment with the corpus boundary is a clue, not proof of the cause. In particular, the second jump soon after 41,000 updates requires further investigation.

### B. The parameter updates may be too aggressive

Your model uses AdaGrad with an initial learning rate of 0.1.

Even when training loss looks stable, a sequence of parameter updates could substantially alter the probability distributions assigned to the validation text.

The unusually sharp loss changes suggest examining parameter-update behaviour rather than assuming that simply increasing training duration will solve the problem.

Potential diagnostics include the gradient norm before clipping, the size of parameter updates and the fraction of gradient elements that require clipping.

I would not change the optimiser yet. We should first establish whether the changes are caused by the training sequence, optimisation instability or an evaluation issue.

### C. We are measuring only a small part of the validation corpus

This is an important limitation of our current experiment.

Your runner evaluates only the first 2,000 character transitions of the held-out validation sequence, rather than the entire validation split.

Pasted text.txt

Pasted text.txt

Consequently, the graph tells us how well the model predicts that particular validation passage.

A change in the model's predictions for relatively common characters within that passage could have a considerable effect on its average loss.

The measurements are useful because they use a fixed passage, but they may not accurately represent the model's performance across all the held-out text.

My first diagnostic would be to evaluate several fixed passages distributed throughout the validation corpus. If they all show the same jumps, the deterioration is more general. If only the first passage shows them, the current validation curve is giving us an incomplete picture.

## 3. Is this overfitting?

There is a pronounced generalisation gap at the end of the run, but I would not attribute the entire pattern to conventional overfitting.

Ordinary overfitting often produces a gradual pattern: training loss continues to decline while validation loss reaches a minimum and then progressively increases.

Your curve instead shows a sharp deterioration, a substantial recovery and another sharp deterioration.

That pattern is consistent with unstable generalisation, potentially influenced by the sequential training data and optimisation behaviour. It also warrants checking the evaluation implementation.

We should first confirm that the validation dataset, preprocessing, hidden-state initialisation and loss calculation remain identical at every checkpoint.

## 4. Would longer training produce better results?

Possibly, but I would not continue this particular run for another 100 minutes without first understanding the instability.

The recovery at 41,000 updates demonstrates that additional training can recover good validation performance after a deterioration. It does not establish that the model will eventually converge to a stable, lower validation loss.

Continuing from the final checkpoint might produce another recovery. It might also produce the same oscillating behaviour for many more corpus passes.

The more useful next experiment is to change one factor at a time while preserving the existing results as a baseline.

## Three controlled experiments

Experiment 1

Repeat the current run with broader validation.

Keep the architecture, learning rate and training sequence unchanged. Evaluate several fixed validation passages and record their individual losses alongside the aggregate.

This tells us whether the jumps affect the wider validation corpus or primarily the first 2,000-character passage.

Experiment 2

Compare a smaller learning rate.

Repeat the experiment with a learning rate such as 0.03, using the same data, initialisation seed and architecture.

Compare the validation curves at equivalent iteration counts and corpus passes. A smaller learning rate may make updates less disruptive, although it could also require more iterations to reach a similar loss.

Experiment 3

Investigate the order of training sequences.

Compare the existing sequential traversal against a version that samples training windows from different positions in the training split.

This would test whether the large changes are associated with the model adapting to consecutive sections of the corpus. It is a change to the training procedure, so I would implement it as an optional experiment rather than silently altering our faithful Karpathy-style baseline.

## 5. Can we get substantially better generated text on this CPU?

There is room to improve, but we should distinguish improvements to training from improvements to the architecture.

Your present network has 100 hidden neurons and uses a vanilla recurrent hidden-state update. Training it longer can help it learn more from the corpus, provided optimisation remains stable.

However, it has limited capacity to preserve useful context over long passages. Even an effectively trained model of this size may produce plausible words and short phrases without maintaining grammatical or narrative consistency.

I would establish stable validation performance before increasing the hidden size or sequence length. Increasing the hidden size also makes the recurrent matrix larger and increases computation per update, so we should measure the benefit rather than assume that a much larger network is preferable.

A later LSTM implementation would be a separate, informative experiment because its gated memory is designed to help retain information over longer sequences. It would not be a like-for-like reproduction of our current vanilla RNN.

## 6. What I would check immediately

Before launching another long run, I would verify the behaviour of the existing saved models.

Your monitoring update was designed to save a separate best-validation model. Check whether your experiment directory contains `best_model.rds`.

If it does, we should compare the generated text from that model with the final `model.rds`, using the same seed character, sampling temperature and generation random seed.

We should also reevaluate both models on several fixed held-out passages. That will tell us whether the dramatic change is reproducible in the actual saved parameters.

The most informative next result

A plot showing validation loss on several fixed held-out passages, with markers at 30,000, 41,000 and 42,000 updates. This would make it much easier to distinguish a local validation-passage effect from a broader failure to generalise.

My conclusion from this run: the network has demonstrated genuine learning, but the current training and evaluation setup produces highly variable held-out performance. The best checkpoint is at 41,000 updates, not at the end. Before committing to longer training or a larger architecture, I would improve validation coverage and investigate the sharp transitions around 30,000 and 42,000 updates.

