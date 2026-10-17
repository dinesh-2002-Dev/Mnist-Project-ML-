# MNIST Controlled Experiments (PyTorch)

Trains a small convolutional neural network (CNN) on the MNIST handwritten-digit dataset
and compares two optimizers (SGD with momentum and Adam) at different learning rates.
Random seeds are fixed and each setting is run 3 times (seeds 0, 1, 2) so the results
are reproducible and show run-to-run variation.

## Setup
- Model: 2 convolution layers + 2 linear layers (SmallCNN)
- Training: 3 epochs, batch size 128, cross-entropy loss
- Metric: accuracy on the 10,000-image MNIST test set
- Only the optimizer and learning rate change between experiments

## How to run
pip install torch torchvision pandas
python mnist_experiments.py

This writes every run to results.csv and prints a summary table.

## Results (test accuracy, mean and std over 3 seeds)

| Optimizer | Learning rate | Mean accuracy | Std |
|---|---|---|---|
| Adam | 0.001 | 98.61% | 0.03% |
| Adam | 0.01 | 98.44% | 0.51% |
| SGD (momentum 0.9) | 0.01 | 98.47% | 0.15% |
| SGD (momentum 0.9) | 0.1 | 98.06% | 0.45% |

## What I observed
- Adam with learning rate 0.001 had the highest mean accuracy and the smallest
  variation between seeds, so it was the best and most stable setting here.
- Raising the learning rate made results less stable for both optimizers (larger std),
  and SGD at 0.1 had the lowest accuracy.
- All settings landed within about 0.55 percentage points of each other, and with only
  3 seeds some of the gaps are within the run-to-run variation, so I would not claim
  a clear winner among the middle three settings.

## Files
- mnist_experiments.py: training and experiment code
- results.csv: raw results for every run
- notebook (.ipynb): the Colab notebook used to run the experiments

## Possible next steps
More epochs, dropout, a learning-rate schedule, and a harder dataset such as CIFAR-10.
