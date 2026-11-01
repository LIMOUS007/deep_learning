# Deep Learning from Scratch

A step-by-step learning path through neural networks. It starts with a single neuron in NumPy, builds a full network by hand, then moves to PyTorch and real classification tasks.

## Topics

| # | File | Topic |
|---|------|-------|
| 01 | `_01_neuron.py` | A single neuron: weighted sum + bias, ReLU activation |
| 02 | `_02_neuron_layer.py` | A layer of neurons using matrix multiplication |
| 03 | `_03_multi_layer.py` | Stacking layers into a multi-layer forward pass |
| 04 | `_04_loss_functions.py` | Loss functions: MSE, MAE, cross-entropy (stable log-softmax), and more |
| 05 | `_05_activation_functions.py` | Activations: step, sigmoid (numerically stable), tanh, ReLU, Leaky ReLU, SiLU, etc. |
| 06 | `_06_basic_backpropagation.py` | Gradient descent on a single neuron with manually derived gradients |
| 07 | `_07_backpropagation.py` | Backpropagation through a multi-layer network |
| 08 | `_08_optimizers.py` | Optimizers: SGD, Momentum, AdaGrad, RMSProp, Adam, AdamW |
| 09 | `_09_neural_network_from_scratch.py` | Full NumPy network trained with every optimizer and loss curves compared |
| 10 | `_10_neural_network_from_scratch_with_OOP.py` | The same network refactored with object-oriented design |
| 11 | `_11_pytorch.py` | PyTorch basics: `nn.Module`, `nn.Sequential`, `Dataset`, `DataLoader` |
| 12 | `_12_pytorch_practice.py` | PyTorch training loop practice with loss plotting |
| 13 | `_13_binary_classification.py` | Binary classification in PyTorch |
| 14 | `_14_multiclass_classification.py` | Multiclass classification in PyTorch |

### Project: Academic Success Classification

`Classification_with_an_Academic_Success_Dataset/main.py` applies these ideas to the Kaggle Playground "Classification with an Academic Success Dataset" competition, which predicts student outcomes. The CSV data is not tracked; download it from Kaggle into that folder.

## Setup

```bash
pip install numpy matplotlib torch
python _01_neuron.py
```

Files 09 and 10 import from 04, 05 and 08, so run them from this directory.
