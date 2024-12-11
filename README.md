# MiniTorch Module 4

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py project/parallel_check.py tests/test_tensor_general.py


## Sentiment Classification

Epoch 1, loss 31.428937504702226, train accuracy: 47.78%
Validation accuracy: 47.00%
Best Valid accuracy: 47.00%
Epoch 2, loss 31.268993564772604, train accuracy: 50.67%
Validation accuracy: 53.00%
Best Valid accuracy: 53.00%
Epoch 3, loss 31.046654676486888, train accuracy: 52.89%
Validation accuracy: 56.00%
Best Valid accuracy: 56.00%
Epoch 4, loss 31.0343766614968, train accuracy: 53.78%
Validation accuracy: 51.00%
Best Valid accuracy: 56.00%
Epoch 5, loss 30.75989856752394, train accuracy: 53.56%
Validation accuracy: 48.00%
Best Valid accuracy: 56.00%
Epoch 6, loss 30.59060109340617, train accuracy: 55.78%
Validation accuracy: 50.00%
Best Valid accuracy: 56.00%
Epoch 7, loss 30.430221734549043, train accuracy: 61.56%
Validation accuracy: 55.00%
Best Valid accuracy: 56.00%
Epoch 8, loss 29.95787279381778, train accuracy: 64.89%
Validation accuracy: 58.00%
Best Valid accuracy: 58.00%
Epoch 9, loss 29.8203330727887, train accuracy: 62.44%
Validation accuracy: 57.00%
Best Valid accuracy: 58.00%
Epoch 10, loss 29.37156220483201, train accuracy: 65.33%
Validation accuracy: 54.00%
Best Valid accuracy: 58.00%
Epoch 11, loss 28.95532397462722, train accuracy: 63.78%
Validation accuracy: 64.00%
Best Valid accuracy: 64.00%
Epoch 12, loss 28.582032059786858, train accuracy: 67.78%
Validation accuracy: 65.00%
Best Valid accuracy: 65.00%
Epoch 13, loss 28.051334183608354, train accuracy: 70.67%
Validation accuracy: 59.00%
Best Valid accuracy: 65.00%
Epoch 14, loss 27.89337317282731, train accuracy: 68.00%
Validation accuracy: 63.00%
Best Valid accuracy: 65.00%
Epoch 15, loss 27.233442726314976, train accuracy: 69.33%
Validation accuracy: 59.00%
Best Valid accuracy: 65.00%
Epoch 16, loss 26.800708503781287, train accuracy: 69.78%
Validation accuracy: 64.00%
Best Valid accuracy: 65.00%
Epoch 17, loss 26.21810405290923, train accuracy: 70.00%
Validation accuracy: 68.00%
Best Valid accuracy: 68.00%
Epoch 18, loss 25.363749827153928, train accuracy: 74.67%
Validation accuracy: 60.00%
Best Valid accuracy: 68.00%
Epoch 19, loss 24.99361138728913, train accuracy: 76.00%
Validation accuracy: 60.00%
Best Valid accuracy: 68.00%
Epoch 20, loss 24.359692458337843, train accuracy: 74.22%
Validation accuracy: 69.00%
Best Valid accuracy: 69.00%
Epoch 21, loss 23.49605682417094, train accuracy: 75.78%
Validation accuracy: 62.00%
Best Valid accuracy: 69.00%
Epoch 22, loss 22.731059878079414, train accuracy: 78.00%
Validation accuracy: 68.00%
Best Valid accuracy: 69.00%
Epoch 23, loss 22.510634540970344, train accuracy: 77.11%
Validation accuracy: 66.00%
Best Valid accuracy: 69.00%
Epoch 24, loss 21.60716997830764, train accuracy: 80.22%
Validation accuracy: 68.00%
Best Valid accuracy: 69.00%
Epoch 25, loss 21.234477412164797, train accuracy: 78.22%
Validation accuracy: 68.00%
Best Valid accuracy: 69.00%
Epoch 26, loss 20.64486039173106, train accuracy: 78.89%
Validation accuracy: 68.00%
Best Valid accuracy: 69.00%
Epoch 27, loss 20.252475813473232, train accuracy: 78.44%
Validation accuracy: 66.00%
Best Valid accuracy: 69.00%
Epoch 28, loss 19.582007703431763, train accuracy: 80.00%
Validation accuracy: 71.00%
Best Valid accuracy: 71.00%
Epoch 29, loss 18.66451971225313, train accuracy: 82.22%
Validation accuracy: 71.00%
Best Valid accuracy: 71.00%
Epoch 30, loss 18.854781150031172, train accuracy: 80.22%
Validation accuracy: 68.00%
Best Valid accuracy: 71.00%
Epoch 31, loss 16.834548898749162, train accuracy: 84.89%
Validation accuracy: 66.00%
Best Valid accuracy: 71.00%
Epoch 32, loss 16.95273547270065, train accuracy: 85.11%
Validation accuracy: 56.00%
Best Valid accuracy: 71.00%
Epoch 33, loss 17.385171745339047, train accuracy: 82.22%
Validation accuracy: 69.00%
Best Valid accuracy: 71.00%
Epoch 34, loss 17.08034528678398, train accuracy: 82.22%
Validation accuracy: 60.00%
Best Valid accuracy: 71.00%
Epoch 35, loss 16.111719973322877, train accuracy: 84.44%
Validation accuracy: 66.00%
Best Valid accuracy: 71.00%
Epoch 36, loss 16.14718979597938, train accuracy: 85.33%
Validation accuracy: 66.00%
Best Valid accuracy: 71.00%
Epoch 37, loss 16.358976642392374, train accuracy: 82.67%
Validation accuracy: 62.00%
Best Valid accuracy: 71.00%