from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    new_height: int = height // kh
    new_width: int = width // kw
    output = input.contiguous().view(batch, channel, height, new_width, kw)
    output = output.permute(0, 1, 3, 2, 4)
    output = output.contiguous().view(batch, channel, new_width, new_height, kh * kw)
    output = output.permute(0, 1, 3, 2, 4)
    return output, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D

    Args:
    ----
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
    -------
        Pooled tensor

    """
    batch, channel, height, width = input.shape
    t, height, weight = tile(input, kernel)
    t = t.mean(4)
    t = t.view(batch, channel, height, weight)
    return t


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor.

    Args:
    ----
        input : input tensor
        dim : dimension to apply argmax


    Returns:
    -------
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward of max should be max reduction"""
        # TODO: Implement for Task 4.4.
        max_red = max_reduce(input, int(dim.item()))
        ctx.save_for_backward(input, max_red)
        return max_red

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward of max should be argmax (see above)"""
        # TODO: Implement for Task 4.4.
        (input, max_red) = ctx.saved_values
        return (grad_output * (max_red == input)), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max to input."""
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    r"""Compute the softmax as a tensor. $z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}$

    Args:
    ----
        input : input tensor
        dim : dimension to apply softmax

    Returns:
    -------
        softmax tensor

    """
    t = input.exp()
    s = t.sum(dim)
    return t / s


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    r"""Compute the log of the softmax as a tensor.

    $z_i = x_i - \log \sum_i e^{x_i}$

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
    ----
        input : input tensor
        dim : dimension to apply log-softmax

    Returns:
    -------
         log of softmax tensor

    """
    t = input.exp()
    t = t.sum(dim)
    t = t.log()
    return input - t


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor : pooled tensor

    """
    batch, channel, height, width = input.shape
    t, _, _ = tile(input, kernel)
    t = max(t, 4)
    t = t.view(batch, channel, t.shape[2], t.shape[3])
    return t


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise.

    Args:
    ----
        input : input tensor
        rate : probability [0, 1) of dropping out each position
        ignore : skip dropout, i.e. do nothing at all

    Returns:
    -------
        tensor with randoom positions dropped out

    """
    if not ignore:
        rand_tensor = rand(input.shape)
        random_drop = rand_tensor > rate
        return input * random_drop
    else:
        return input
