import numpy as np

from typing import Callable
from numpy import ndarray


def deriv(
        func: Callable[[ndarray], ndarray],
        input_: ndarray,
        delta: float = 0.001
        ) -> ndarray:
    """
    Evaluates the derivative of a function at every element in the input_ array
    :param func: function to evaluate
    :param input_: points where to evaluate
    :param delta: delta value for the limit derivative formula
    :return: the derivatives from the function at the input points
    """

    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)


def square(
        x: ndarray
        ) -> ndarray:
    """
    Square each element in the input array
    :param x: numpy array
    :return: the numpy array with its element squared
    """
    return np.power(x, 2)


def leaky_relu(
        x: ndarray
        ) -> ndarray:
    """
    Apply "Leaky ReLU function to each element in the input ndarray.
    :param x: input ndarray
    :return: ndarray with the ReLU operation applied to each of its elements
    """
    return np.maximum(0.2 * x, x)
