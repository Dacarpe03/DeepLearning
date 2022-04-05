import numpy as np

from typing import Callable, List
from numpy import ndarray


# A Function takes in an ndarrayy as a argument and produces an ndarray
Array_Function = Callable[[ndarray], ndarray]

# A Chain is a list of functions
Chain = List[Array_Function]


def deriv(
        func: Array_Function,
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


def chain_length_2(
        chain: Chain,
        x: ndarray
        ) -> ndarray:
    """
    Evaluates two functions in a row, in a "Chain". (Applying in the list order)
    :param chain: the chain of functions
    :param x: the input ndarray
    :return: the result ndarray after performing the chain functions on it
    """
    if len(chain) == 2:
        f1 = chain[0]
        f2 = chain[1]

        return f2(f1(x))
    else:
        msg = "Error: chain_length_2 only supports a chain of two functions"
        print(msg)
        return None


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
