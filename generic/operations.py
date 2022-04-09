import numpy as np

from typing import Callable, List
from numpy import ndarray


# A Function takes in an ndarrayy as a argument and produces an ndarray
Array_Function = Callable[[ndarray], ndarray]

# A Chain is a list of functions
Chain = List[Array_Function]


def deriv(func: Array_Function,
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


def chain_deriv_2(chain: Chain,
                  input_range: ndarray
                  ) -> ndarray:
    """
    Uses teh chain rule to compute the derivative of two nested functions:
    :param chain: chain of functions
    :param input_range: input ndarray for which to calculate the chain derivative
    :return: results of evaluating the chain derivate in the input points
    """
    if len(chain) != 2:
        msg = "This function requires 'Chain' objects of length 2"
        print(msg)
        return None

    if input_range.ndim != 1:
        msg = "This function requires a 1 dimensional ndarray as input_range"
        print(msg)
        return None

    f1 = chain[0]
    f2 = chain[1]

    f1_of_x = f1(input_range)
    deriv_f1_dx = deriv(f1, input_range)
    deriv_f2_du = deriv(f2, f1_of_x)

    return deriv_f2_du * deriv_f1_dx


def chain_deriv_3(chain: Chain,
                  input_range: ndarray
                  ) -> ndarray:
    """
    Uses the chain rule to compute teh derivative of three nested functions:
    (f3(f2(f1)))' = f3'(f2(f1(x))) * f2'(f1(x)) * f1'(x)
    :param chain: chain of functions, first in the list being f1
    :param input_range: The point where to compute de derivative
    :return: ndarray with the values of the derivative in the input points
    """
    assert len(chain) == 3, "This function requires. 'Chain' objects to have length 3"

    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[3]

    f1_of_x = f1(input_range)
    f2_of_f1_of_x = f2(f1_of_x)

    df3du = deriv(f3, f2_of_f1_of_x)
    df2du = deriv(f2, f1_of_x)
    df1dx = deriv(f1, input_range)

    return df1dx * df2du * df3du


def chain_length_2(chain: Chain,
                   x: ndarray
                   ) -> ndarray:
    """
    Evaluates two functions in a row, in a "Chain". (Applying in the list order)
    :param chain: the chain of functions
    :param x: the input ndarray
    :return: the result ndarray after performing the chain functions on it
    """
    if len(chain) != 2:
        f1 = chain[0]
        f2 = chain[1]

        return f2(f1(x))
    else:
        msg = "Error: chain_length_2 only supports a chain of two functions"
        print(msg)
        return None


def square(x: ndarray
           ) -> ndarray:
    """
    Square each element in the input array
    :param x: numpy array
    :return: the numpy array with its element squared
    """
    return np.power(x, 2)


def leaky_relu(x: ndarray
               ) -> ndarray:
    """
    Apply "Leaky ReLU function to each element in the input ndarray.
    :param x: input ndarray
    :return: ndarray with the ReLU operation applied to each of its elements
    """
    return np.maximum(0.2 * x, x)


def sigmoid(x: ndarray,
            ) -> ndarray:
    """
    Apply the sigmoid function to each element in the input ndarray
    :param x: input ndarray
    :return: ndarray with the sigmoid function applied to each one of the elements
    """
    return 1 / (1 + np.exp(-x))


def gamma(x: ndarray,
          y: ndarray
          ) -> ndarray:
    """
    Gamma function that adds x and y
    :param x: x array
    :param y: y array
    :return: x+y array
    """
    return x + y


def multiple_inputs_add(x: ndarray,
                        y: ndarray,
                        sigma: Array_Function
                        ) -> float:
    """
    Function with multiple inputs and addition, forward pass

    :param x: first input of the function
    :param y: second input of the function
    :param sigma: sigma function
    :return: result of the operation
    """
    assert x.shape == y.shape
    a = gamma(x, y)
    return sigma(a)


def multiple_inputs_add_backward(x: ndarray,
                                 y: ndarray,
                                 sigma: Array_Function) -> {float, float}:
    """
    Computes the derivative of the function forward "pass"
    :param x:
    :param y:
    :param sigma:
    :return:
    """
    a = gamma(x, y)
    dsigma_da = deriv(sigma, a)
    da_dx = deriv(gamma, x)
    da_dy = deriv(gamma, y)

    return dsigma_da * da_dx, dsigma_da * da_dy


def matmul_forward(X: ndarray,
                   W: ndarray
                   ) -> ndarray:
    """
    Computs the forward pass of a matrix multiplication
    :param X: X matrix
    :param W: W matrix
    :return: The product matrix
    """
    assert X.shape[1] == W.shape[0] .format(X.shape[1], W.shape[0])

    N = np.dot(X, W)

    return N
