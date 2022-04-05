import numpy as np

from operations import deriv, square

if __name__ == "__main__":
    points = np.array([5])
    print(deriv(square, points))
