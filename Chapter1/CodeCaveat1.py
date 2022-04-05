import numpy as np


def main():
	print("Python list operations")
	a = [1, 2, 3]
	b = [4, 5, 6]
	print("a:", a)
	print("b:", b)
	print("a+b:", a+b)
	try:
		print(a*b)
	except TypeError:
		print("a*b has no meaning for Python lists")

	print()
	print("Numpy array operations")
	a = np.array(a)
	b = np.array(b)
	print("a+b:", a+b)
	print("a*b:", a*b)

	print()
	print("Adding axis")
	a = np.array([[1, 2], [3, 4]])
	print("a:", a)
	print("a.sum(axis=0):", a.sum(axis=0))
	print("a.sum(axis=1):", a.sum(axis=1))

	print()
	print("Adding 1D array to a 2D array.")
	a = np.array([[1, 2, 3], [4, 5, 6]])
	b = np.array([10, 20, 30])
	print("a:", a)
	print("b:", b)
	print("a+b:", a+b)


if __name__ == "__main__":
	main()
