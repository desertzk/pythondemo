import numpy as np

v = np.array([12, 24, 36])
w = np.array([45, 55])

# To compute an outer product we first
# reshape v to a column vector of shape 3x1
# then broadcast it against w to yield an output
# of shape 3x2 which is the outer product of v and w
print(np.reshape(v, (3, 1)) * w)

X = np.array([[12, 22, 33], [45, 55, 66]])

# x has shape 2x3 and v has shape (3, )
# so they broadcast to 2x3,
print(X + v)

# Add a vector to each column of a matrix X has
# shape 2x3 and w has shape (2, ) If we transpose X
# then it has shape 3x2 and can be broadcast against w
# to yield a result of shape 3x2.

# Transposing this yields the final result
# of shape 2x3 which is the matrix.
print((X.T + w).T)

# Another solution is to reshape w to be a column
# vector of shape 2X1 we can then broadcast it
# directly against X to produce the same output.
print(X + np.reshape(w, (2, 1)))

# Multiply a matrix by a constant, X has shape 2x3.
# Numpy treats scalars as arrays of shape();
# these can be broadcast together to shape 2x3.
print(X * 2)
