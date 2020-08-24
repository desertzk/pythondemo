import numpy as np
import h5py
import matplotlib.pyplot as plt


def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image,
    as illustrated in Figure 1.

    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions

    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """

    ### START CODE HERE ### (≈ 1 line)
    X_pad = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
    ### END CODE HERE ###

    return X_pad



# GRADED FUNCTION: conv_forward



def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function

    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C) n_C_prev前一层通道数 n_C期望输出通道数也就是filter的个数
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"

    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """

    ### START CODE HERE ###
    # Retrieve dimensions from A_prev's shape (≈1 line)
    (m, n_C_prev, n_H_prev, n_W_prev) = A_prev.shape

    # Retrieve dimensions from W's shape (≈1 line)
    (n_C_prev, n_C,f, f) = W.shape

    # Retrieve information from "hparameters" (≈2 lines)
    stride = hparameters['stride']
    pad = hparameters['pad']

    # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
    n_H = 1 + int((n_H_prev + 2 * pad - f) / stride)
    n_W = 1 + int((n_W_prev + 2 * pad - f) / stride)

    # Initialize the output volume Z with zeros. (≈1 line)
    Z = np.zeros((m, n_C, n_H, n_W))

    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):                               # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i]                               # Select ith training example's padded activation
        for c in range(n_C):  # loop over channels (= #filters) of the output volume
            for h in range(n_H):                           # loop over vertical axis of the output volume
                for w in range(n_W):                       # loop over horizontal axis of the output volume


                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                    a_slice_prev = a_prev_pad[:,vert_start:vert_end, horiz_start:horiz_end]

                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                    Z[i, c, h, w] = np.sum(np.multiply(a_slice_prev, W[:, c , :, :]) + b[:, c, :, :])
    #                     这里的c代表刚好取c这个channel

    ### END CODE HERE ###

    # Making sure your output shape is correct
    assert(Z.shape == (m, n_C, n_H, n_W))

    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)

    return Z, cache


# GRADED FUNCTION: pool_forward

# def pool_forward(A_prev, hparameters, mode="max"):
#     """
#     Implements the forward pass of the pooling layer
#
#     Arguments:
#     A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
#     hparameters -- python dictionary containing "f" and "stride"
#     mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
#
#     Returns:
#     A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
#     cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters
#     """
#
#     # Retrieve dimensions from the input shape
#     (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
#
#     # Retrieve hyperparameters from "hparameters"
#     f = hparameters["f"]
#     stride = hparameters["stride"]
#
#     # Define the dimensions of the output
#     n_H = int(1 + (n_H_prev - f) / stride)
#     n_W = int(1 + (n_W_prev - f) / stride)
#     n_C = n_C_prev
#
#     # Initialize output matrix A
#     A = np.zeros((m, n_H, n_W, n_C))
#
#     ### START CODE HERE ###
#     for i in range(m):  # loop over the training examples
#         for h in range(n_H):  # loop on the vertical axis of the output volume
#             for w in range(n_W):  # loop on the horizontal axis of the output volume
#                 for c in range(n_C):  # loop over the channels of the output volume
#
#                     # Find the corners of the current "slice" (≈4 lines)
#                     vert_start = h * stride
#                     vert_end = vert_start + f
#                     horiz_start = w * stride
#                     horiz_end = horiz_start + f
#
#                     # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
#                     a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
#
#                     # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
#                     if mode == "max":
#                         A[i, h, w, c] = np.max(a_prev_slice)
#                     elif mode == "average":
#                         A[i, h, w, c] = np.mean(a_prev_slice)
#
#     ### END CODE HERE ###
#
#     # Store the input and hparameters in "cache" for pool_backward()
#     cache = (A_prev, hparameters)
#
#     # Making sure your output shape is correct
#     assert (A.shape == (m, n_H, n_W, n_C))
#
#     return A, cache


def conv_backward(dZ, cache):
    """
    Implement the backward propagation for a convolution function

    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()

    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """

    ### START CODE HERE ###
    # Retrieve information from "cache"
    (A_prev, W, b, hparameters) = cache

    # Retrieve dimensions from A_prev's shape
    (m, n_C_prev, n_H_prev, n_W_prev) = A_prev.shape

    # Retrieve dimensions from W's shape
    (n_C_prev, n_C,f, f) = W.shape

    # Retrieve information from "hparameters"
    stride = hparameters['stride']
    pad = hparameters['pad']

    # Retrieve dimensions from dZ's shape
    (m, n_C, n_H, n_W) = dZ.shape

    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros((m, n_C_prev, n_H_prev, n_W_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros(( 1, n_C ,1, 1))

    # Pad A_prev and dA_prev
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):  # loop over the training examples

        # select ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for c in range(n_C):  # loop over the channels of the output volume
            for h in range(n_H):  # loop over vertical axis of the output volume
                for w in range(n_W):  # loop over horizontal axis of the output volume


                    # Find the corners of the current "slice"
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # Use the corners to define the slice from a_prev_pad
                    a_slice = a_prev_pad[:,vert_start:vert_end, horiz_start:horiz_end]

                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    da_prev_pad[:,vert_start:vert_end, horiz_start:horiz_end] += W[:, c ,:, :] * dZ[i, c, h, w]
                    dW[:, c,:, :] += a_slice * dZ[i, c, h, w]
                    db[:, c,:, :] += dZ[i, c, h, w]

        # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
        dA_prev[i, :, :, :] = dA_prev_pad[i, :, pad:-pad, pad:-pad]
    ### END CODE HERE ###

    # Making sure your output shape is correct
    assert (dA_prev.shape == (m, n_C_prev, n_H_prev, n_W_prev))

    return dA_prev, dW, db


def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.

    Arguments:
    x -- Array of shape (f, f)

    Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """

    ### START CODE HERE ### (≈1 line)
    mask = (x == np.max(x))
    ### END CODE HERE ###

    return mask








np.random.seed(1)
# training examples,channel, height,weight
A_prev = np.random.randn(10,3,4,4)
# , n_Channel_prev, n_Channel filter, filter
W = np.random.randn(3,8,2,2)
b = np.random.randn(1,8,1,1)
hparameters = {"pad" : 2,
               "stride": 1}

Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
print("Z's mean =", np.mean(Z))
print("Z's shape =", Z.shape)
print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])
dA, dW, db = conv_backward(Z, cache_conv)
print("dA_mean =", np.mean(dA))
print("dW_mean =", np.mean(dW))
print("db_mean =", np.mean(db))



