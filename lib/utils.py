import numpy as np
# import numba

class MaxPooling(object):

    def __init__(self, X, kernel_size=(2,2), stride=(2,2), padding=0):

        if len(X.shape) != 4:
            raise ValueError("Input must have be a tensor of shape N*C*H*W!")

        self.X = X
        if padding:
            self.X = zero_pad(self.X, padding)
        self.input_shape = self.X.shape
        N, C, H, W = self.input_shape

        self.kernel_size_h, self.kernel_size_w = kernel_size
        self.stride_h, self.stride_w = stride

        self.out_h = np.floor((H-self.kernel_size_h)/self.stride_h) + 1
        self.out_w = np.floor((W-self.kernel_size_w)/self.stride_w) + 1

        self.out_h = int(self.out_h)
        self.out_w = int(self.out_w)

        self.out = np.empty((N, C, self.out_h, self.out_w))

        # for each output feature map, store the corresponding index in the original feature map
        # in an 2*out_h*out_w tensor. For each element in output feature map, store the index in
        # the same position in the max_index tensor. First element denote row index, second element
        # denote column index
        self.max_index = np.empty((N, C, 2, self.out_h, self.out_w), dtype=np.int16)

    # @numba.jit
    def forward(self):
        N, C, _, _ = self.X.shape

        for n in range(N):
            for c in range(C):
                for h in range(self.out_h):
                    for w in range(self.out_w):

                        h_start = h*self.stride_h
                        h_end =  h_start + self.kernel_size_h
                        w_start = w*self.stride_w
                        w_end = w_start + self.kernel_size_w

                        self.out[n, c, h, w] = np.max(self.X[n, c, h_start:h_end, w_start:w_end])

                        scalar_ind = np.argmax(self.X[n, c, h_start:h_end, w_start:w_end])
                        # ind is in (row_ind, col_ind) format
                        ind = np.unravel_index(scalar_ind, (self.kernel_size_h, self.kernel_size_w))

                        # real index of maximum element in the local region
                        real_ind = (ind[0]+h_start, ind[1]+w_start)

                        # store this real index in two part
                        self.max_index[n, c, 0, h, w] = real_ind[0]
                        self.max_index[n, c, 1, h, w] = real_ind[1]
        return self.out

# @numba.jit
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
    X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), 'constant', constant_values = (0,0))
    ### END CODE HERE ###
    
    return X_pad


def sigmoid(x):
    return 1 / (1 + np.exp(-x))