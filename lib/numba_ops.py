import numpy as np
from numba import jit, double

def custom_convolution(image, kernel):
    m, n, c = kernel.shape
    image = np.pad(image, ((0,m-1),(0,n-1),(0,0)), mode='constant')
    new_image = np.zeros_like(image)
    y, x, c = image.shape
    y = y - m + 1
    x = x - n + 1
    for k in range(c):
        for i in range(y):
            for j in range(x):
                for ii in range(i,i+m):
                    for jj in range(j, j+n):
                        new_image[i][j][k] += (image[ii, jj, k]*kernel[ii-i, jj-j, 0])
    return new_image[:y,:x,:]

convolve = jit(double[:, :, :](double[:, :, :],double[:, :, :]))(custom_convolution)

def derivative(f, dir=0):
    if dir == 0:
        kernel = np.array([[-1,1],[-1,1]], dtype=np.float64)[:,:,None]
    else:
        kernel = np.array([[-1,1],[-1,1]], dtype=np.float64)[:,:,None]
    fx = convolve(f, kernel)
    return fx / np.max(np.abs(fx))

def _gradient(f):
    fx = derivative(f, 0)
    fy = derivative(f, 1)
    return [fx, fy]

def _laplacian3d(f):
    kernel = np.array([[0,1,0],
                       [1,-4,1],
                       [0,1,0]], dtype=np.float64)[:,:,None]
    return convolve(f, kernel)
