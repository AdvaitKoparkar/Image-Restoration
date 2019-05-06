import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import convolve, gaussian_filter

def derivative(f, dir=0, type="forward"):
    if dir == 0:
        if type == "forward":
            kernel = np.array([[1,-1]], dtype=np.float64)[:,:,None]
        else:
            kernel = np.array([[-1,1]], dtype=np.float64)[:,:,None]
    else:
        if type == "forward":
            kernel = np.array([[1],[-1]], dtype=np.float64)[:,:,None]
        else:
            kernel = np.array([[-1],[1]], dtype=np.float64)[:,:,None]

    fx = convolve(f, kernel, mode="constant")
    return fx / np.max(np.abs(fx))

def minmod(x, y):
    s = (np.sign(x) + np.sign(y))/2.0
    modx, mody = np.abs(x), np.abs(y)
    mx, my = modx<mody, mody<=modx
    ret = s*(mx*modx + my*mody)
    return ret / np.max(1e-12 + np.abs(ret))

def gradient(f):
    fx = derivative(f, 0)
    fy = derivative(f, 1)
    return [fx, fy]

def smooth(x):
    return gaussian_filter(x, sigma=(5,5,0), order=0)

def hessian(f):
    fx = derivative(f, 0)
    fy = derivative(f, 1)
    fxy = derivative(fx, 1)
    fyx = derivative(fy, 0)
    fxx = derivative(fx, 0)
    fyy = derivative(fy, 1)
    return [fxx, fyy, fxy, fyx]

def laplacian2d(f):
    kernel = np.array([[0,1,0],
                       [1,-4,1],
                       [0,1,0]])
    return convolve2d(f, kernel, mode='same')

def laplacian3d(f):
    kernel = np.array([[0,1,0],
                       [1,-4,1],
                       [0,1,0]])[:, :, None]
    return convolve(f, kernel, mode='constant')
