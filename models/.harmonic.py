from models.model import Model
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from lib.ops import *

class Harmonic(Model):
    def __init__(self, **params):
        super(Harmonic, self).__init__()
        self.lambda_ = params.get('lambda', 10)
        self.eps = params.get('eps', 1e-5)
        self.max_iter = params.get('maxiter', 500)
        self.dt = params.get('dt', 0.1)

    def solve(self, noisy, mask):
        if noisy.ndim==3:
            M,N,C = noisy.shape
        else:
            M,N = noisy.shape
            C = 1
        lambda_ = self.lambda_
        dt = self.dt
        maxiter = self.max_iter
        tol = self.eps
        u = noisy.copy()
        channel_status = np.zeros((C,1), dtype=np.int32)
        loss = []

        for iter in range(0,maxiter):
            for c in range(0,C):
                if channel_status[c] != 0:
                    break
                laplacian_cv = cv2.Laplacian(u[:,:,c],cv2.CV_64F)
                laplacian_me = laplacian2d(u[:,:,c])
                # plt.subplot(1,2,1)
                # plt.hist(laplacian_cv)
                # plt.subplot(1,2,2)
                # plt.hist(laplacian_me)
                # plt.show()
                unew = u[:,:,c] + dt*( laplacian_me + lambda_ * mask[:,:,c] * (noisy[:,:,c]-u[:,:,c]) )

                diff_u = np.linalg.norm(unew.reshape(M*N,1)-u[:,:,c].reshape(M*N,1),2)/np.linalg.norm(unew.reshape(M*N,1),2);

                u[:,:,c] = unew

                if diff_u<tol:
                    channel_status[c] = 1

        plt.subplot(1,2,1)
        plt.imshow(noisy)
        plt.subplot(1,2,2)
        plt.imshow(u)
        plt.show()
        return u
