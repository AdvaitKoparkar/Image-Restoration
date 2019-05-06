from models.model import Model
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from lib.ops import *
# from lib.numba_ops import *

class Harmonic(Model):
    def __init__(self, **params):
        super(Harmonic, self).__init__(**params)
        self.name = "Harmonic"
        self.alpha = params.get('lambda', 10)
        self.eps = params.get('eps', 1e-5)
        self.maxiter = params.get('maxiter', 1500)
        self.dt = params.get('dt', 0.1)
        self.loss_hist = []
        self.accuracy = []
        self.save_path = "dataset/"+self.name+"_"+str(self.maxiter)+".pkl"

    def loss(self):
        ux, uy = gradient(self.u)
        self.loss_hist.append(np.sum(ux*ux + uy*uy + self.mask*(self.u - self.noisy)**2))

    def solve(self, img, noisy, mask):
        self.img = img
        self.noisy = noisy
        self.mask = mask

        if noisy.ndim==3:
            M,N,C = noisy.shape
        else:
            M,N = noisy.shape
            C = 1

        self.u = noisy.copy()
        self.gen_metrics()
        self.loss()
        for iter in tqdm(range(0,self.maxiter)):
            laplacian = laplacian3d(self.u[:,:,:])
            u_dash = self.u + self.dt*(laplacian + self.smoothfidelity())
            diff_u = np.linalg.norm(u_dash-self.u/np.linalg.norm(u_dash))
            self.u = u_dash
            self.u[self.u > 1] = 1.0
            self.u[self.u < 0] = 0.0
            self.loss()
            self.gen_metrics()
            if iter%self.update_every == 0:
                self.update_figure()
            if diff_u<self.eps:
                break

        self.save_plots()
        return self.u
