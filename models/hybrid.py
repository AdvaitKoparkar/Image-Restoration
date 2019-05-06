from models.model import Model
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from lib.ops import *
import matplotlib.animation as animation

class Hybrid(Model):
    def __init__(self, **params):
        super(Hybrid, self).__init__()
        self.name = "Hybrid"
        self.alpha = params.get('lambda', 100)
        self.eps = params.get('eps', 1e-5)
        self.maxiter = params.get('maxiter', 500)
        self.w1 = params.get('w1', 0.5)
        self.w2 = params.get('w2', 0.5)
        self.dt = params.get('dt', 0.01)
        self.loss_hist = []
        self.accuracy = []

    def loss(self):
        ux, uy = gradient(self.u)
        self.loss_hist.append(np.sum(ux*ux + uy*uy + np.sqrt(ux*ux + uy*uy) + self.mask*(self.u - self.noisy)**2))

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
        delta = 1e-8
        for iter in tqdm(range(0,self.maxiter)):
            # Harmonic
            laplacian = laplacian3d(self.u[:,:,:])

            # TV
            ux_forward = derivative(self.u,dir=0,type="forward")
            ux_forward2 = ux_forward*ux_forward
            uy_mm = minmod(derivative(self.u, dir=1, type="forward"), derivative(self.u, dir=1, type="backward"))
            # uy_mm = derivative(u,dir=1,type="forward")

            uy_forward = derivative(self.u,dir=1,type="forward")
            uy_forward2 = uy_forward*uy_forward
            ux_mm = minmod(derivative(self.u, dir=0, type="forward"), derivative(self.u, dir=0, type="backward"))
            # ux_mm = derivative(u,dir=0,type="forward")

            uxx = derivative((ux_forward)/np.sqrt(delta + ux_forward2 + (uy_mm**2)), dir=0, type="backward")
            uyy = derivative((uy_forward)/np.sqrt(delta + uy_forward2 + (ux_mm**2)), dir=1, type="backward")
            u_dash = self.u + self.dt*self.w1*(uxx+uyy+self.l2fidelity()) + self.dt*self.w2*(laplacian + self.l2fidelity())

            diff_u = np.linalg.norm(u_dash-self.u/np.linalg.norm(u_dash))
            self.u = u_dash
            self.u[self.u > 1] = 1.0
            self.u[self.u < 0] = 0.0
            if iter%self.update_every == 0:
                self.update_figure()
            self.loss()
            # plt.imshow(u)
            # plt.show()
            if diff_u<self.eps:
                break

        self.save_plots()
        return self.u
