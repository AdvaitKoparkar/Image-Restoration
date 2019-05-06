from models.model import Model
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
from lib.ops import *

class Exponential(Model):
    def __init__(self, **params):
        super(Exponential, self).__init__()
        self.name = "Exponential"
        self.alpha = params.get('lambda', 100)
        self.eps = params.get('eps', 1e-5)
        self.maxiter = params.get('maxiter', 1500)
        self.kappa = params.get('kappa', 1)
        self.dt = params.get('dt', 0.01)
        self.accuracy = []
        self.loss_hist = []
        self.save_path = "dataset/"+self.name+"_"+str(self.maxiter)+".pkl"

    def loss(self):
        ux, uy = gradient(self.u)
        self.loss_hist.append(np.sum(1-np.exp(-1*self.kappa*(ux*ux + uy*uy)) + self.mask*(self.u - self.noisy)**2))

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
        delta = 1e-8
        for iter in tqdm(range(0,self.maxiter)):
            ux = derivative(self.u,dir=0,type="forward")
            uy = derivative(self.u,dir=1,type="forward")
            exp = np.exp(-1*self.kappa*(ux*ux + uy*uy))
            uxx = derivative((ux*exp), dir=0, type="backward")
            uyy = derivative((uy*exp), dir=1, type="backward")

            reg = self.kappa * (uxx + uyy)
            u_dash = self.u + self.dt*(reg+self.l2fidelity())

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
