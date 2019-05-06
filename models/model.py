from abc import abstractmethod, ABC
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from lib.ops import *
import skimage
import pickle

class Model(object):
    def __init__(self, **params):
        self.update_every = params.get('update_every', 1)
        self.fig = plt.figure()
        self.imgs = []
        self.psnr = []
        self.ssim = []
        self.nrmse = []

    def l2fidelity(self):
        return self.alpha*self.mask*(self.noisy-self.u)

    def smoothfidelity(self):
        return self.alpha*smooth(self.mask)*self.mask*(self.noisy-self.u)

    def update_figure(self):
        self.imgs.append([plt.imshow(self.u, vmin=0, vmax=1)])

    def gen_metrics(self):
        self.psnr.append(skimage.measure.compare_psnr(self.u, self.img))
        self.ssim.append(skimage.measure.compare_ssim(self.u, self.img, multichannel=True))
        self.nrmse.append(skimage.measure.compare_nrmse(self.u, self.img))

    def save_plots(self):
        ani = animation.ArtistAnimation(self.fig, self.imgs, interval=100, blit=True, repeat_delay=1000)
        ani.save('results/'+self.name+'.mp4', writer='ffmpeg')
        saved_state = {'img': self.u,
                       'noisy': self.noisy,
                       'mask': self.mask,
                       'psnr': self.psnr,
                       'ssim': self.ssim,
                       'nrmse': self.nrmse,
                       'loss': self.loss_hist}
        with open(self.save_path, "wb") as fh:
            pickle.dump(saved_state, fh)
        with open("dataset/loss_"+self.name+"_dt%s_iter%d.pkl" %(str(self.dt), self.maxiter), "wb") as fh:
            pickle.dump({'loss':self.loss_hist, 'dt':self.dt}, fh)
