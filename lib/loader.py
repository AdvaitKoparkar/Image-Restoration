import matplotlib.image as mimg
import matplotlib.pyplot as plt
import numpy as np
from lib.draw import Draw
import cv2
import os
import pickle

class Loader(object):
    def __init__(self, fpath):
        self.fpath = fpath
        self.mask_drawer = Draw("Draw Mask")

    def gen_mask(self, restore='dataset/saved_state.pkl'):
        if restore is not None and os.path.exists(restore):
            with open(restore, "rb") as fh:
                load_dict = pickle.load(fh)
            self.img, self.noisy, self.mask = load_dict['img'], load_dict['noisy'], load_dict['mask']
        else:
            self.img = self._load_image()
            self.img = self.img /  np.max(self.img)
            self.mask_drawer.set_clean_img(self.img)
            self.mask = self._load_mask()

            if self.img.ndim == 3:
                M, N, C = self.img.shape
                if self.mask.ndim < 3:
                    self.mask = np.repeat(self.mask[:, :, np.newaxis], C, axis=2)

            else:
                M, N = self.img.shape
                C = 3
                self.img = scipy.expand_dims(self.img, axis=2)
                self.mask  = scipy.expand_dims(self.mask, axis=2)

            self._add_noise()
            # pdb.set_trace()
            self.noisy = self.noisy / np.max(self.noisy)
            with open("dataset/saved_state.pkl", "wb") as fh:
                load_dict = pickle.dump({'img': self.img, 'mask': self.mask, 'noisy':self.noisy}, fh)
        return self.img, self.noisy, self.mask

    def _load_image(self):
        return mimg.imread(self.fpath)[:,:,0:3]

    def _load_mask(self):
        self.mask_drawer.set_img_size((self.img.shape[0], self.img.shape[1]))
        return self.mask_drawer.run()

    def _add_noise(self):
        M, N, C = self.img.shape
        n = np.random.rand(M, N, C)
        self.noisy = self.mask * self.img + (1 - self.mask)*n
