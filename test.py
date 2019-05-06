from lib.loader import Loader
from models.harmonic import Harmonic
from models.tv import TV
from models.hybrid import Hybrid
from models.exponential import Exponential
from lib.ops import *
import matplotlib.pyplot as plt

prob = Loader(fpath='./dataset/lena.jpg')
img, noisy, mask = prob.gen_mask()
# Harmonic Inpainting
# inpainter = Harmonic()
# inpainter.solve(img, noisy, mask)
# TV Inpainting
inpainter = TV()
inpainter.solve(img, noisy, mask)
# Exponential Inpainting
# inpainter = Exponential()
# inpainter.solve(img, noisy, mask)
