from PIL import Image
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as snd

im = np.array(Image.open('./images/jpg/trailer.jpg').convert('L'))
filt = np.array([[0,1,0],[1,1,1],[0,1,0]])
filt = 1/5*filt
print(type(filt))
im_cor = snd.correlate(im, filt, mode='constant')
im_conv = snd.convolve(im, filt, mode='nearest')
# Gaussian Filtering
im_gauss = snd.filters.gaussian_filter(im,3)
im_gauss2 = snd.gaussian_filter(im,5)
im_list = np.concatenate((im, im_gauss,im_gauss2), axis=1)
imgplot = plt.imshow(im_list, cmap='gray') 
plt.show()