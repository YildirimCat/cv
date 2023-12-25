import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#Weak

im = np.array(Image.open('C:\\Users\\Yldrm\\Desktop\\YL\\Bilgisayarla_Gorme\\Lab\\lab1\\images\\jpg\\baby.jpg').convert('L'))


im = snd.filters.gaussian_filter(im, 3)

sy = snd.filters.sobel(im, axis=0, mode="constant")
sx = snd.filters.sobel(im, axis=1, mode="constant")

B1 = np.power(sy, 2)
A1 = np.power(sx, 2)
C1 = np.multiply(sx, sy)

A = snd.filters.gaussian_filter(A1, 3)
B = snd.filters.gaussian_filter(B1, 3)
C = snd.filters.gaussian_filter(C1, 3)

#Deneysel olarak en iyi değer
k = 0.05 
corner_det = np.multiply(A, B) - np.multiply(C,2)
corner_trc = A+B
cornerness = corner_det - k * np.power(corner_trc, 2)
c_max = cornerness.max()
c_thresh = 0.01 * c_max
I_cornerness = cornerness
se = np.array([0, 1, 0], [1, 1, 1], [0, 1, 0])

#Binarization
se = se > 0

I_cornerness_dilated = morphology.binary_dilation(I_cornerness, se, iterations = 1)

im_corner = np.zeros((im.shape[0], im.shape[1], 3), dtype='uint8')
im_corner [:,:,0] = im.copy()
im_corner [:,:,1] = im.copy()
im_corner [:,:,2] = im.copy()
im_corner[I_cornerness_dilated] = [255, 0, 0]

fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols=3, figsize = (8, 8), sharex=True, sharey=True)
ax1.imshow(I_cornerness, cmap=plt.cm.gray)


