from PIL import Image
from pylab import *
import numpy as np
import matplotlib.pyplot as plt

def im_hist(im):
    hist = np.zeros(256, dtype=np.int)
    width = im.shape[0]
    height = im.shape[1]
    for i in range(width):
        for j in range(height):
            a = im[i,j]
            hist[a] += 1
    return hist

def cum_sum(hist):
    cum_hist = np.copy(hist)
    for i in range(1,len(hist)):
        cum_hist[i] = hist[i] + cum_hist[i-1]
    return cum_hist

def hist_match(img, img_ref):
    width = img.shape[0]
    height = img.shape[1]
    hist_img = im_hist(img)
    hist_ref = im_hist(img_ref)
    cum_img = cum_sum(hist_img)
    cum_ref = cum_sum(hist_ref)
    cum_img = cum_img/max(cum_img)
    cum_ref = cum_ref/max(cum_ref)
    n = 255
    img_val_eq = (cum_img/max(cum_img)*n).round()
    img_ref_eq = (cum_ref/max(cum_ref)*n).round()
    new_values = np.zeros((n))
    img_new = np.copy(img)
    j = 0   
    for i in range(n): 
        while True:
            if j > n-2 or abs(img_val_eq[i]-img_ref_eq[j])<abs(img_val_eq[i]-img_ref_eq[j+1]):
                new_values[i] = j
                break
            j = j + 1
    for i in range(n):
        img_new[img==i] = new_values[i]
    return img_new

im1 = np.array(Image.open('./images/jpg/trailer.jpg'))
im2 = np.array(Image.open('./images/jpg/tomato.jpg'))
im1_matched = np.zeros(im1.shape)
im1_matched[:,:,0] = hist_match(im1[:,:,0] ,im2[:,:,0])
im1_matched[:,:,1] = hist_match(im1[:,:,1] ,im2[:,:,1])
im1_matched[:,:,2] = hist_match(im1[:,:,2] ,im2[:,:,2])
im1_matched = uint8(im1_matched)
plt.imshow(im1_matched)
plt.show()
plt.plot(im_hist(im2[:,:,0]))
plt.show()
plt.plot(im_hist(im1_matched[:,:,0]))
plt.show()
