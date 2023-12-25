from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math


# CTRL+K-C--> comment  CTRL+KU-->uncomment
# GENERATION OF SINGLE COLOR IMAGES AND CONCATENATION
im = np.array(Image.open('./images/jpg/tomato.jpg'))
im_R = im[:, :, 0]
im_G = im[:, :, 1]
im_B = im[:, :, 2]

im_RGB = np.concatenate((im_R, im_G, im_B), axis=1)
# im_RGB = np.hstack((im_R, im_G, im_B))
# im_RGB = np.c_['1', im_R, im_G, im_B]
imgplot = plt.imshow(im_RGB,'gray')  
plt.show()


