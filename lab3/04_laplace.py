import numpy as np
from PIL import Image


im = np.array(Image.open('C:\\Users\\Yldrm\\Desktop\\YL\\Bilgisayarla_Gorme\\Lab\\lab1\\images\\jpg\\baby.jpg').convert('L'))

im = np.float32(im)




