import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import filters

kaynak_imge = cv2.imread('../images/png/street.png')
kaynak_imge = cv2.cvtColor(kaynak_imge, cv2.COLOR_BGR2GRAY)

h, w = kaynak_imge.shape[:2]

# Gauss gürültüsünün oluşturulup uygulanması
gauss_gurultusu = np.random.normal(0, 30, (h, w))
gurultulu_imge = kaynak_imge + gauss_gurultusu
kirpilmis_imge =  gurultulu_imge[:400, :400]

# Görselleştirme
plt.figure(figsize=(12, 6))
plt.subplot(131), plt.imshow(kaynak_imge)
plt.title('Kaynak İmge'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(gurultulu_imge)
plt.title('Gürültülü İmge'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(kirpilmis_imge)
plt.title('Kirpilmis İmge'), plt.xticks([]), plt.yticks([])
plt.show()


def olcek_uzayi_olustur(image):

    scale_space = []
    sigma = 1

    for i in range(8):
        filtered_image = filters.gaussian_filter(image, sigma)
        scale_space.append(filtered_image)
        sigma *= 1.5

    return scale_space

# Ölçek uzayının oluşturulması
olcek_uzayi = olcek_uzayi_olustur(kirpilmis_imge)

def DoG_olustur(scale_space):

    DoG = []
    for i in range(7):
        DoG.append(scale_space[i+1] - scale_space[i])

    return DoG

# DoG uzayının oluşturulması
DoG = DoG_olustur(olcek_uzayi)

def LoG_olustur(scale_space):

    LoG = []
    for i in range(8):
        LoG.append(filters.gaussian_laplace(scale_space[i], 1))

    return LoG

# LoG uzayının oluşturulması
LoG = LoG_olustur(olcek_uzayi)


def min_max_bul(imge):

    min_val = np.min(imge)
    max_val = np.max(imge)

    normalized_image = (imge - min_val) / (max_val - min_val)

    return normalized_image

# Görselleştirme
def gorselleri_cizdir(imge, length, title):
    plt.figure(figsize=(12, 6))
    for i in range(length):
        plt.subplot(2, 4, i+1), plt.imshow(imge[i], cmap='gray')
        plt.title(title + str(i+1)), plt.xticks([]), plt.yticks([])
    plt.show()


# Ölçek uzayı görselleştirme
normal_olcek_uzayi = min_max_bul(olcek_uzayi)
gorselleri_cizdir(normal_olcek_uzayi, 8, 'Ölçek Uzayı ')

# DoG uzayı görselleştirme
normal_DoG = min_max_bul(DoG)
gorselleri_cizdir(normal_DoG, 7, 'DoG Uzayı ')

# LoG uzayı görselleştirme
normal_LoG = min_max_bul(LoG)
gorselleri_cizdir(normal_LoG, 8, 'LoG Uzayı ')