import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.exposure import cumulative_distribution

def hist_cizdir(imge, baslik):

    # İmgenin RGB kanallarına ayrılması
    R, B, G = cv2.split(imge)
    channels = [R, G, B]
    colors = ('r', 'g', 'b')

    # Her bir kanalın histogramının çizdirilmesi
    for channel, col in zip(channels, colors):
        plt.hist(channel.flat, bins=256, range=(0, 255), color=col)
        plt.xlabel("Piksel Değerleri")
        plt.ylabel("Piksel Sayısı")
        plt.title(baslik)
        plt.show()


# İmge isleme sonucunun çizdirilmesi
def plotResult(imInput, imTemplate, imResult):
    plt.figure(figsize=(10,7))
    plt.subplot(1,3,1)
    plt.title('Kaynak İmge')
    plt.imshow(imInput, cmap='gray')
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.title('Hedef İmge')
    plt.imshow(imTemplate, cmap='gray')
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.title('Eşlenmiş İmge')
    plt.imshow(imResult, cmap='gray')
    plt.axis('off')
    plt.show()

# İmgenin okunması
kaynak_imge = cv2.imread('../images/jpg/trailer.jpg')
rgb_im = cv2.cvtColor(kaynak_imge, cv2.COLOR_BGR2RGB)
# İmgenin RGB kanallarına ayrılması
R, B, G = cv2.split(rgb_im)

hist_cizdir(rgb_im, "Kaynak İmge RGB Histogram")

# Her bir kanalda histogram eşitleme işleminin yapılması
equalized_hist_R = cv2.equalizeHist(R)
equalized_hist_G = cv2.equalizeHist(G)
equalized_hist_B = cv2.equalizeHist(B)

equalized_hist = [equalized_hist_R, equalized_hist_G, equalized_hist_B]
merged_eqaulized_hist = cv2.merge(equalized_hist)
hist_cizdir(merged_eqaulized_hist, "Kaynak İmge Eşitlenmiş Histogram")

colors = ('r', 'g', 'b')
# Her bir kanalın eşitlenmiş histogramının birleştirilip çizdirilmesi
for i, col in enumerate(colors):
    hist = cv2.calcHist([kaynak_imge], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])
plt.title("Kaynak İmge Birleştirilmiş Eşitlenmiş Histogram")
plt.show()

hedef_imge = cv2.imread('../images/jpg/tomato.jpg')
merged_equalized_hist = cv2.merge((equalized_hist_R, equalized_hist_G, equalized_hist_B))

# CDF hesaplanması
def cdf_hesapla(image):
    cdf, bins = cumulative_distribution(image)
    cdf = np.insert(cdf, 0, [0]*bins[0])
    cdf = np.append(cdf, [1]*(255-bins[-1]))
    return cdf

# histogram eslemenin yapilmasi
def hist_esleme(cdfInput, cdfTemplate, imageInput):
    pixels = np.arange(256)
    pixelValues = np.arange(256)
    new_pixels = np.interp(cdfInput, cdfTemplate, pixels)
    imageMatch = (np.reshape(new_pixels[imageInput.ravel()], imageInput.shape)).astype(np.uint8)
    return imageMatch


# CDF hesaplanması
cdf_kaynak_imge = cdf_hesapla(kaynak_imge)
cdf_hedef_imge = cdf_hesapla(hedef_imge)
esleme_sonuc = hist_esleme(cdf_kaynak_imge, cdf_hedef_imge, kaynak_imge)

hist_cizdir(hedef_imge, "Hedef İmge RGB Histogram (tomato.jpg)")
hist_cizdir(esleme_sonuc, "Eşlenmiş İmge RGB Histogram (trailer.jpg)")
plotResult(kaynak_imge, hedef_imge, esleme_sonuc)











