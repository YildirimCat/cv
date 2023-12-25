import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2


kaynak_imge = cv2.imread('../images/jpg/office_4.jpg')
kaynak_imge = cv2.cvtColor(kaynak_imge, cv2.COLOR_BGR2GRAY)

# sobel ile x ve y yönünde kısmi türevlerinin hesaplanması
x_sobel = cv2.Sobel(kaynak_imge,cv2.CV_64F,1,0,ksize=5)
y_sobel = cv2.Sobel(kaynak_imge,cv2.CV_64F,0,1,ksize=5)


# Eğim genliği ve açısının hesaplanması
egim_genligi = np.sqrt(np.square(x_sobel) + np.square(y_sobel))
egim_acisi = np.arctan2(y_sobel, x_sobel)

# Eğim genliği ve açısının görselleştirilmesi
plt.figure(figsize=(12, 6))

plt.subplot(131), plt.imshow(kaynak_imge, cmap='gray')
plt.title('Kaynak İmge'), plt.xticks([]), plt.yticks([])

plt.subplot(132), plt.imshow(egim_genligi, cmap='gray')
plt.title('Eğim Genliği'), plt.xticks([]), plt.yticks([])

plt.subplot(133), plt.imshow(egim_acisi, cmap='hsv')
plt.title('Eğim Açısı'), plt.xticks([]), plt.yticks([])

plt.show()


# Düşey kenarları vurgula
dusey_kenar = np.zeros_like(kaynak_imge, dtype=np.uint8)
dusey_kenar[(egim_acisi >= 0) & (egim_acisi <= 45)] = 255

# Yatay kenarları vurgula
yatay_kenar = np.zeros_like(kaynak_imge, dtype=np.uint8)
yatay_kenar[(egim_acisi >= 0) & (egim_acisi <= 180)] = 255

# Görselleştirme
plt.figure(figsize=(12, 6))

plt.subplot(131), plt.imshow(egim_acisi, cmap='hsv')
plt.title('Eğim Açısı'), plt.xticks([]), plt.yticks([])

plt.subplot(132), plt.imshow(dusey_kenar, cmap='gray')
plt.title('Belirli Açı Aralığındaki Dusey Kenarlar'), plt.xticks([]), plt.yticks([])

plt.subplot(133), plt.imshow(yatay_kenar, cmap='gray')
plt.title('Belirli Açı Aralığındaki Yatay Kenarlar'), plt.xticks([]), plt.yticks([])

plt.show()


# Kenarların ikili imgeye dönüştürülmesi ve görselleştirilmesi
ret, ikili_imge_yatay = cv2.threshold(yatay_kenar, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
ret, ikili_imge_dusey = cv2.threshold(dusey_kenar, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

birlestirilmis_sonuc = np.concatenate((ikili_imge_yatay, ikili_imge_dusey), axis=1)
plt.imshow(birlestirilmis_sonuc, cmap='gray')
plt.title("İkili Yatay ve Düşey Kenarlar")
plt.show()