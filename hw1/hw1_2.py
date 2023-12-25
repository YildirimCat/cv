import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2


def imge_goster(imge, uygulanan_islem_ismi):
    plt.imshow(imge, cmap='gray')
    plt.title(uygulanan_islem_ismi)
    plt.show()

def gamma_duzeltmesi(kaynak_imge):

    ters_imge = 255 - kaynak_imge #invert image
    kenetlenmis_imge = (100.0/255) * kaynak_imge + 100 #clamp to interval 100...200

    im4 = 255.0 * (kaynak_imge/255.0)**0.4 #squared

    im5 = 255*(kaynak_imge >= 50) #thresholded
    im6 = kaynak_imge
    mask = kaynak_imge < 10
    im6[mask]=0

    im_trans = np.concatenate((kaynak_imge, im4), axis=1)
    return im_trans

"""
cv2.convertScaleAbs(src[, dst[, alpha[, beta]]]) → dst

α: Kontrast ayarı faktörü. 1.0, hiçbir değişiklik anlamına gelir. Değer arttıkça kontrast artar.
β: Parlaklık ayarı. 0, parlaklık değişikliği olmaksızın kontrast ayarı yapılmasını sağlar. Değer arttıkça parlaklık artar.
"""

def parlaklik_kontrast_ayarla(kaynak_imge, alpha, beta):

    sonuc_imge = cv2.convertScaleAbs(kaynak_imge, alpha=alpha, beta=beta)
    birlestirilmis_sonuc = np.concatenate((kaynak_imge, sonuc_imge), axis=1)

    return birlestirilmis_sonuc

# İmgenin okunması
kaynak_imge = np.array(Image.open('../images/png/printedtext.png').convert('L'))


# 1) Parlaklik duzeltme islemi
parlaklik_ayarlanmis_imge = parlaklik_kontrast_ayarla(kaynak_imge, 1, 50)
imge_goster(parlaklik_ayarlanmis_imge, "Parlaklik Duzeltme Islemi | alpha=1, beta=50")

# 2) Kontrast duzeltme islemi
kontrast_ayarlanmis_imge = parlaklik_kontrast_ayarla(kaynak_imge, 2, 0)
imge_goster(kontrast_ayarlanmis_imge, "Kontrast Duzeltme Islemi | alpha=2, beta=0")

# 3) Parlaklik ve kontrast duzeltme islemi
parlaklik_kontrast_ayarlanmis_imge = parlaklik_kontrast_ayarla(kaynak_imge, 2, 50)
imge_goster(parlaklik_kontrast_ayarlanmis_imge, "Parlaklik ve Kontrast Duzeltme Islemi | alpha=2, beta=50")

# 4) Gamma düzeltilmesi islemi
gamma_duzeltilmis_imge = gamma_duzeltmesi(kaynak_imge)
imge_goster(gamma_duzeltilmis_imge, "Gamma Duzeltme Islemi")


# Görüntüyü ikili hale getirme
"""
Uygun eşik değerini bulmak için yaygın kullanılan yöntem Otsu yöntemidir.
"""
ret, ikili_imge = cv2.threshold(kaynak_imge, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imge_goster(ikili_imge, "Otsu Yontemi ile Ikili Imge")

imge_boyut = ikili_imge.shape
print("Imge Boyutu: ", imge_boyut)

# İsabet ve Iska Yontemi
isabet_kernel = np.array(([1, 0, 0, 0, 1], [0, 1, 0,1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 0, 0 ,0, 1]), dtype="int")
iska_kernel = np.array(([0, 1, 1, 1, 0], [1, 0, 1, 0, 1], [1, 1, 0, 1, 1], [1, 0, 1, 0, 1], [0, 1, 1, 1, 0]), dtype="int")
#birlestirilmis_kernel = np.array(([1, -1, 1], [-1, 1, -1], [1, -1, 1]), dtype="int")
#print(birlestirilmis_kernel)


"""
the hit-or-miss operation comprises three steps:

Erode image A with structuring element B1.
Erode the complement of image A ( Ac) with structuring element B2.
AND results from step 1 and step 2.

Result = (A⊖B1) ∩ (Ac⊖B2)
"""
def isabet_ve_iska_donusturumu(img, kernel_hit, kernel_miss):

    A = img
    B1 = kernel_hit
    B2 = kernel_miss
    Acomp = 255 - A

    A_erode_B1 = cv2.erode(A, B1, iterations=1)
    Ac_erode_B2 = cv2.erode(Acomp, B2, iterations=1)
    A_hit_miss_B1 = cv2.bitwise_and(A_erode_B1, Ac_erode_B2)

    return A_hit_miss_B1


#isabet_iska_donusturulmus_imge = isabet_ve_iska_donusturumu(ikili_imge, isabet_kernel, iska_kernel)
#imge_goster(isabet_iska_donusturulmus_imge, "Isabet ve Iska Donusturumu")

def bagli_bilesen_algoritmasi(imge, x_sablon):

    result = cv2.matchTemplate(imge, x_sablon, cv2.TM_CCOEFF_NORMED)
    esik_degeri = 0.8
    _, label, stat, _ = cv2.connectedComponentsWithStats((result > esik_degeri).astype(np.uint8))
    x_harfi_sayisi = len(stat)

    return x_harfi_sayisi

x_harfi_sablonu = cv2.imread('../../../../x.png', 0)
x_harfi_sayisi = bagli_bilesen_algoritmasi(ikili_imge, x_harfi_sablonu)
print(x_harfi_sayisi)   # 1 adet x harfi bulunmaktadir.