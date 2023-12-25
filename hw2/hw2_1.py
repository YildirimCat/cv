import numpy as np
import cv2
import matplotlib.pyplot as plt
import cv.hw1.hw1_3 as hw1

##################################################################################
# Bir önceki ödevdeki kaynak kodları kullanarak DoG ölçek uzayının oluşturulması #
##################################################################################

kaynak_imge = cv2.imread('../images/png/street.png')
kaynak_imge = cv2.cvtColor(kaynak_imge, cv2.COLOR_BGR2GRAY)

h, w = kaynak_imge.shape[:2]

# Gauss gürültüsünün oluşturulup uygulanması
gauss_gurultusu = np.random.normal(0, 30, (h, w))
gurultulu_imge = kaynak_imge + gauss_gurultusu
kirpilmis_imge = gurultulu_imge[:400, :400]


# Ölçek uzayının oluşturulması
olcek_uzayi = hw1.olcek_uzayi_olustur(kirpilmis_imge)

# DoG uzayının oluşturulması
DoG = hw1.DoG_olustur(olcek_uzayi)

#########################################################################################
# Yerel Minimum ve Yerel Maksimum Noktalarının Hessian Matrisini Kullanarak Tespit Etme #
#########################################################################################

def find_local_extrema(dog_scale_space):

    # Yerel ekstremum noktalarının tutulacağı liste
    keypoints = []

    for scale_idx in range(1, len(dog_scale_space) - 1):
        current_scale = dog_scale_space[scale_idx]
        above_scale = dog_scale_space[scale_idx + 1]
        below_scale = dog_scale_space[scale_idx - 1]

        for i in range(1, current_scale.shape[0] - 1):
            for j in range(1, current_scale.shape[1] - 1):

                # Bahsi geçen 26 komşuluğun kontrolü
                neighbors = np.array([
                    above_scale[i-1:i+2, j-1:j+2],
                    current_scale[i-1:i+2, j-1:j+2],
                    below_scale[i-1:i+2, j-1:j+2]
                ])

                # Hessian matrisini hesapla
                hessian_matrix = compute_hessian_matrix(neighbors, sigma=1.6)

                # Hessian matrisine bağlı koşulu kontrol et
                if is_edge_point(hessian_matrix):
                    keypoints.append((j, i, scale_idx))

    return keypoints

def compute_hessian_matrix(neighbors, sigma):
    # Gauss Filtresi
    neighbors_smoothed = cv2.GaussianBlur(neighbors, (0, 0), sigma)

    # İkinci dereceden türevlerin kullanılması
    gradient_xx = cv2.Sobel(neighbors_smoothed, cv2.CV_64F, 2, 0, ksize=3)
    gradient_yy = cv2.Sobel(neighbors_smoothed, cv2.CV_64F, 0, 2, ksize=3)
    gradient_xy = cv2.Sobel(neighbors_smoothed, cv2.CV_64F, 1, 1, ksize=3)

    # Gradyanların sigma ile ağırlıklandırılması
    weight = np.exp(-(np.arange(-1, 2) ** 2) / (2 * sigma ** 2))
    weight = weight / np.sum(weight)

    # Hessian Matrisi
    hessian_matrix = np.array([
        [np.sum(gradient_xx * weight), np.sum(gradient_xy * weight)],
        [np.sum(gradient_xy * weight), np.sum(gradient_yy * weight)]
    ])

    return hessian_matrix


# Hessian matrisi ile kosulun kontrolü
def is_edge_point(hessian_matrix):

    trace_squared = (hessian_matrix[0,0] + hessian_matrix[1,1])**2
    det = hessian_matrix[0,0]*hessian_matrix[1,1] - hessian_matrix[0,1]*hessian_matrix[1,0]

    ratio = trace_squared / det

    lower_limit = 0
    upper_limit = 12

    return lower_limit <= ratio <= upper_limit

# Orijinal görüntü ve ilgi noktalarının çizdirilmesi
def draw_keypoints(image, keypoints, radius=2, color=(0, 0, 255), thickness=1):
    drawn_image = image.copy()

    for keypoint in keypoints:
        center = (int(keypoint[0]), int(keypoint[1]))
        drawn_image = cv2.circle(drawn_image, center, radius, color, thickness)

    return drawn_image

def main():

    # Yerel ekstremum noktalarını bul
    keypoints = find_local_extrema(DoG)

    original_image = cv2.imread("C:\\Users\\Yldrm\\Desktop\\YL\\Bilgisayarla_Gorme\\Lab\\cv\\images\\png\\street.png")  # Gerçek dosya yoluyla değiştirilmeli
    expanded_image = draw_keypoints(original_image, keypoints)

    # Genişletilmiş görüntüyü gösterme
    cv2.imshow("Image with Keypoints", expanded_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
