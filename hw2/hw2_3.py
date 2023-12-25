import cv2
import numpy as np

def find_homography(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return homography, kp1, kp2, good_matches

def apply_homography(img1, img2, homography):
    result = cv2.warpPerspective(img1, homography, (img2.shape[1], img2.shape[0]))
    return result

def blend_images(img1, img2, homography, kp1, kp2, good_matches):
    # Homografi ile eğme (warping)
    img1_warped = cv2.warpPerspective(img1, homography, (img2.shape[1], img2.shape[0]))

    # Blend işlemi için maske oluşturma
    mask = np.zeros_like(img2)
    pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    cv2.fillConvexPoly(mask, np.int32(pts), (255, 255, 255), 8, 0)

    # İki görüntüyü blend işlemine tabi tutma
    blended = cv2.seamlessClone(img1_warped, img2, mask, (img2.shape[1]//2, img2.shape[0]//2), cv2.NORMAL_CLONE)

    return blended

# İki görüntüyü yükleme
img1 = cv2.imread('C:\\Users\\Yldrm\\Desktop\\YL\\Bilgisayarla_Gorme\\campus\\kampus1.jpg')
img2 = cv2.imread('C:\\Users\\Yldrm\\Desktop\\YL\\Bilgisayarla_Gorme\\campus\\kampus2.jpg')

# Homografi matrisini bulma
homography_matrix, kp1, kp2, good_matches = find_homography(img1, img2)

# Uyumlu ve aykırı noktaları çizdirme
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('Matches', img_matches)

# Homografik dönüşümü uygulama
result_image = apply_homography(img1, img2, homography_matrix)

# Blend işlemi uygulama
blended_image = blend_images(img1, img2, homography_matrix, kp1, kp2, good_matches)


print("Homography Matrix:")
print(homography_matrix)


# Sonuçları gösterme
cv2.imshow('Original Image 1', img1)
cv2.imshow('Original Image 2', img2)
cv2.imshow('Homography Transformation', result_image)
cv2.imshow('Blended Image', blended_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# C:\\Users\\Yldrm\\Desktop\\YL\\Bilgisayarla_Gorme\\campus\\kampus1.jpg