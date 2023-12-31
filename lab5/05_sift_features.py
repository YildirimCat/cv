import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
# With jupyter notebook uncomment below line 
# %matplotlib inline 
# This plots figures inside the notebook



def compute_sift_features(img):
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp = sift.detect(gray,None) 
    cv2.drawKeypoints(img, kp, img)
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()



def compute_fast_det(img, is_nms=True, thresh = 10):
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector_create() #FastFeatureDetector()

#     # find and draw the keypoints
    if not is_nms:
        fast.setNonmaxSuppression(0)

    fast.setThreshold(thresh)

    img_fast = img
    kp = fast.detect(img,None)
    cv2.drawKeypoints(img_fast, kp, img, color=(255,0,0), flags=2)
    
    

    sift = cv2.SIFT()
    kp = sift.detect(gray,None)
    img_SIFT = img
    cv2.drawKeypoints(img_SIFT,kp,gray)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_SIFT, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def main():
    # read an image 
    img = cv2.imread('./images/jpg/yellowlily.jpg')
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

    # compute harris corners and display 
    compute_sift_features(img)
    
    # Do plot
    #plot_cv_img(gray, dst)

if __name__ == '__main__':
    main()