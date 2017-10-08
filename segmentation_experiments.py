import numpy as np
import cv2
from matplotlib import pyplot as plt

# Read in carotid image
carotid_img = cv2.imread("Carotid.jpg")
grayscale = cv2.cvtColor(carotid_img, cv2.COLOR_BGR2GRAY)

# Adaptive Thresholding
B1 = cv2.adaptiveThreshold(grayscale, maxValue = 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType = cv2.THRESH_BINARY_INV, blockSize = 5,C = 15)
B2 = cv2.adaptiveThreshold(grayscale, maxValue = 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType = cv2.THRESH_BINARY, blockSize = 5,C = 15)
B3 = cv2.adaptiveThreshold(grayscale, maxValue = 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType = cv2.THRESH_BINARY_INV, blockSize = 5,C = 15)
B4 = cv2.adaptiveThreshold(grayscale, maxValue = 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType = cv2.THRESH_BINARY, blockSize = 5,C = 15)

# Storing images
images = [B1,B2,B3,B4]
# Titles
titles = ["Adaptive Mean - Binary Inv","Adaptive Mean - Binary","Adaptive Gaussian - Binary Inv","Adaptive Gaussian - Binary"]


cv2.imshow("grayscale",grayscale)
cv2.waitKey(0)
# Show Images
for i in xrange(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()