import os
import cv2
import numpy as np

img = cv2.imread('train/happy/im0.png', 0)
img = img[20:40,15:35]
cv2.imshow('img', img)
cv2.waitKey(0)