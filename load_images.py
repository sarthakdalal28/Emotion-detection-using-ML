import cv2
from glob import glob
import numpy as np
import random
from sklearn.utils import shuffle
import pickle
import os

def pickle_images_labels(s):
	images_labels = []
	images = glob("{}/*/*.png".format(s))
	images.sort()
	for image in images:
		print(image)
		label = image[image.find(os.sep)+1: image.rfind(os.sep)]
		img = cv2.imread(image, 0)
		images_labels.append((np.array(img, dtype=np.uint8), label))
	return images_labels

images_labels = pickle_images_labels('train')
images_labels = shuffle(shuffle(shuffle(shuffle(images_labels))))
images, labels = zip(*images_labels)
print("Length of images_labels", len(images_labels))
with open("train_images", "wb") as f:
	pickle.dump(images, f)
del images
with open("train_labels", "wb") as f:
	pickle.dump(labels, f)
del labels
images_labels = pickle_images_labels('test')
images_labels = shuffle(shuffle(shuffle(shuffle(images_labels))))
images, labels = zip(*images_labels)
print("Length of images_labels", len(images_labels))
with open("test_images", "wb") as f:
	pickle.dump(images, f)
del images
with open("test_labels", "wb") as f:
	pickle.dump(labels, f)
del labels
