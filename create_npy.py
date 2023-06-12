import os
import cv2
import numpy as np
from keras.utils import to_categorical

def create(s):

    emotions = ['happy', 'neutral', 'sad', 'angry', 'surprised', 'disgusted', 'fearful']
    label_map = {emotion: idx for idx, emotion in enumerate(emotions)}
    
    dataset_dir = '{}'.format(s)

    # Initialize lists to store the images and labels
    images = []
    labels = []

    # Traverse the directory structure
    for emotion in emotions:
        emotion_dir = os.path.join(dataset_dir, emotion)
        for filename in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, filename)

            img = cv2.imread(img_path, 0)
            #img = img[20:40, 15:35]
            images.append(img)
            labels.append(label_map[emotion])

    # Convert the lists to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)

    labels = to_categorical(labels, num_classes=len(emotions))

    # Save the dataset to .npy files
    np.save('{}_data.npy'.format(s), images)
    np.save('{}_labels.npy'.format(s), labels)

create('train')
create('test')