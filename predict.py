import pickle
import cv2
import numpy as np

# Load the model from the pickle file
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Load and preprocess the input image
image_path = 'happy.jpg'
image = cv2.imread(image_path, 0)  # Read the image in grayscale
image = cv2.resize(image, (48, 48))  # Resize the image to match the input size used during training
image = np.expand_dims(image, axis=0)  # Add a batch dimension
#image = image.reshape(-1, 48, 48, 1).astype('float32')/255.0
# Perform any additional preprocessing, if required

# Make predictions on the image
predictions = loaded_model.predict(image)

# Get the predicted emotion label
emotions = ['happiness', 'neutral', 'sadness', 'anger', 'surprise', 'disgust', 'fear']
predicted_emotion = emotions[np.argmax(predictions)]

# Print the predicted emotion
print("Predicted emotion:", predicted_emotion)