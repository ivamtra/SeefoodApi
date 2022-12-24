from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.ensemble import RandomForestClassifier
import joblib
# Load the model from the file
with open('hot_dog_rf.pkl', 'rb') as file:
  loaded_model = joblib.load(file)


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def fix_color(images):
  return np.flip(images, axis=-1) 

def scale(images):
  return_list = []

  for index, image in enumerate(images):
    im2 = Image.fromarray(images[index].astype('uint8'), 'RGB')
    im2 = im2.resize((256,256))
    im2 = np.array(im2)
    return_list.append(im2)
  return np.array(return_list)  


# Algo:
# 1. take in image
# 2. reshape to (1, n)
# 3. Reshape to (256,256)
# 4. Fix color
# 5. Flatten
# 6. Predict

def get_predicted_value(image):
    images = np.array(image)
    images = np.reshape(images, (1,) + image.shape)
    print(images.shape)
    images = fix_color(scale(images))
    images_c = np.reshape(images, (images.shape[0], int(images.size/images.shape[0])))
    predicted_value = loaded_model.predict(images_c)[0]
    return predicted_value



def test():
    return "Test"