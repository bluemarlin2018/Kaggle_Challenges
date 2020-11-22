#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers,models


#Load model from checkpoint in folder model/
def load_model(checkpoint):
    model = models.load_model("models/"+ checkpoint)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model
    
# Load kaggle images from test folder 
def load_images(test_folder, image_size):
    images = []
    for img in os.listdir(test_folder):
        img = os.path.join(test_folder, img)
        img = image.load_img(img, target_size=(image_size[0], image_size[1]))
        img = image.img_to_array(img)/255.0
        print(img)
        img = np.expand_dims(img, axis=0)
        images.append(img)
        
    images = np.vstack(images)
    return images

# Use CNN model to predict on test images using batches 
def predict(model, images):
    classes = model.predict_classes(images, batch_size=32)
    return classes

# =============================================================================
# Call this function first
# Initiliaze model from saved checkpoint
# Load images from test folder
# Get predictions
# =============================================================================
def init_test(test_folder, image_size, checkpoint):     
    model = load_model(checkpoint)
    images = load_images(test_folder, image_size)
    classes = predict(model, images)
    return classes