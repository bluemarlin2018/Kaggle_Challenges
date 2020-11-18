#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import numpy as np
import cv2
from matplotlib import  pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import layers,models




def rename_images(folder):
    os.mkdir(folder+"/cat")
    os.mkdir(folder+"/dog")
    for filename_old in os.listdir(folder):
        if (len(filename_old)>3 and filename_old[0]!="."):
            print(filename_old)
            filename_new = filename_old[0:3] + "_image_" + filename_old[4:]
            print (filename_new)
            Path(folder+"/"+filename_old).rename(folder+"/"+filename_old[0:3]+"/"+filename_new)
    

    
        
        


image_size = (128, 128)
batch_size = 32
folder = "train"
# rename_images(folder)

train_data = tf.keras.preprocessing.image_dataset_from_directory("train",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=image_size,
    batch_size=batch_size,
)

val_data = tf.keras.preprocessing.image_dataset_from_directory("train",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=image_size,
    batch_size=batch_size,
)


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)



model = tf.keras.models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation = 'relu', input_shape = (image_size[0],image_size[1],3)))
model.add(layers.MaxPooling2D((3,3)))
model.add(layers.Conv2D(64,(3,3),activation = 'relu'))
model.add(layers.MaxPooling2D((3,3)))
model.add(layers.Conv2D(32,(2,2),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

opt = tf.keras.optimizers.RMSprop(learning_rate=0.001)
model.compile(optimizer=opt ,loss='binary_crossentropy',metrics =['accuracy'])


history = model.fit(train_data, epochs=15, validation_data=val_data)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(val_data, verbose=2)
