#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import tensorflow as tf


# =============================================================================
# Create folder and data structure for keras image utilities
# Should be only used when raw data is downloaded from kaggle and is unprocessed
# =============================================================================
def rename_images(folder):
    os.mkdir(folder+"/cat")
    os.mkdir(folder+"/dog")
    for filename_old in os.listdir(folder):
        if (len(filename_old)>3 and filename_old[0]!="."):
            print(filename_old)
            filename_new = filename_old[0:3] + "_image_" + filename_old[4:]
            print (filename_new)
            Path(folder+"/"+filename_old).rename(folder+"/"+filename_old[0:3]+"/"+filename_new)
        
# =============================================================================
# Create training and validation data generator using keras image utility functions
# Data is augmented through rotation and shifting 
# Data is rescaled to [0,1]
# =============================================================================

def get_training_data(image_size, batch_size,train_folder):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1.0/255.0,
        validation_split=0.2,
        rotation_range=20,
        zoom_range = 0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range = 0.2,
        horizontal_flip=True,
        fill_mode = 'nearest')
    
    train_generator = train_datagen.flow_from_directory(
        directory = train_folder, 
        target_size = image_size, 
        batch_size = batch_size,
        class_mode="binary",
        subset = "training")
    
    val_generator = train_datagen.flow_from_directory(
        directory = train_folder,
        target_size = image_size,
        batch_size = batch_size,
        class_mode = "binary",
        subset = "validation")
    return train_generator, val_generator