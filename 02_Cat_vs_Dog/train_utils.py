#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
from matplotlib import  pyplot as plt
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import data_utils

# =============================================================================
# Build model with 3 Conv layer
# Batch Normalization and Dropout improved learning performance
# Parameter Tuning on architecture still necessary
# =============================================================================

def build_model(image_width,image_height):
    model = tf.keras.models.Sequential()
    
    model.add(layers.Conv2D(48,(3,3),activation = 'relu', input_shape = (image_width,image_height,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(64,(3,3),activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(128,(3,3),activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(256,activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()
    
    opt = tf.keras.optimizers.RMSprop(learning_rate=0.002)
    model.compile(optimizer=opt ,loss='binary_crossentropy',metrics =['accuracy'])
    return model

# =============================================================================
# Batch train model for several epochs  on train data provided by keras image generators
# =============================================================================
def train_model(model,train_generator,val_generator, epoch, batch_size):
    
    #Create name for best performing checkpoint
    date_string = str(datetime.datetime.now().strftime("%M-%H-%d-%m-%Y-"))
    checkpoint = ModelCheckpoint("models/"+date_string+"model.hdf5", monitor='loss', verbose=1,
        save_best_only=True, mode='auto', period=1)
    
    #train model
    history = model.fit(
        train_generator,
        steps_per_epoch = train_generator.samples //batch_size, #perform batch training
        validation_data = val_generator,
        validation_steps = val_generator.samples // batch_size, #batch eval on validation set
        epochs = epoch,
        callbacks = [checkpoint] # pass checkpoint information to save best model
        )
    return history


# =============================================================================
# Plot the model train and val performance over epochs
# Todo: plot losses
# =============================================================================
def get_model_performance(history,model,val_generator):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    test_loss, test_acc = model.evaluate(val_generator, verbose=2)


# =============================================================================
# To start training call init_train.
# It creates training and validation data by using keras image helpers
# The model is built from scratch
# Performance acurracy is plot
# =============================================================================

def init_train(EPOCH,image_size,batch_size, train_DIR, test_DIR):
    image_width = image_size[0]
    image_height = image_size[1]
    train_generator, val_generator = data_utils.get_training_data(image_size, batch_size, train_DIR)
    model = build_model(image_width,image_height) #Built CNN model from scratch
    history = train_model(model, train_generator, val_generator, EPOCH, batch_size) #train model
    get_model_performance(history, model, val_generator) #Get the performance on train and val data of the model over epochs
    