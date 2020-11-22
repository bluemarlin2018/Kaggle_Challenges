#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import datetime
import numpy as np
import pandas as pd


import train_utils
import test_utils
import data_utils
            



# Specify training parameters
EPOCH = 1
image_size = (128, 128)
batch_size = 32
train_DIR = "train"
test_DIR = "test/"
checkpoint = ""


##Only call when kaggle data is unprocessed, prepares data for keras image utility functions
# data_utils.rename_images(train_folder) 

#train model and save checkpoint
train_utils.init_train(EPOCH,image_size,batch_size, train_DIR, test_DIR) 

#get predictions using Kaggle Test images 
classes = test_utils.init_test(test_DIR, image_size, checkpoint) 

# prepare Kaggle submission results
submission_result = pd.read_csv('sample_submission.csv') #Create dataframe from submission template
submission_result['label'] = classes #Paste predictions into dataframe
pd.DataFrame.to_csv(submission_result, 'result.csv',index=False) #save dataframe as new csv file
    









