#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from xgboost import XGBClassifier

from sklearn.model_selection import KFold
from sklearn import metrics

import feature_utils

# =============================================================================
# Classifier Section:
# =============================================================================

#Training best performing, 
#Test submission second best performing so far
#Further parameter tuning possible
def extreme_gradient_boosting_classifier(X_train,y_train): 
    model = XGBClassifier(learning_rate=0.05, random_state=42, n_estimators= 100, max_depth=8)
    model.fit(X_train, y_train)
    return model
    
def svm_classifier(X_train, y_train):
    model = svm.NuSVC(gamma='auto', random_state=42)
    model.fit(X_train, y_train)
    return model


def gradient_boosting_classifier(X_train, y_train):
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    return model
 
#Training second best performing, 
#Test submission best performing so far
#Further parameter tuning possible
def random_forest_classifier(X_train,y_train): 
    model = RandomForestClassifier(random_state=42, max_depth=6, criterion='gini')
    model.fit(X_train, y_train)
    return model

#Worst Performing, model construction 
#Other sequential model build might be better performing
#Takes too much time to find promising architecture: stopped
def neural_network_classifier(X_train,y_train): 
        model = keras.Sequential([
        keras.layers.Flatten(input_shape=(4,)),
        keras.layers.Dense(8, activation=tf.nn.relu),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)])
        model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=1)
        return model
    

# =============================================================================
# Model Section
# =============================================================================

def determine_model_performance(classifier,X,y):
    print("Determine Best K-Fold Perfomance for given Classifier: " + classifier)
    max_fold = 12 
    min_fold = 4
    fold_accuracy = np.zeros(max_fold) #Accuracy array to store k-fold performance for each k

    for fold in range(min_fold,max_fold):    
        kf = KFold(n_splits=fold, shuffle=True, random_state=42) 
        counter = 0
        accuracy = np.zeros(kf.get_n_splits())
        for train,test in kf.split(X):
            X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test] 
            
            #Important to split train and test data for training before fillna and scaling
            #Test data should not contain any information about train data through scaling
            X_train = feature_utils.data_model_preparation(X_train) #Fare+Price: Fill Na with mean values and use standard scale
            X_test = feature_utils.data_model_preparation(X_test)
            k_Fold_classifier = classifierDict[classifier](X_train,y_train) #train selected classifier on K-fold train data
            
            y_pred=k_Fold_classifier.predict(X_test) #predict with selected classifier on K-fold test data
            accuracy[counter] = metrics.accuracy_score(y_test, y_pred) #store accuracy of k-fold test data in array
            counter+=1
        fold_accuracy[fold-1]= accuracy.mean() #k-fold classifier performance for given k = mean accuracy over all k-folds for given k
    print("Classifier: "+ classifier)
    print("Maximum K-fold accuracy: " + str(fold_accuracy.max()))
    print("Best performing K: " + str(min_fold - 1 + fold_accuracy.argmax()))


#Train model on whole dataset with given classifier
def get_model(classifier,X,y):
    X_train, y_train = X, y
    X_train = feature_utils.data_model_preparation(X_train) #Fare+Price: Fill Na with mean values and use standard scale
    return classifierDict[classifier](X_train,y_train)

#Get predictions on whole test data  
def get_prediction(classifier,entries, submission):
    prediction = np.zeros(entries)
    prediction = prediction+classifier.predict(submission) #Every prediction {0,1} that is not zero changes prediction array
    return pd.Series(prediction,dtype='int32')
    return prediction

#Dictionary of classifier functions to simplify switching classifier through chaning keywords
classifierDict = {
    "GradientBoosting": gradient_boosting_classifier,
    "RandomForest": random_forest_classifier,
    "NeuralNetwork": neural_network_classifier,
    "SVM": svm_classifier,
    "ExtremeGradientBoosting": extreme_gradient_boosting_classifier}