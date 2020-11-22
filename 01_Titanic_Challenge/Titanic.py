#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import classifier_utils
import feature_utils


# =============================================================================
# Available classifier keywords
# "GradientBoosting""
# "RandomForest"
# "NeuralNetwork
# "SVM"
# "ExtremeGradientBoosting"
# =============================================================================

CLASSIFIER =  "RandomForest"
FOLD = 10

# =============================================================================
# Training through K-folds on train data
# =============================================================================

df = pd.read_csv('train.csv') 
df.info()
df.describe()


feature_utils.feature_analysis(df) #Analyse Features
df = feature_utils.feature_engineering(df) #Extract and add features from existing ones
df = feature_utils.feature_selection(df) #Drop insignificant features

X = df.drop(['Survived'],axis=1) #Drop target variable
y = df['Survived'] #Define target variable

#Determine best performing model through K-Folds and varying classifier
classifier_utils.determine_model_performance(CLASSIFIER,X,y,FOLD) #Submit selected classifier and data for K-fold performance analysis
model = classifier_utils.get_model(CLASSIFIER, X,y) #train selected and parameter-tuned classifier on whole dataset
 
# =============================================================================
# Predictions on test data
# =============================================================================

test_df = pd.read_csv('test.csv')
test_df = feature_utils.feature_engineering(test_df) #Extract and add features from existing ones
test_df = feature_utils.feature_selection(test_df) #Drop insignificant features
test_df = feature_utils.data_model_preparation(test_df) #Fare+Price: Fill Na with mean values and use standard scale

y_pred = classifier_utils.get_prediction(model,len(test_df),test_df) #Get predictions on test data
submission_result = pd.read_csv('submission_template.csv') #Create dataframe from submission template
submission_result['Survived'] = pd.Series(y_pred,dtype="int32") #Paste predictions into dataframe
pd.DataFrame.to_csv(submission_result, 'result.csv',index=False) #save dataframe as new csv file
