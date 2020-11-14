#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow import keras


def ExtremeGradientBoosting_Classifier(X_train,y_train):
    model = XGBClassifier(learning_rate=0.05, random_state=42, n_estimators= 100, max_depth=8)
    model.fit(X_train, y_train)
    return model
    
def SVM_Classifier(X_train, y_train):
    model = svm.NuSVC(gamma='auto', random_state=42)
    model.fit(X_train, y_train)
    return model


def GradientBoosting_Classifier(X_train, y_train):
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    return model
 
    
def RandomForest_Classifier(X_train,y_train):
    model = RandomForestClassifier(random_state=42, max_depth=6, criterion='gini')
    model.fit(X_train, y_train)
    return model


def NeuralNetwork_Classifier(X_train,y_train):
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
    

def Get_Model(classifier,X,y):
    X_train, y_train = X, y
    X_train = PrepareDataForModel(X_train)
    return classifierDict[classifier](X_train,y_train)
   
def Get_Prediction(classifier,entries, submission):
    prediction = np.zeros(entries)
    prediction = prediction+classifier.predict(submission)
    return pd.Series(prediction,dtype='int32')
    return prediction
 

def Feature_Engineering(df):
    df['Married_Women'] = np.where(df.Name.str.contains( "Mrs"), 1, 0)
    df['Boy'] = np.where(df.Name.str.contains('Master'), 1, 0)
    df['Girl'] = np.where(df.Name.str.contains('Miss'), 1, 0)
    df['has_Children_aboard'] = np.where((df['Parch']>0)& (df['Age']>=18), 1, 0)
    df['has_Parents_aboard'] = np.where((df['Parch']>0)& (df['Age']<18), 1, 0)
    df['is_alone_aboard'] = np.where((df['Parch']==0)& (df['SibSp']==0), 1, 0)
    df = pd.get_dummies(df, columns=["Pclass"],dtype=int)
    df = pd.get_dummies(df, columns=["Embarked"],dtype=int)
    df = pd.get_dummies(df, columns=["Sex"])
    return df


def Drop_Variables(df):
    # df = df.drop(['Sex'],axis=1)
    df = df.drop(['Sex_male'],axis=1)
    df = df.drop(['Parch'],axis=1)
    df = df.drop(['SibSp'],axis=1)
    # df = df.drop(['Girl'],axis=1)
    df = df.drop(['Married_Women'],axis=1)
    # df = df.drop(['Boy'],axis=1)
    # df = df.drop(['is_alone_aboard"],axis=1)
    df = df.drop(['has_Children_aboard'],axis=1)
    df = df.drop(['has_Parents_aboard'],axis=1)
    df = df.drop(["Embarked_S","Embarked_C","Embarked_Q"],axis=1)
    # df = df.drop(["Pclass_1","Pclass_2", "Pclass_3"],axis=1)
    df = df.drop(['Cabin'],axis=1)
    df = df.drop(['Name'],axis=1)
    df = df.drop(['Ticket'],axis=1)
    df = df.drop(['PassengerId'],axis=1)
    return df


def Feature_Analysis(df):
    print("Proportion of complete data within categories")
    for col in df.columns:
        part = np.round(df[col].count()/len(df[col])*100,2)
        print(col+": "+ str(part) +"%")
    
    women = df[df['Sex']=='female']
    women_survived = women[women['Survived']==1]
    women_not_survived =women[women['Survived']==0]  
    women["Married"] = np.where(women.Name.str.contains( "Ms"), 1, 0).astype(int)
    women_survived["Married"] = np.where(women_survived.Name.str.contains( "Ms"), 1, 0).astype(int)
    men = df[df['Sex']=='male']
    men_survived = men[men['Survived']==1]
    men_not_survived = men[men['Survived']==0]
    men["Is_Boy"] = np.where(men.Name.str.contains( "Master"), 1, 0).astype(int)
    men_survived["Is_Boy"] = np.where(men_survived.Name.str.contains( "Master"), 1, 0).astype(int)
    
    
    women_fig, women_axes = plt.subplots(nrows=3, ncols=2,figsize=(12, 10))
    men_fig, men_axes = plt.subplots(nrows=3, ncols=2,figsize=(12, 10))
    ax = sns.set_palette("ocean",2)
    
    ax = sns.distplot(women.Age.dropna(), bins=20, label = 'Total', ax = women_axes[0][0], kde =False)
    ax = sns.distplot(women_survived.Age.dropna(), bins=20, label = 'Survived', ax = women_axes[0][0], kde =False)
    ax.legend()
    ax.set_title('Female Passenger Total-Survival-Age Distribution')
    
    ax = sns.distplot(women.Fare.dropna(), bins=10, label = 'Total', ax = women_axes[0][1], kde =False)
    ax = sns.distplot(women_survived.Fare.dropna(), bins=10, label = 'Survived', ax = women_axes[0][1], kde =False)
    ax.legend()
    ax.set_title('Female Passenger Total-Survival-Fare Distribution')
    
    ax = sns.distplot(women.Parch.dropna(), bins=6, label = 'Total', ax = women_axes[1][0], kde =False)
    ax = sns.distplot(women_survived.Parch.dropna(), bins=6, label = 'Survived', ax = women_axes[1][0], kde =False)
    ax.legend()
    ax.set_title('Test')
    
    ax = sns.distplot(women.SibSp.dropna(), bins=8, label = 'Total', ax = women_axes[1][1], kde =False)
    ax = sns.distplot(women_survived.SibSp.dropna(), bins=8, label = 'Survived', ax = women_axes[1][1], kde =False)
    ax.legend()
    ax.set_title('Test')
    
    ax = sns.distplot(women.Married, bins=2, label = 'Total', ax = women_axes[2][0], kde =False)
    ax = sns.distplot(women_survived.Married, bins=2, label = 'Survived', ax = women_axes[2][0], kde =False)
    ax.legend()
    ax.set_title('Test') 
    
    ax = sns.distplot(men.Age.dropna(), bins=20, label = 'Total', ax = men_axes[0][0], kde =False)
    ax = sns.distplot(men_survived.Age.dropna(), bins=20, label = 'Survived', ax = men_axes[0][0], kde =False)
    ax.legend()
    ax.set_title('Male Passenger Total-Survival-Age Distribution')
    
    ax = sns.distplot(men.Fare.dropna(), bins=10, label = 'Total', ax = men_axes[0][1], kde =False)
    ax = sns.distplot(men_survived.Fare.dropna(), bins=10, label = 'Survived', ax = men_axes[0][1], kde =False)
    ax.legend()
    ax.set_title('Male Passenger Total-Survival-Fare Distribution')
    
    ax = sns.distplot(men.Parch.dropna(), bins=6, label = 'Total', ax = men_axes[1][0], kde =False)
    ax = sns.distplot(men_survived.Parch.dropna(), bins=6, label = 'Survived', ax = men_axes[1][0], kde =False)
    ax.legend()
    ax.set_title('Test')
    
    ax = sns.distplot(men.SibSp.dropna(), bins=8, label = 'Total', ax = men_axes[1][1], kde =False)
    ax = sns.distplot(men_survived.SibSp.dropna(), bins=8, label = 'Survived', ax = men_axes[1][1], kde =False)
    ax.legend()
    ax.set_title('Test')
    
    ax = sns.distplot(men.Is_Boy, bins=2, label = 'Total', ax = men_axes[2][0], kde =False)
    ax = sns.distplot(men_survived.Is_Boy, bins=2, label = 'Survived', ax = men_axes[2][0], kde =False)
    ax.legend()
    ax.set_title('Test')
    
    
    gender_fig, gender_axes = plt.subplots(nrows=1, ncols=4,figsize=(12, 10))
    gender_axes[0] = sns.catplot(data = df, x="Pclass", y="Survived", hue="Sex", kind='point')
    gender_axes[1] = sns.catplot(data = df, x="Parch", y="Survived", hue="Sex", kind='point')
    gender_axes[2] = sns.catplot(data = df, x="SibSp", y="Survived", hue="Sex", kind='point')
    gender_axes[2] = sns.catplot(data = df, x="Embarked", y="Survived", hue="Sex", kind='point')


def Determine_Model_Performance(classifier,X,y):
    print("Determine Best K-Fold Perfomance for given Classifier: " + classifier)
    max_fold = 12
    min_fold = 4
    fold_accuracy = np.zeros(max_fold)

    for fold in range(min_fold,max_fold):    
        kf = KFold(n_splits=fold, shuffle=True, random_state=42)
        counter = 0
        accuracy = np.zeros(kf.get_n_splits())
        for train,test in kf.split(X):
            X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test] 
            X_train = PrepareDataForModel(X_train)
            X_test = PrepareDataForModel(X_test)
            k_Fold_classifier = classifierDict[classifier](X_train,y_train)
            
            y_pred=k_Fold_classifier.predict(X_test)
            accuracy[counter] = metrics.accuracy_score(y_test, y_pred)
            counter+=1
        fold_accuracy[fold-1]= accuracy.mean()
    print("Classifier: "+ classifier)
    print("Maximum K-fold accuracy: " + str(fold_accuracy.max()))
    print("Best performing K: " + str(min_fold - 1 + fold_accuracy.argmax()))

def PrepareDataForModel(df):
    df['Age'] = df['Age'].fillna(np.mean(df['Age']))
    df['Fare'] = df['Fare'].fillna(np.mean(df['Fare']))
    scaler = StandardScaler()
    df[['Age','Fare']] = scaler.fit_transform(df[['Age','Fare']]).copy()
    return df
    
classifierDict = {
    "GradientBoosting": GradientBoosting_Classifier,
    "RandomForest": RandomForest_Classifier,
    "NeuralNetwork": NeuralNetwork_Classifier,
    "SVM": SVM_Classifier,
    "ExtremeGradientBoosting": ExtremeGradientBoosting_Classifier
} 

CLASSIFIER =  "RandomForest"


df = pd.read_csv('train.csv')
df.info()
df.describe()

Feature_Analysis(df)
df = Feature_Engineering(df)
df = Drop_Variables(df)

X = df.drop(['Survived'],axis=1)
y = df['Survived']
Determine_Model_Performance(CLASSIFIER,X,y)


model = Get_Model(CLASSIFIER, X,y)
 
test_df = pd.read_csv('test.csv')
test_df = Feature_Engineering(test_df)
test_df = Drop_Variables(test_df)
test_df = PrepareDataForModel(test_df)

y_pred = Get_Prediction(model,len(test_df),test_df)
submission_result = pd.read_csv('gender_submission.csv')
submission_result['Survived'] = pd.Series(y_pred,dtype="int32")
pd.DataFrame.to_csv(submission_result, 'result.csv',index=False)
