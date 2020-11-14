#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


#Create features through combining several featurs (1)
#Create features through extended analysis of existing ones (2)
#Create featurs through one-hot encoding (3)
def feature_engineering(df):
    df['Married_Women'] = np.where(df.Name.str.contains( "Mrs"), 1, 0) #(2)
    df['Boy'] = np.where(df.Name.str.contains('Master'), 1, 0) #(2)
    df['has_Children_aboard'] = np.where((df['Parch']>0)& (df['Age']>=18), 1, 0) #(1)
    df['has_Parents_aboard'] = np.where((df['Parch']>0)& (df['Age']<18), 1, 0) #(1)
    df['is_alone_aboard'] = np.where((df['Parch']==0)& (df['SibSp']==0), 1, 0) #(1)
    df = pd.get_dummies(df, columns=["Pclass"],dtype=int) #(3)
    df = pd.get_dummies(df, columns=["Embarked"],dtype=int) #(3)
    df = pd.get_dummies(df, columns=["Sex"]) #(3)
    return df

#Drop insignificant features
def feature_selection(df):
    # df = df.drop(['Sex'],axis=1)
    df = df.drop(['Sex_male'],axis=1)
    df = df.drop(['Parch'],axis=1)
    df = df.drop(['SibSp'],axis=1)
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

#Prepare data for classifier
#Important to split train and test data for training before
#Test data should not contain any information about train data through scaling 
#Fare+Price: Fill Na with mean values and use standard scale
def data_model_preparation(df):
    df['Age'] = df['Age'].fillna(np.mean(df['Age']))
    df['Fare'] = df['Fare'].fillna(np.mean(df['Fare']))
    scaler = StandardScaler()
    df[['Age','Fare']] = scaler.fit_transform(df[['Age','Fare']]).copy()
    return df

#Analysing male and female survival rate separated from each other
#Varying variables such as fare, married, class, age
def feature_analysis(df):
    print("Proportion of complete data within categories")
    for col in df.columns:
        part = np.round(df[col].count()/len(df[col])*100,2)
        print(col+": "+ str(part) +"%")
    
    women = df[df['Sex']=='female']
    women_survived = women[women['Survived']==1]
    women["Married"] = np.where(women.Name.str.contains( "Ms"), 1, 0).astype(int)
    women_survived["Married"] = np.where(women_survived.Name.str.contains( "Ms"), 1, 0).astype(int)
    
    men = df[df['Sex']=='male']
    men_survived = men[men['Survived']==1]
    men["Is_Boy"] = np.where(men.Name.str.contains( "Master"), 1, 0).astype(int)
    men_survived["Is_Boy"] = np.where(men_survived.Name.str.contains( "Master"), 1, 0).astype(int)
    
    
    women_fig, women_axes = plt.subplots(nrows=3, ncols=2,figsize=(12, 10))
    men_fig, men_axes = plt.subplots(nrows=3, ncols=2,figsize=(12, 10))
    ax = sns.set_palette("ocean",2)
    
    ax = sns.distplot(women.Age.dropna(), bins=20, label = 'Total', ax = women_axes[0][0], kde =False)
    ax = sns.distplot(women_survived.Age.dropna(), bins=20, label = 'Survived', ax = women_axes[0][0], kde =False)
    ax.legend()
    ax.set_title('Female Passenger Total - Survival - Age Distribution')
    
    ax = sns.distplot(women.Fare.dropna(), bins=10, label = 'Total', ax = women_axes[0][1], kde =False)
    ax = sns.distplot(women_survived.Fare.dropna(), bins=10, label = 'Survived', ax = women_axes[0][1], kde =False)
    ax.legend()
    ax.set_title('Female Passenger Total - Survival - Fare Distribution')
    
    ax = sns.distplot(women.Parch.dropna(), bins=6, label = 'Total', ax = women_axes[1][0], kde =False)
    ax = sns.distplot(women_survived.Parch.dropna(), bins=6, label = 'Survived', ax = women_axes[1][0], kde =False)
    ax.legend()
    ax.set_title('Female Passenger Total - Survival - (Number of family members aboard) Distribution')
    
    ax = sns.distplot(women.SibSp.dropna(), bins=8, label = 'Total', ax = women_axes[1][1], kde =False)
    ax = sns.distplot(women_survived.SibSp.dropna(), bins=8, label = 'Survived', ax = women_axes[1][1], kde =False)
    ax.legend()
    ax.set_title('Female Passenger Total - Survival - (Number of siblings aboard) Distribution')
    
    ax = sns.distplot(women.Married, bins=2, label = 'Total', ax = women_axes[2][0], kde =False)
    ax = sns.distplot(women_survived.Married, bins=2, label = 'Survived', ax = women_axes[2][0], kde =False)
    ax.legend()
    ax.set_title('Female Passenger Total - Survival - Married Women Distribution') 
    
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
    ax.set_title('Male Passenger Total - Survival - (Number of family members aboard) Distribution')
    
    ax = sns.distplot(men.SibSp.dropna(), bins=8, label = 'Total', ax = men_axes[1][1], kde =False)
    ax = sns.distplot(men_survived.SibSp.dropna(), bins=8, label = 'Survived', ax = men_axes[1][1], kde =False)
    ax.legend()
    ax.set_title('Male Passenger Total - Survival - (Number of siblings aboard) Distribution')
    
    ax = sns.distplot(men.Is_Boy, bins=2, label = 'Total', ax = men_axes[2][0], kde =False)
    ax = sns.distplot(men_survived.Is_Boy, bins=2, label = 'Survived', ax = men_axes[2][0], kde =False)
    ax.legend()
    ax.set_title('Boy Passenger Total - Survival Distribution')
    
    gender_fig, gender_axes = plt.subplots(nrows=1, ncols=4,figsize=(12, 10))
    gender_axes[0] = sns.catplot(data = df, x="Pclass", y="Survived", hue="Sex", kind='point')
    gender_axes[1] = sns.catplot(data = df, x="Parch", y="Survived", hue="Sex", kind='point')
    gender_axes[2] = sns.catplot(data = df, x="SibSp", y="Survived", hue="Sex", kind='point')
    gender_axes[2] = sns.catplot(data = df, x="Embarked", y="Survived", hue="Sex", kind='point')