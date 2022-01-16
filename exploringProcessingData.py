# exploringProcessingData.py
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# set the path for the raw data
raw_data_path = os.path.abspath('C:\jkr\PythonCode\DSwithPython\\venv\Scripts\\titanic\\data\\raw')
train_file_path = os.path.join(raw_data_path, 'train.csv')
test_file_path = os.path.join(raw_data_path, 'test.csv')

# read the data with all default parameters
train_df = pd.read_csv(train_file_path, index_col='PassengerId')
test_df = pd.read_csv(test_file_path, index_col="PassengerId")

# test_df does not have a survived column. This adds it with a default value.
test_df['Survived'] = -888

df = pd.concat((train_df, test_df))

df.info()

df[df.Embarked.isnull()]

# provides a distinct count of each value in the column.
df.Embarked.value_counts()

# which embarked point has higher survival counts
pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived != -888].Embarked)

#impute the missing values with 'S'
# df.loc[df.Embarked.isnull(), 'Embarked'] = 'S'
# df.Embarked.fillna('S', inplace=True)

# option 2 : explore the fare of each class for each embarkment point
df.groupby(['Pclass', 'Embarked']).Fare.median()

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!RUN THIS
# replace the missing values with 'C'
df.Embarked.fillna('C', inplace=True)

# check if any null value remaining
df[df.Embarked.isnull()]

# resolving null values for Fare
df[df.Fare.isnull()]
# passengerID 1044 has null fare. Embarked from S and is Pclass 3, median fare is 8.05
df.groupby(['Pclass', 'Embarked']).Fare.median()

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!RUN THIS
# median fare for passengers in 3rd class embarking from S
median_fare = df.loc[(df.Pclass == 3) & (df.Embarked == 'S'),'Fare'].median()
print(median_fare)
df.Fare.fillna(median_fare, inplace=True)
#df.Fare.fillna(8.05, inplace=True)
df[df.Fare.isnull()]
df.loc[1044]

# Resolving Age null values
df.info()
df[df.Age.isnull()]
df.Age.plot(kind='hist', bins=20, color='c')
df.groupby('Sex').Age.median()
df[df.Age.notnull()].boxplot('Age', 'Sex');
# if you wanted to use sex as the driver for age this code would update the df
# age_sex_median = df.groupby('Sex').Age.transform('median')
# df.Age.fillna(age_sex_median, inplace=True)

df[df.Age.notnull()].boxplot('Age', 'Pclass');

# exploring name
df.Name
# get the persons title from the name field
def GetTitle(name):
    first_name_with_title = name.split(',')[1]
    title = first_name_with_title.split('.')[0]
    title = title.strip().lower()
    return title

# use map function to apply the function on each Name value row i
# the x within the map function represent df.Name
df.Name.map(lambda x : GetTitle(x)) # alternative code df.Name.map(GetTitle)
# return only unique values
df.Name.map(lambda x : GetTitle(x)).unique()

#!!!!!!!!!!!!!!!!!!!!!!RUN THIS
# revised GetTitle function that will map each title to a title group.
def GetTitle(name):
    title_group = {'mr': 'Mr',
                   'mrs' : 'Mrs',
                   'miss' : 'Miss',
                   'master' : 'Master',
                   'don' : 'Sir',
                   'rev' : 'Sir',
                   'dr' : 'Officer',
                   'mme' : 'Mrs',
                   'ms' : 'Mrs',
                   'major' : 'Officer',
                   'lady' : 'Lady',
                   'sir' : 'Sir',
                   'mlle' : 'Miss',
                   'col' : 'Officer',
                   'capt' : 'Officer',
                   'the countess' : 'Lady',
                   'jonkheer' : 'Sir',
                   'dona' : 'Lady'
    }
    first_name_with_title = name.split(',')[1]
    title = first_name_with_title.split('.')[0]
    title = title.strip().lower()
    return title_group.get(title)

# add a new column to the df and fill it with the title data from the GetTitle function.
df['Title'] = df.Name.map(lambda x : GetTitle(x))
df.Name.map(lambda x : GetTitle(x))
#!!!!!!!!!!!!!!!!!!!!!!!!!RUN DOWN TO HERE

df[df.Age.notnull()].boxplot('Age', 'Title');
# I think this is the most accurate way to get the median age for passengers that don't have an age
df.groupby(['Pclass', 'Title']).Age.median()

#!!!!!!!!!!!!!!!!!!!!!!!!RUN THIS
# replace the missing values
title_age_median = df.groupby('Title').Age.transform('median')
df.Age.fillna(title_age_median, inplace=True)

# create seperate df for the 263 people with a null age
dfAge = df[df.Age.isnull()]
dfAge.to_csv('Age.csv')
# looking at data by Pclass. Majority of people were in class 3
dfAge.groupby(['Pclass']).count()
dfAge.groupby(['Pclass', 'Embarked', 'Sex']).count()
dfAge1 = df[df.Age.notnull()]
dfAge1.groupby(['Pclass', 'Embarked', 'Sex']).Age.median()
df.groupby(['Pclass', 'Sex']).Age.median()

# How to deal with outliers
df.Age.plot(kind='hist', bins=20, color='c');
df.loc[df.Age > 70]

# Fare outliers
df.Fare.plot(kind='hist', title='histogram for Fare', bins=20, color='c')
df.Fare.plot(kind='box')
df.loc[df.Fare == df.Fare.max()]

# try some tranformations to reduce the skewness
LogFare = np.log(df.Fare + 1) # Adding 1 to accomdate zero fares : log(0) is not defined

# histogram of LogFare
LogFare.plot(kind='hist', color='c', bins=20)

# binning
pd.qcut(df.Fare, 4)
pd.qcut(df.Fare, 4, labels=['very_low', 'low', 'high', 'very_high'])
pd.qcut(df.Fare, 4, labels=['very_low', 'low', 'high', 'very_high']).value_counts()
pd.qcut(df.Fare, 4, labels=['very_low', 'low', 'high', 'very_high']).value_counts().plot(kind='bar', color='c', rot=0)

#!!!!!!!!!!!!!!!!!!RUN THIS
# create the fare bin attribute
df['Fare_Bin'] = pd.qcut(df.Fare, 4, labels=['very_low', 'low', 'high', 'very_high'])

# Feature Engineering
# AgeState based on Age
df['AgeState'] = np.where(df['Age'] >= 18, 'Adult', 'Child')
# AgeState counts
df['AgeState'].value_counts()
# crosstab
pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived != -888].AgeState)

#!!!!!!!!!!!!!!!!!!RUN THIS
# Family size feature
# Adding parents with siblings
df['FamilySize'] = df.Parch + df.SibSp + 1 # 1 for self
# explore the family feature
df['FamilySize'].plot(kind='hist', color='c')
# family with max family members
df.loc[df.FamilySize == df.FamilySize.max(), ['Name', 'Survived', 'FamilySize', 'Ticket', 'Pclass']]
pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived != -888].FamilySize)

#!!!!!!!!!!!!!!!!!!RUN THIS
# Mother feature
# a lady aged more than 18 who has Parch > 0 and is married (not Miss)
df['IsMother'] = np.where(((df.Sex == 'female') & (df.Parch > 0) & (df.Age > 18) & (df.Title != 'Miss')), 1, 0)
# crosstab with IsMother
pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived !=-888].IsMother)

# Deck feature
# explore cabin
df.Cabin
# use unique to get unique values for Cabin feature
df.Cabin.unique()
# look at the cabin = T
df.loc[df.Cabin == 'T']
# set the value to NaN
df.loc[df.Cabin == 'T', 'Cabin'] = np.NaN
# look at the unique values of Cabin again
df.Cabin.unique()

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!RUN THIS
# extract first character of the cabin string to the deck
def get_deck(cabin):
    return np.where(pd.notnull(cabin),str(cabin) [0].upper(),'Z')
df['Deck'] = df['Cabin'].map(lambda x : get_deck(x))

df.Deck.value_counts()
pd.crosstab(df[df.Survived !=-888].Survived, df[df.Survived !=-888].Deck)

# Categorical feature encoding.
# binary encoding works when you only have 2 values and can set them to 0 or 1
# Lable encoding assigning values a numeric value 1=low, 2=medium, etc
# one-hot encoding is creating a feature for each value for instance Embarked values is A, B, or C you would create
# a new column of Is_A set to 0 or 1 Is_B set to 0 or 1, etc

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!RUN THIS
# sex
df['IsMale'] = np.where(df.Sex == 'male', 1, 0)
# columns Desk, Pclass, Title, AgeState
# get_dummies function will perform the categorical encoding for you.
df = pd.get_dummies(df,columns=['Deck', 'Pclass', 'Title', 'Fare_Bin', 'Embarked', 'AgeState'])
print(df.info())

#Drop and Reorder columns
df.drop(['Cabin', 'Name', 'Ticket', 'Parch', 'SibSp', 'Sex'], axis=1, inplace=True)
# reorder columns
columns = [column for column in df.columns if column != 'Survived']
columns = ['Survived'] + columns
df = df[columns]

# Save processed data to datasets
processed_data_path = os.path.abspath('C:\jkr\PythonCode\DSwithPython\\venv\Scripts\\titanic\\data\\processed')
write_train_path = os.path.join(processed_data_path, 'train.csv')
write_test_path = os.path.join(processed_data_path, 'test.csv')
# train data
df.loc[df.Survived !=-888].to_csv(write_train_path)
# test data
columns = [column for column in df.columns if column != 'Survived']
df.loc[df.Survived == -888, columns].to_csv(write_test_path)

# Advanced visualizations with MatPlotlib
plt.hist(df.Age)
plt.hist(df.Age, bins=20, color='c')
plt.show()

plt.hist(df.Age, bins=20, color='c')
plt.title('Histogram : Age')
plt.xlabel('Bins')
plt.ylabel('Counts')
plt.show()

f , ax = plt.subplots()
ax.hist(df.Age, bins=20, color='c')
ax.set_title('Histogram : Age')
ax.set_xlabel('Bins')
ax.set_ylabel('Counts')
plt.show()

# working with the subplots
f , (ax1, ax2) = plt.subplots(1, 2, figsize=(14,3))

ax1.hist(df.Fare, bins=20, color='C')
ax1.set_title('Histogram : Fare')
ax1.set_xlabel('Bins')
ax1.set_ylabel('Counts')

ax2.hist(df.Age, bins=20, color='b')
ax2.set_title('Histogram : Age')
ax2.set_xlabel('Bins')
ax2.set_ylabel('Counts')

plt.show()