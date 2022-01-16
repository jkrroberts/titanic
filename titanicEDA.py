# titanicEDA.py
# Exploratory Data Analysis for titanic data

import pandas as pd
import numpy as np
import os

# set the path of the raw data
#C:\jkr\PythonCode\DSwithPython\venv\Scripts\titanic\data\raw
raw_data_path = os.path.abspath('C:\jkr\PythonCode\DSwithPython\\venv\Scripts\\titanic\\data\\raw')
train_file_path = os.path.join(raw_data_path, 'train.csv')
test_file_path = os.path.join(raw_data_path, 'test.csv')
# very useful in knowing if the path is correct.
# os.listdir(raw_data_path)

# read the data with all default parameters
train_df = pd.read_csv(train_file_path, index_col='PassengerId')
test_df = pd.read_csv(test_file_path, index_col="PassengerId")

type(train_df)

# provides meta data about the df. Column names, non null counts, & data types
train_df.info()
test_df.info()

# add survived column to the test df to make them equal.
test_df['Survived'] = -888

# axis = 0 performs a union of both df's. axis = 1 is a join
df = pd.concat((train_df, test_df),axis=0)

df.head()
df.tail()

# selection does the same thing.
df.Name
df['Name']

#Multiple list of columns
df[['Name', 'Age']]

# selects all columns starting with row 5 to 10
df.loc[5:10,]

# provides row and a range of columns. Order matters for the colunn range
df.loc[5:10, 'Pclass' : 'Age']

# provide row range and specific columns.
df.loc[5:10, ['Survived', 'Fare', 'Embarked']]

# position based indexing. iloc[ row range, column range] count from zero
df.iloc[5:10, 3:8]

# gets only male passengers then prints the number
male_passengers = df.loc[df.Sex == 'male',:]
print('Number of male passengers : {0}'.format(len(male_passengers)))

male_passengers_first_class = df.loc[(((df.Sex == 'male') & df.Pclass == 1)),:]
print('Number of male passengers in first class : {0}'.format(len(male_passengers_first_class)))

# EDA summary statistics module

# provides a statistical break down of the data frame.
df.describe()

# numerical feature
# centrality measures
print('Mean fare : {0}'.format(df.Fare.mean()))
print('Median fare : {0}'.format(df.Fare.median()))

# dispersion measures
print('Min fare : {0}'.format(df.Fare.min()))
print('Max fare : {0}'.format(df.Fare.max()))
print('Fare range : {0}'.format(df.Fare.max() - df.Fare.min()))
print('25 percentile : {0}'.format(df.Fare.quantile(.25)))
print('50 percentile : {0}'.format(df.Fare.quantile(.50)))
print('75 percentile : {0}'.format(df.Fare.quantile(.75)))
print('Variance fare : {0}'.format(df.Fare.var()))
print('Standard deviation fare : {0}'.format(df.Fare.std()))

# box plot
df.Fare.plot(kind='box')

# Categorical data
# include = all allows the describe function to work on non-numeric values
df.describe(include='all')

# gives value and count of those values within the column
df.Sex.value_counts()
# gives percentage of values
df.Sex.value_counts(normalize=True)

df[df.Survived != -888].Survived.value_counts()
df.Pclass.value_counts()

# visualize the counts
df.Pclass.value_counts().plot(kind='bar')
df.Pclass.value_counts().plot(kind='bar',rot = 0, title='Class wise passenger count', color='c');