# exploringProcessingDataDemo1.py
import pandas as pd
import numpy as np
import os

# set the path for the raw data
raw_data_path = os.path.abspath('C:\jkr\PythonCode\DSwithPython\\venv\Scripts\\titanic\\data\\raw')
train_file_path = os.path.join(raw_data_path, 'train.csv')
test_file_path = os.path.join(raw_data_path, 'test.csv')

# read the data with all default parameters
train_df = pd.read_csv(train_file_path, index_col='PassengerId')
test_df = pd.read_csv(test_file_path, index_col="PassengerId")

# get the type
type(train_df)

# provides column names number of values and data types
train_df.info()

test_df.info()

train_df['Survived']

# test_df does not have a survived column. This adds it with a default value.
test_df['Survived'] = -888

df = pd.concat((train_df, test_df))

# axis=0 is a full dataframe join. the axis parameter controls how the data is joined.
df = pd.concat((train_df, test_df), axis=0)
# axis=1 joins by columns
df = pd.concat((train_df, test_df), axis=1)

df.info()

df.head()
df.tail()

# below 2 commands do the same thing. One via the dot notation the other via pandas select
df.Name
df['Name']
df[['Name', 'Age']]

# loc is a search function. The below finds passenger ids from 5 to 10
df.loc[5:10,]
df.loc[5:10, 'Pclass' : 'Age']

# below is passengerid from 5 to 10 and columns 3 to 8
df.iloc[5:10, 3:8]

# filter rows based on the condition. This creates another df with only male passengers included in it.
male_passengers = df.loc[(df.Sex == 'male'),:]
print('Number of male Passengers: {0}'.format(len(male_passengers)))

# use & or | operators to build complex logic
male_passengers_first_class = df.loc[((df.Sex == 'male') & (df.Pclass == 1)),:]
print('Number of male Passengers in First Class: {0}'.format(len(male_passengers_first_class)))

# Summary statistics on the data frame
df.describe()

print('Mean fare : (0)'.format(df.Fare.mean()))
df.Fare.mean()