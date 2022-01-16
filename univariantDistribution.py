# univariantDistribution.py
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

# test_df does not have a survived column. This adds it with a default value.
test_df['Survived'] = -888

df = pd.concat((train_df, test_df))

# create histograms
df.Age.plot(kind='hist', title='histogram for Age', color='c');

df.Age.plot(kind='hist', title='histogram for Age', color='c', bins=20);

# create kde plots
df.Age.plot(kind='kde', title='Density plot for Age', color='c');

print('skewness for age : {0:.2f}'.format(df.Age.skew()))