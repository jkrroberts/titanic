# biVaraiantDistribution.py
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

df.plot.scatter(x='Age', y='Fare', color='c', title='scatter plot : Age vs Fare');

#Alpha sets the transparency of the scatter plot.
df.plot.scatter(x='Age', y='Fare', color='c', title='scatter plot : Age vs Fare', alpha=0.1);

df.plot.scatter(x='Pclass', y='Fare', color='c', title='scatter plot : Passenger Class vs Fare', alpha=0.15);

# grouping and aggregation
df.groupby('Sex').Age.median()

df.groupby('Pclass').Fare.median()
df.groupby('Pclass').Age.median()

df.groupby(['Pclass'])['Fare', 'Age'].median()

df.groupby(['Pclass']).agg({'Fare' : 'mean', 'Age' : 'median'})

# complex aggregation
aggregations = {
    'Fare': {    # work on the "Fare" column
        'mean_Fare': 'mean', # get the mean fare
        'median_Fare' : 'median', # get median fare
        'max_Fare': max,
        'min_Fare': np.min
    },
    'Age': {    # work on the "Age" column
        'median_Age': 'median', #Find the max, call the result "max_date"
        'min_Age': min,
        'max_Age': max,
        'range_Age': lambda x: max(x) - min(x) # Calculate the age range per group
    }
}
df.groupby(['Pclass']).agg(aggregations)

df.groupby(['Pclass', 'Embarked']).Fare.median()

pd.crosstab(df.Sex, df.Pclass)
pd.crosstab(df.Sex, df.Pclass).plot(kind='bar')

df.pivot_table(index='Sex',columns= 'Pclass', values='Age', aggfunc='mean')
df.groupby(['Sex', 'Pclass']).Age.mean()
df.groupby(['Sex', 'Pclass']).Age.mean().unstack()