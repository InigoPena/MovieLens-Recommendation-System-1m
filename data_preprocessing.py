import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


##=======================================================================================================================================
## Movies & Ratings Merged Data
##=======================================================================================================================================

movie_items = pd.read_csv('ml-1m/movies.dat', sep='::',
                          engine='python', names=['MovieID', 'Title', 'Genres'],
                          encoding='ISO-8859-1')

movies = movie_items.drop('Genres', axis=1)

ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', engine='python',
                      names=['UserID', 'MovieID', 'Rating', 'Timestamp'])

merged_dataset = pd.merge(ratings, movies, how='inner', on='MovieID')

# Encoding users and movie titles
user_enc = LabelEncoder()
merged_dataset['User'] = user_enc.fit_transform(merged_dataset['UserID'].values)
n_users = merged_dataset['User'].nunique()

item_enc = LabelEncoder()
merged_dataset['Movie'] = item_enc.fit_transform(merged_dataset['Title'].values)
n_movies = merged_dataset['Movie'].nunique()

merged_dataset['Rating'] = merged_dataset['Rating'].values.astype(np.float32) # Convert ratings to float32 for efficient storage
print(merged_dataset.head())
print('Merged dataset shape: {}'.format(merged_dataset.shape))

if not merged_dataset.empty:
    merged_dataset.to_csv('ml-1m/merged.csv', index=False)

##=======================================================================================================================================
## Train, Test & Validation Split
##=======================================================================================================================================

def split_data(data,  split_ratio=(0.2, 0.5)):

    """Split the data into train, validation and test sets"""
    train, temp = train_test_split(data, test_size=split_ratio[0], random_state=42)
    val, test = train_test_split(temp, test_size=split_ratio[1], random_state=42)

    return train, val, test

train, val, test = split_data(merged_dataset)

print('Train shape: {}'.format(train.shape))
print('Validation shape: {}'.format(val.shape))
print('Test shape: {}'.format(test.shape))

if not train.empty & val.empty & test.empty:
    train.to_csv('ml-1m/train.csv', index=False)
    val.to_csv('ml-1m/val.csv', index=False)
    test.to_csv('ml-1m/test.csv', index=False)
