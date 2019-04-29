import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.preprocessing import MinMaxScaler

# Groups 1's and 2's into 1's
def group_output_type( dataFrame ):
    dataFrame['y'][dataFrame['y'] == 1 ] = 1
    dataFrame['y'][dataFrame['y'] == 2 ] = 1

    return dataFrame

# Random sample - returns x and y
def under_sample_random( X, y ):
    rus = RandomUnderSampler()

    x_rus, y_rus = rus.fit_sample( X, y )
    return x_rus, y_rus

# SMOTE - upsample less frequent values
def smote_over_sample( X, y ):
    smote = SMOTE(ratio = 'minority')
    xSmote, ySmote = smote.fit_sample(X, y)
    return xSmote, ySmote

def over_sample_random( X, y ):
    ros = RandomOverSampler()

    x_ros, y_ros = ros.fit_sample( X, y )
    return x_ros, y_ros

# Scale the X with minmaxscalar
def min_max_scalar( X ):
    scalar = MinMaxScaler()
    X = scalar.fit_transform(X)
    return X

def preprocess_test_set( X, y ):
    X = min_max_scalar(X)
    X, y = under_sample_random( X, y )
    return X, y

# ---------------------------- PUBLIC METHODS ------------------------------

def preprocess_data_frame( dataFrame ):
    dataFrame = group_output_type( dataFrame )
    return dataFrame

# num_split = number of zero/one samples that should be used for the training set
def preprocess_split_data( dataFrame, percent_split ):
    one_samples = dataFrame.query('y == 1')
    zero_samples = dataFrame.query('y == 0')

    one_sample_train = one_samples.sample(frac = percent_split )
    one_sample_test = one_samples.drop(one_sample_train.index)

    zero_sample_train = zero_samples.sample(frac = percent_split )
    zero_sample_test = zero_samples.drop(zero_sample_train.index)

    train_data_frame_list = [ one_sample_train, zero_sample_train ]
    test_data_frame_list = [ one_sample_test, zero_sample_test ]

    train_data_frame = pd.concat( train_data_frame_list )
    test_data_frame = pd.concat( test_data_frame_list )

    train_y = train_data_frame[['y']].values
    train_x = train_data_frame.drop(['y'], axis = 1).values

    test_y = test_data_frame[['y']].values
    test_x = test_data_frame.drop(['y'], axis = 1).values

    train_x, train_y = preprocess_test_set( train_x, train_y )
    test_x = min_max_scalar(test_x)

    return train_x, train_y, test_x, test_y