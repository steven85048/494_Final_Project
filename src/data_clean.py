import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
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

# Scale the X with minmaxscalar
def min_max_scalar( X ):
    scalar = MinMaxScaler()
    X = scalar.fit_transform(X)
    return X

def preprocess_data_frame( dataFrame ):
    dataFrame = group_output_type( dataFrame )
    return dataFrame

def preprocess_split_data( X, y ):
    X = min_max_scalar(X)
    X, y = smote_over_sample( X, y )
    return X, y
