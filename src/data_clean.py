import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, TomekLinks, AllKNN, InstanceHardnessThreshold, NearMiss, RepeatedEditedNearestNeighbours
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.preprocessing import MinMaxScaler

# Groups 1's and 2's into 1's
def group_output_type( dataFrame ):
    dataFrame['y'][dataFrame['y'] == 1 ] = 1
    dataFrame['y'][dataFrame['y'] == 2 ] = 1

    return dataFrame

# Random sample - returns x and y
def under_sample_random( X, y ):
    rus = RandomUnderSampler( random_state = 30 )

    x_rus, y_rus = rus.fit_sample( X, y )
    return x_rus, y_rus

def under_sample_centroids( X, y ):
    cc_x = X
    cc_y = y

    cc = RandomUnderSampler()
    #cc2 = SMOTEENN( )
    #cc3 = InstanceHardnessThreshold()
    #cc4 = RandomUnderSampler()

    cc_x, cc_y = cc.fit_sample( cc_x, cc_y )
    #cc_x, cc_y = cc2.fit_sample( cc_x, cc_y )
    #cc_x, cc_y = cc3.fit_sample( cc_x, cc_y )
    #cc_x, cc_y = cc4.fit_sample( cc_x, cc_y )

    return cc_x, cc_y 

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