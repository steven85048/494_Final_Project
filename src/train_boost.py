from data_clean import preprocess_data_frame, preprocess_split_data
from testing_utils import print_class_count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, StratifiedKFold
from sklearn.metrics import mean_squared_error, make_scorer, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

np.random.seed(111)

DATA_DIR = '../data/drug-discovery.csv'
OUTPUT_DIR = '../output/'

# ---------------------------------------------------------------
# ----------------------- DATA CLEANING -------------------------
# ---------------------------------------------------------------

dataFrame = pd.read_csv(DATA_DIR)
dataFrame = preprocess_data_frame( dataFrame )

# ---------------------------------------------------------------
# ----------------- HYPERPARMETER SEARCHING ---------------------
# ---------------------------------------------------------------

y = dataFrame[['y']].values
X = dataFrame.drop(['y'], axis=1).values

X, y = preprocess_split_data( X, y )

print_class_count(y)

# We optimize our hyperparameters with grid search:
# @param max_depth = number of leaves in the tree
# @param n_estimators = number of trees in the model (will be summed up in regression)
xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    silent=True, nthread=1)

# CV-Score
cvScore = cross_val_score( xgb, X, y.ravel(), cv = 5, scoring = 'accuracy' )
print("The final cross-validation score is: ")
print(cvScore.mean())

# Confusion Matrix
predictedY = cross_val_predict( xgb, X, y.ravel(), cv = 5 )
confMat = confusion_matrix( y, predictedY )
print(confMat)