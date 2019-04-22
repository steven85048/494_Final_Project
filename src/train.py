import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #for plotting
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
#from sklearn.tree import export_graphviz #plot tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
#from sklearn.metrics import roc_curve, auc #for model evaluation
#from sklearn.metrics import classification_report #for model evaluation
#from sklearn.metrics import confusion_matrix #for model evaluation
#from sklearn.model_selection import train_test_split #for data splitting
from sklearn.model_selection import cross_val_predict, GridSearchCV
import eli5 #for purmutation importance
from eli5.sklearn import PermutationImportance
import shap #for SHAP values
from pdpbox import pdp, info_plots #for partial plots

from IPython.display import display, HTML

np.random.seed(123)

DATA_DIR = '../data/StudentsPerformance.csv'
OUTPUT_DIR = '../output/'

FOREST_OUT_DOT = 'random_forest.dot'
FOREST_OUT_PNG = 'random_forest.png'

# columns:
# generation, race/ethnicity, parental level of education, lunch, test preparation course, math score, reading score, writing score

# ---------------------------------------------------------------
# ----------------------- DATA CLEANING -------------------------
# ---------------------------------------------------------------

dataFrame = pd.read_csv(DATA_DIR)

print( dataFrame.describe() )

# print( dataFrame.dtypes )

# Convert certain columns into categorical variables
dataFrame['gender'] = dataFrame['gender'].astype('object')
dataFrame['race/ethnicity'] = dataFrame['race/ethnicity'].astype('object')
dataFrame['parental level of education'] = dataFrame['parental level of education'].astype('object')
dataFrame['lunch'] = dataFrame['lunch'].astype('object')
dataFrame['test preparation course'] = dataFrame['test preparation course'].astype('object')


# Hot encoding = Convert categorical column values into a separate column each with values of 0 or 1
# depending on if it is or if it is not
dataFrame = pd.get_dummies( dataFrame, drop_first=True)

'''
# Try using LabelEncoder instead of hot encoding with dummies
labelEncoder = LabelEncoder()

for col in ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']:
    dataFrame[col] = labelEncoder.fit_transform( dataFrame[col] )
'''

dataFrame['average test score'] = ( dataFrame['math score'] + dataFrame['writing score'] + dataFrame['reading score'] ) / 3

# There's only 2438 columns! (due to the number of countries)
# print(dataFrame.head())

# Check for NaNs
# print( dataFrame.isnull().sum() )

# Final type check
print( dataFrame.info() )

# ---------------------------------------------------------------
# ----------------- HYPERPARMETER SEARCHING ---------------------
# ---------------------------------------------------------------

y = dataFrame[['average test score']].values
X = dataFrame.drop(['math score', 'reading score', 'writing score', 'average test score'], axis=1).values

'''
# Standardize quantitative data to 0-1 range
scalar = MinMaxScaler( feature_range = (0, 1) )
X = scalar.fit_transform(X)
y = scalar.fit_transform(y)
'''

'''
# We optimize our hyperparameters with grid search:
# @param max_depth = number of leaves in the tree
# @param n_estimators = number of trees in the model (will be summed up in regression)
gridSearch = GridSearchCV(  estimator = RandomForestRegressor(),
                            param_grid = {  'max_depth': range(4, 20),
                                           'n_estimators': range(10, 100) },
                            cv = 5,
                            scoring = 'neg_mean_squared_error',
                            verbose = 0,
                            n_jobs = -1 )

gridResult = gridSearch.fit(X, y.ravel())
gridBestParams = gridResult.best_params_

print( gridBestParams )
'''

# ---------------------------------------------------------------
# ----------------- FINAL MODEL ---------------------------------
# ---------------------------------------------------------------

gridBestParams = {}
gridBestParams["max_depth"] = 4
gridBestParams["n_estimators"] = 18

forestModel = RandomForestRegressor ( max_depth = gridBestParams["max_depth"],
                                      n_estimators = gridBestParams["n_estimators"],
                                      random_state = False,
                                      verbose = False )

cvScore = cross_val_score( forestModel, X, y.ravel(), cv = 40, scoring = 'explained_variance' )
print("The final cross-validation score is: ")
print(cvScore.mean())

perm = PermutationImportance( forestModel.fit(X, y), random_state = 42 ).fit(X, y)
print( eli5.explain_weights(perm, feature_names = dataFrame.drop(['math score', 'reading score', 'writing score', 'average test score'], axis = 1).columns.tolist(), top = 10) )