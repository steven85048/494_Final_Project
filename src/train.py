import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #for plotting
from sklearn.preprocessing import MinMaxScaler
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
np.random.seed(123)

DATA_DIR = '../data/suicide_cleaned.csv'
OUTPUT_DIR = '../output/'

FOREST_OUT_DOT = 'random_forest.dot'
FOREST_OUT_PNG = 'random_forest.png'

# columns:
# country,year,sex,age,suicides_no,population,suicides/100k pop,country-year,HDI for year, gdp_for_year ($) ,gdp_per_capita ($),generation

# ---------------------------------------------------------------
# ----------------------- DATA CLEANING -------------------------
# ---------------------------------------------------------------

dataFrame = pd.read_csv(DATA_DIR)

# Certain columns already accounted for in other columns
# Drop HDI since there are too many NaN values
dataFrame = dataFrame.drop(['country-year', 'suicides/100k pop', 'HDI for year'], axis = 1)

# print( dataFrame.dtypes )

# Convert certain columns into categorical variables
dataFrame['country'] = dataFrame['country'].astype('object')
dataFrame['sex'] = dataFrame['sex'].astype('object')
dataFrame['generation'] = dataFrame['generation'].astype('object')

# Convert categorical column values into a separate column each with values of 0 or 1
# depending on if it is or if it is not
dataFrame = pd.get_dummies( dataFrame, drop_first=True)

# There's only 2438 columns! (due to the number of countries)
# print(dataFrame.head())

# Check for NaNs
# print( dataFrame.isnull().sum() )

# ---------------------------------------------------------------
# ----------------- HYPERPARMETER SEARCHING ---------------------
# ---------------------------------------------------------------


y = dataFrame[['suicides_no']].values
X = dataFrame.drop('suicides_no', axis=1).values

# Standardize quantitative data to 0-1 range
scalar = MinMaxScaler( feature_range = (0, 1) )
X = scalar.fit_transform(X)
y = scalar.fit_transform(y)

'''
# We optimize our hyperparameters with grid search:
# @param max_depth = number of leaves in the tree
# @param n_estimators = number of trees in the model (will be summed up in regression)
gridSearch = GridSearchCV(  estimator = RandomForestRegressor(),
                            param_grid = {  'max_depth': range(4,8),
                                           'n_estimators': (5, 10, 20, 30) },
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
gridBestParams["max_depth"] = 25
gridBestParams["n_estimators"] = 5

forestModel = RandomForestRegressor ( max_depth = gridBestParams["max_depth"],
                                      n_estimators = gridBestParams["n_estimators"],
                                      random_state = False,
                                      verbose = False )

cvScore = cross_val_score( forestModel, X, y.ravel(), cv = 10, scoring = 'explained_variance' )
print("The final cross-validation score is: ")
print(cvScore.mean())