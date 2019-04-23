import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #for plotting
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import export_graphviz #plot tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer, confusion_matrix
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

DATA_DIR = '../data/drug-discovery.csv'
OUTPUT_DIR = '../output/'

# ---------------------------------------------------------------
# ----------------------- DATA CLEANING -------------------------
# ---------------------------------------------------------------

dataFrame = pd.read_csv(DATA_DIR)

print( dataFrame.shape )
print( dataFrame.dtypes )

# ---------------------------------------------------------------
# ----------------- HYPERPARMETER SEARCHING ---------------------
# ---------------------------------------------------------------

y = dataFrame[['y']].values
X = dataFrame.drop(['y'], axis=1).values

# Standardize quantitative data
scalar = MinMaxScaler()
X = scalar.fit_transform(X)

# We optimize our hyperparameters with grid search:
# @param max_depth = number of leaves in the tree
# @param n_estimators = number of trees in the model (will be summed up in regression)
gridSearch = GridSearchCV(  estimator = RandomForestClassifier(),
                            param_grid = {  'max_depth': range(4, 20),
                                           'n_estimators': range(10, 100) },
                            cv = 5,
                            scoring = 'accuracy',
                            verbose = 0,
                            n_jobs = -1 )

gridResult = gridSearch.fit(X, y.ravel())
gridBestParams = gridResult.best_params_

print( gridBestParams )

# ---------------------------------------------------------------
# ---------------------- FINAL MODEL ----------------------------
# ---------------------------------------------------------------

'''
gridBestParams = {}
gridBestParams["max_depth"] = 4
gridBestParams["n_estimators"] = 18
'''

forestModel = RandomForestClassifier ( max_depth = gridBestParams["max_depth"],
                                       n_estimators = gridBestParams["n_estimators"],
                                       criterion = 'entropy',
                                       random_state = False,
                                       verbose = False )

# CV-Score
'''
cvScore = cross_val_score( forestModel, X, y.ravel(), cv = 5, scoring = 'accuracy' )
print("The final cross-validation score is: ")
print(cvScore.mean())
'''

# Confusion Matrix
predictedY = cross_val_predict( forestModel, X, y.ravel(), cv = 5 )
confMat = confusion_matrix( y, predictedY )
print(confMat)

'''
perm = PermutationImportance( forestModel.fit(X, y), random_state = 42 ).fit(X, y)
print( eli5.explain_prediction_tree_classifier(perm, feature_names = dataFrame.drop(['y'], axis = 1).columns.tolist(), top = 10) )
'''