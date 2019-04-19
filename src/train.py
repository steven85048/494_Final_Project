import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #for plotting
from sklearn.ensemble import RandomForestRegressor #for the model
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz #plot tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.metrics import roc_curve, auc #for model evaluation
from sklearn.metrics import classification_report #for model evaluation
from sklearn.metrics import confusion_matrix #for model evaluation
from sklearn.model_selection import train_test_split #for data splitting
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
# ---------------------------- MODEL ----------------------------
# ---------------------------------------------------------------

# Split the data into train/test; we use the number of suicides as the predictor (must also remove from the dataframe)
# [Tuning parameter] 80-20 train-test split
xTrain, xTest, yTrain, yTest = train_test_split(dataFrame.drop('suicides_no', 1), dataFrame['suicides_no'], test_size = .2, random_state = 11)

# [Tuning parameter] max_depth = 5
random_forest_model = RandomForestRegressor( random_state = 42 )
random_forest_model.fit( xTrain, yTrain )

y = dataFrame[['suicides_no']].values
X = dataFrame.drop('suicides_no', axis=1).values

random_forest_model = RandomForestRegressor( random_state = 42 )
print( cross_val_score(random_forest_model, X, y, cv = 5).mean() )

# Decision tree plot:
#estimator = random_forest_model.estimators_[1]
#feature_names = [i for i in xTrain.columns]
#yTrainStr = yTrain.astype('str')
#yTrainStr = yTrainStr.values

#tree_dot_dir = OUTPUT_DIR + FOREST_OUT_DOT
#tree_png_dir = OUTPUT_DIR + FOREST_OUT_PNG

#export_graphviz(estimator, 
#                out_file= tree_dot_dir, 
#                feature_names = feature_names,
#                rounded = True, proportion = True,
#                label = 'root',
#                precision = 2, filled = True)

#from subprocess import call
#call(['dot', '-Tpng', tree_dot_dir, '-o', tree_png_dir, '-Gdpi=600' ])

#from IPython.display import Image
#Image(filename = tree_png_dir)