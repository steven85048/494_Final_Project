import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #for plotting
from sklearn.ensemble import RandomForestClassifier #for the model
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz #plot tree
from sklearn.metrics import roc_curve, auc #for model evaluation
from sklearn.metrics import classification_report #for model evaluation
from sklearn.metrics import confusion_matrix #for model evaluation
from sklearn.model_selection import train_test_split #for data splitting
import eli5 #for purmutation importance
from eli5.sklearn import PermutationImportance
import shap #for SHAP values
from pdpbox import pdp, info_plots #for partial plots
np.random.seed(123)

# columns:
# country,year,sex,age,suicides_no,population,suicides/100k pop,country-year,HDI for year, gdp_for_year ($) ,gdp_per_capita ($),generation

# ---------------------------------------------------------------
# ----------------------- DATA CLEANING -------------------------
# ---------------------------------------------------------------

dataFrame = pd.read_csv('../data/suicide_cleaned.csv')

# Certain columns already accounted for in other columns
dataFrame.drop(['country-year', 'suicides/100k pop'], axis = 1)

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