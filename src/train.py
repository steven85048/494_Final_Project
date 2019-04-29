from data_clean import preprocess_data_frame, preprocess_split_data, smote_over_sample, under_sample_random
from testing_utils import print_class_count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #for plotting
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, GridSearchCV
import eli5 #for purmutation importance
from eli5.sklearn import PermutationImportance
import shap #for SHAP values
from pdpbox import pdp, info_plots #for partial plots

np.random.seed(123)

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

skf = StratifiedKFold(n_splits = 5)
data_splits = []

y = dataFrame[['y']].values
#X = dataFrame.drop(['y'], axis = 1).values
X = dataFrame[[ 
'x.Yindex',
'x.D.Dr05',
'x.Jhetp',
'x.J',
'x.SIC0', 
'x.CSI',
'x.VEA1', 
'x.ECC', 
'x.Xindex',
'x.LP1',
'x.Jhete',
'x.CIC1',
'x.IVDM',
'x.Vindex',
'x.BIC5',
'x.T.N..N.',
'x.IVDE',
'x.BIC3',
'x.BIC2',
'x.Eig1v',
'x.X2sol',
'x.IC2',
'x.HVcpx',
'x.IDE',
'x.ICR',
'x.SEigZ',
'x.SEigm' ]].values

feature_names = dataFrame.drop(['y'], axis = 1).columns

for train_index, test_index in skf.split( X, y ):
    train_x, test_x = X[train_index], X[test_index]
    train_y, test_y = y[train_index], y[test_index]
    data_splits.append( [ train_x, test_x, train_y, test_y ] )

# ---------------------------------------------------------------
# ---------------------- FINAL MODEL ----------------------------
# ---------------------------------------------------------------

scores_df = pd.DataFrame( columns = ['ROC_AUC', 'Accuracy', 'F1'] )

forestModel = RandomForestClassifier (  n_estimators = 100, bootstrap = False, criterion = 'entropy' )

for cvIndex, data_set in enumerate( data_splits ):
    train_x, test_x, train_y, test_y = data_set[0], data_set[1], data_set[2], data_set[3]

    # oversample with SMOTE
    train_x, train_y = under_sample_random( train_x, train_y )

    # apply scalar normalization
    scalar = StandardScaler()
    test_x = scalar.fit_transform(test_x)
    train_x = scalar.fit_transform(train_x)

    # fit the model to the training data and predict the y with the resultant model
    forestModel.fit( train_x, train_y )
    pred_y = forestModel.predict( test_x )

    # Evaluate the split
    roc_auc = roc_auc_score( test_y, pred_y )
    accuracy = accuracy_score( test_y, pred_y )
    f1 = f1_score( test_y, pred_y )

    conf_matrix = confusion_matrix( test_y, pred_y )
    print(conf_matrix)

    scores_df = scores_df.append({'ROC_AUC' : roc_auc, 'Accuracy' : accuracy, 'F1' : f1}, ignore_index = True)

print( "Scores of the stratified CV: ")
print( scores_df )

'''
print( "Most important features: ")
feature_importance_list = []
for feature_pair in zip( feature_names, forestModel.feature_importances_ ):
    feature_importance_list.append(feature_pair)

feature_importance_list.sort(key = lambda x: x[1] )
for feature_pair in feature_importance_list[10:]:
    print(feature_pair)
'''