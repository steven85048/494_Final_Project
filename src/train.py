from data_clean import group_output_type, smote_over_sample, under_sample_random, under_sample_centroids
from testing_utils import print_class_count, plot_roc_curve, pca_plot_2d

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from imblearn.ensemble import BalancedBaggingClassifier, RUSBoostClassifier, EasyEnsembleClassifier, BalancedRandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score, roc_curve
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
dataFrame = group_output_type( dataFrame )

# ---------------------------------------------------------------
# ----------------- HYPERPARMETER SEARCHING ---------------------
# ---------------------------------------------------------------

skf = StratifiedKFold(n_splits = 10)
data_splits = []

y = dataFrame[['y']].values
#X = dataFrame.drop(['y'], axis = 1).values
X = dataFrame[[ 
'x.HVcpx',
'x.BIC3',
'x.TI1',
'x.Jhetp',
'x.Yindex',
'x.MAXDN',
'x.TI2',
'x.D.Dr05',
'x.SEigZ',
'x.IC1',
'x.Vindex',
'x.J',
'x.WA',
'x.IVDE',
'x.IDE',
'x.AECC',
'x.ICR',
'x.SEigm',
'x.IC2',
'x.DECC',
'x.BIC2',
]].values

feature_names = dataFrame.drop(['y'], axis = 1).columns

for train_index, test_index in skf.split( X, y ):
    train_x, test_x = X[train_index], X[test_index]
    train_y, test_y = y[train_index], y[test_index]
    data_splits.append( [ train_x, test_x, train_y, test_y ] )

# ---------------------------------------------------------------
# ---------------------- FINAL MODEL ----------------------------
# ---------------------------------------------------------------

scores_df = pd.DataFrame( columns = ['ROC_AUC', 'Accuracy', 'F1'] )

#forestModel = RandomForestClassifier (  n_estimators = 30, criterion = 'gini', max_features = 'sqrt' )
#forestModel = GradientBoostingClassifier( learning_rate = .2 , n_estimators = 60 )
#forestModel = AdaBoostClassifier( n_estimators = 100, learning_rate = 1 )
#forestModel = BalancedBaggingClassifier( n_estimators = 500, max_samples = .8, bootstrap_features = True )
forestModel = BalancedRandomForestClassifier( n_estimators = 500 )

for cvIndex, data_set in enumerate( data_splits ):
    train_x, test_x, train_y, test_y = data_set[0], data_set[1], data_set[2], data_set[3]

    # oversample with SMOTE
    train_x, train_y = under_sample_random( train_x, train_y )
    test_x, test_y = under_sample_random(test_x, test_y)
    print_class_count( train_y )

    # apply scalar normalization
    scalar = StandardScaler()
    test_x = scalar.fit_transform(test_x)
    train_x = scalar.fit_transform(train_x)

    '''
    # Plot reduced PCA plot
    pca_plot_2d( train_x, train_y )
    '''

    # fit the model to the training data and predict the y with the resultant model
    forestModel.fit( train_x, train_y )
    pred_y = forestModel.predict( test_x )

    # Evaluate the split
    roc_auc = roc_auc_score( test_y, pred_y )
    accuracy = accuracy_score( test_y, pred_y )
    f1 = f1_score( test_y, pred_y )

    conf_matrix = confusion_matrix( test_y, pred_y )
    print(conf_matrix)

    '''
    # Plot the ROC Curve
    test_probs = forestModel.predict_proba( test_x )
    test_probs_pos = test_probs[:, 1]
    false_positive_rate, true_positive_rate, thresholds = roc_curve( test_y, test_probs_pos )

    plot_roc_curve( false_positive_rate, true_positive_rate )
    '''

    scores_df = scores_df.append({'ROC_AUC' : roc_auc, 'Accuracy' : accuracy, 'F1' : f1}, ignore_index = True)

print( "Scores of the stratified CV: ")
print( scores_df )

print("ROC_AUC Mean: ")
print( scores_df['ROC_AUC'].mean() )

print( "Most important features: ")
feature_importance_list = []
for feature_pair in zip( feature_names, forestModel.feature_importances_ ):
    feature_importance_list.append(feature_pair)

feature_importance_list.sort(key = lambda x: x[1] )
for feature_pair in feature_importance_list[10:]:
    print(feature_pair)