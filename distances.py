import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import time
import matplotlib.pyplot as plt
import operator
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.cross_validation import train_test_split, KFold
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min, log_loss, confusion_matrix, accuracy_score, auc
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from scipy.sparse import csr_matrix, vstack, hstack
from scipy.spatial.distance import minkowski
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier

# <editor-fold desc="Distances">

def pick_smallest(data, list = None, greatest = False):
    '''
    Finding smallest elements in each row of data according to list
    complexity don't depends on len of list
    data : 2-dim np.array
    list : list, np.array - list of minimum orders for output
    '''

    indices = np.argsort(data, axis=1)
    shape1 = list.__len__()
    result = np.array([[0.0] * shape1] * data.shape[0])
    for line in range(data.shape[0]):
        for num in range(shape1):
            if (greatest == False): place = list[num]
            else: place = data.shape[1] + 1 - list[num]
            result[line,num] = data[line,indices[line][place]]
    return result

def find_n_neighbors_multiclass(test,train,target_class,list_neighbors,n_classes,verbose=True):
    '''
    finding distances from each element of test to n-th further element of train part with corresponding target_class
    complexity don't depends on len of list_neighbors
    test, train: 2-dim np.arrays
    target_class: corresponding classes of train
    list_neighbors: list or np.array
    # substractions : whether to include substractions of distances to different classes. Works ONLY for n_classes = 2
    verbosity : whether to print finish message
    '''

    result = np.array([[0] * 0]* test.shape[0])
    for n_class in range(n_classes):
        part = train[target_class == n_class]
        distances = pairwise_distances(test,part,metric='euclidean',n_jobs=-1)
        result = np.hstack([result,pick_smallest(distances, list = list_neighbors)])

    if verbose: print('Neighbors found')
    return result

def find_n_neighbors_multiclass_meta(data, target, list_neighbors,n_classes, n_folds=2):
    # It is a meta-predictor of the find_n_neighbors_multiclass function. O(1-1/n) complexity

    # Generating indices
    split_list = list(KFold(n = data.shape[0], n_folds=n_folds, shuffle=True, random_state=0))
    result = np.array([[0.0] * (n_classes  * list_neighbors.__len__())] * data.shape[0])
    for num in range(n_folds):
        # Spliting data for train, validation and test folds
        train_fold_indices = split_list[num][0]
        test_fold_indices = split_list[num][1]
        train_fold = data[train_fold_indices]
        test_fold = data[test_fold_indices]
        train_target_fold = target[train_fold_indices]
        # test_target_fold = target.iloc[test_fold_indices]
        test_fold_distances = find_n_neighbors_multiclass(test_fold, train_fold, train_target_fold, list_neighbors,n_classes,verbose=False)
        result[test_fold_indices] = test_fold_distances
        print("Step #", num, "done")
    print("Cross_val_distance done!")
    return result

# list_neighbors = 2**np.arange(10)-np.ones(10)
#
# dist_train = find_n_neighbors_multiclass_meta(x_train,y_train,list_neighbors,n_folds=10)
# dist_val = find_n_neighbors_multiclass(x_val,x_train,y_train,list_neighbors)
# dist_holdout = find_n_neighbors_multiclass(x_holdout,x_train,y_train,list_neighbors)
# # dist_test = find_n_neighbors_multiclass(x_test,train_new,target_new,list_neighbors)
# dist_test = find_n_neighbors_multiclass(x_test,x_train,y_train,list_neighbors)

# print 'Distances found'

# </editor-fold>
