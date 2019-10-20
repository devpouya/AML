#  Load the Pandas libraries with alias 'pd' 
import pandas as pd 
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.metrics import r2_score
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
data_x = pd.read_csv("X_train.csv", delimiter=',') 
data_y = pd.read_csv("y_train.csv", delimiter=',')

test_data = pd.read_csv("X_test.csv", delimiter=',')
test_x = np.array(test_data.iloc[:, 1:])
#  drop the index column , getting the real features
train = np.array(data_x.iloc[:, 1:])
label = np.array(data_y.iloc[:, 1:])

print("shape of train:  {}".format(train.shape))
print("shape of label:  {}".format(label.shape))

# train_scaled = MinMaxScaler().fit_transform(train)
# test_scaled = MinMaxScaler().fit_transform(test_x)


train_scaled = train
test_scaled = test_x

# actually the following is filled with mean
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
train_scaled_filled = imp.fit_transform(train_scaled)                  
test_scaled_filled = imp.fit_transform(test_scaled)

######### outlier detection: only training data##############
clf = IsolationForest( behaviour = "new", max_samples=50, random_state = 1, contamination= "auto")
pred_outlier = clf.fit_predict(train_scaled_filled)

num_outlier = 0

for i in range(pred_outlier.shape[0]):
    if pred_outlier[i] == -1:
        num_outlier = num_outlier + 1

print(num_outlier)

to_del_idx  = np.empty(num_outlier)
pointer = 0
for i in range(pred_outlier.shape[0]):
    if pred_outlier[i] == -1:
        to_del_idx[pointer] = i
        pointer = pointer + 1

train_del = np.delete(train_scaled_filled, to_del_idx, axis = 0)
label_del = np.delete(label, to_del_idx, axis = 0)
################################################
# train_del = train_scaled_filled 
# label_del = label


###############feature selection####TRAIN########
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression

sel = SelectKBest(mutual_info_regression, k =256)
train_selected = sel.fit_transform(train_del, label_del)

# train_selected = sel.tranform(train_del)




###############feature selection####TEST########
test_selected = sel.transform(test_scaled_filled)

################normalizer######################
train_normalized = train_selected
test_normalized = test_selected
# from sklearn.preprocessing import Normalizer

# train_normalized = Normalizer().fit_transform(train_selected)
# test_normalized =  Normalizer().fit_transform(test_selected)

# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# train_normalized =  scaler.fit_transform(train_selected)
# test_normalized =  scaler.fit_transform(test_selected)

################################################


################model######################
X_train, X_train_val, y_train, y_train_val = train_test_split(train_normalized, label_del, test_size = 0.2, random_state = 0, shuffle = True)


# clf = svm.SVR(C=1, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
#     gamma='auto_deprecated', kernel='linear', max_iter=-1, shrinking=True,
#     tol=0.001, verbose=True)

from sklearn import linear_model
# from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# clf =svm.SVR(C=15.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
#     gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
#     tol=0.001, verbose=False)

from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=500,
           max_features=200,  max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=150, n_jobs=4,
           oob_score=False, random_state=0, verbose=0, warm_start=False)

# easy_validate_y = clf.predict(X_test_val)

# print("easy_validate_score {}".format(r2_score(clf.predict(X_test_val), y_test_val)))

# clf.fit(train_normalized, label_del)

# clf.fit(X_train, y_train)

from sklearn.metrics import make_scorer
r2_scorer = make_scorer(r2_score)
scores = cross_val_score(clf, train_normalized, label_del, cv=5)

print("cv score = {}".format(scores))
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf.fit(train_normalized, label_del)

train_pred = clf.predict(train_normalized)
score_on_train = r2_score(train_pred, label_del)
print("score on entire train : {}".format(score_on_train))

validate_pred = clf.predict(X_train_val)
score_on_val = r2_score(validate_pred, y_train_val)
print("score on validation set : {}".format(score_on_val))

#################preprocess  test data#####################

y_pred_svr = clf.predict(test_normalized)


sample = pd.read_csv("sample.csv",delimiter=',')

sample['y'] = y_pred_svr

sample.to_csv("y_pred_randForest10.csv", index = False)