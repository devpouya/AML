import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#from xgboost import XGBClassifier
from scipy.stats import mode


train_path = "X_train.csv"
test_path = "X_test.csv"
label_path = "y_train.csv"

df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
labeldf = pd.read_csv(label_path)


labels = labeldf.values
data = df.values
test_data = test_df.values

data = data[:,1:]
test_data = test_data[:,1:]
labels = labels[:,1]


(num_samples,num_features) = data.shape
(num_test,_) = test_data.shape


# count the class instances
unique, counts = np.unique(labels,return_counts=True)
class_counts = dict(zip(unique,counts))


scaler = StandardScaler()

scaler.fit(data)
X = scaler.transform(data)
X_test = scaler.transform(test_data)
y = labels





"""
anamoly_detector = IsolationForest()

X_anamoly = anamoly_detector.fit_predict(X)
X = X[X_anamoly==1]
(num_samples,_) = X.shape

y = y[np.argwhere(X_anamoly==1).reshape(num_samples,)]
"""

y_class_0 = y[y==0]
y_class_1 = y[y==1]
y_class_2 = y[y==2]

X_0 = X[np.argwhere(y==0)]
X_1 = X[np.argwhere(y==1)]
X_2 = X[np.argwhere(y==2)]

X_1s = np.split(X_1,6,axis=0)
y_1s = np.split(y_class_1,6)

ys = np.empty((6,1800,1))
Xs = np.empty((6,1800,num_features))

for i in range(6):
    ys[i] = np.hstack((y_1s[i],y_class_0,y_class_2)).reshape(1800,1)
    Xs[i] = np.hstack((X_1s[i],X_0,X_2)).reshape(1800,num_features)




support_machine = SVC(C=0.25, class_weight={0:1,1:1/6,2:1},gamma="auto")
support_machine1 = SVC(C=0.3, class_weight={0:6,1:1,2:6},gamma="auto")
support_machine2 = SVC(C=0.45,class_weight={0:1,1:1/6,2:1},gamma="auto")
support_machine3 = SVC(C=0.5,class_weight={0:1,1:1/6,2:1},gamma="auto")
support_machine4 = SVC(C=4,class_weight={0:1,1:1/6,2:1},gamma="auto")
support_machine5 = SVC(C=5,class_weight={0:1,1:1/6,2:1},gamma="auto")

ratio = 6




dt = DecisionTreeClassifier(criterion='gini', splitter='best',
                            max_depth=3, min_samples_split=2, min_samples_leaf=5,
                            min_weight_fraction_leaf=0.15, max_features=None, random_state=None,
                            max_leaf_nodes=5, min_impurity_decrease=0.0, min_impurity_split=None,
                            class_weight={0: ratio, 1: 1, 2: ratio}, presort=False)

rf = RandomForestClassifier(n_estimators=20, criterion='gini',
                                        max_depth=3, min_samples_split=2, min_samples_leaf=5,
                                        min_weight_fraction_leaf=0.15, max_features='auto',
                                        max_leaf_nodes=5, min_impurity_decrease=0.0,
                                        min_impurity_split=None, bootstrap=True, oob_score=False,
                                        n_jobs=-1, random_state=None, verbose=0, warm_start=False,
                                        class_weight={0: ratio, 1: 1.0, 2: ratio})

knn = KNeighborsClassifier(n_neighbors=1, weights='distance',
                           algorithm='auto', leaf_size=30, p=2, metric='minkowski',
                           metric_params=None, n_jobs=-1)

# xgboost =GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0)
"""
xgboost = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=66, verbosity=1,
                        objective='multi:softmax', booster='gbtree', tree_method='auto',
                        n_jobs=-1, gpu_id=0, gamma=0, min_child_weight=1,
                        max_delta_step=0, subsample=1, colsample_bytree=1,
                        colsample_bylevel=1, colsample_bynode=1, reg_alpha=0,
                        reg_lambda=1, scale_pos_weight=1, base_score=0.5,
                        random_state=0, missing=None)
"""
"""
imb_xgboost = imb_xgb(booster='gbtree', eta=0.3, eval_metric='logloss',
                  focal_gamma=None, imbalance_alpha=1, max_depth=10,
                  num_round=10, objective_func='multi:softmax',
                  silent_mode=True, special_objective='weighted')
"""
mn = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=0.0005, fit_intercept=True,
                        intercept_scaling=1, class_weight={0: ratio, 1:1.0, 2: ratio}, random_state=None,
                        max_iter=500, multi_class='ovr', solver = 'lbfgs',
                        verbose=0, warm_start=False, n_jobs=-1)



nn = MLPClassifier(alpha=1, max_iter=1000)



best_score = -1000000


scores = []
y_finals = np.empty((6,num_test,1))

for i in range(6):
    X_train, X_val,y_train, y_val = train_test_split(Xs[i],ys[i],test_size=0.2,random_state=7)

    clf = SVC(C=0.5, kernel='rbf', degree=3,
            # gamma='scale',
            gamma=1 / (1000 * X_train.std()),
            coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=500,
            class_weight=
            # 'balanced',
            {0: ratio, 1: 0.95, 2: ratio},
            verbose=False, max_iter=-1, decision_function_shape='ova',
            random_state=None)
    tuned_parameters = [{
                                'svm__C': [0.45],
                                # 'svm2__C':[0.6],
                                'svm3__C':[0.5], 'svm__decision_function_shape':['ovo'],
                                #
                                # 'knn__n_neighbors':[n_neighbor],
                                # 'dt__max_leaf_nodes':[5],
                                # 'rf__n_estimators':[20],
                                #'boost__n_estimators':[10], 'boost__max_depth':[3],
                                # # 'imb_boost__num_round':[10]
                                'mn__C':[0.0005],
                                'mn2__C': [0.0007], 'mn2__multi_class':['multinomial'],
                                'mn3__C': [0.0006], 'mn3__multi_class':['multinomial'],
                                # 'bag__n_estimators':[10]
                                }]

    estimators = VotingClassifier(estimators=[('svm', clf),
                                                      ('svm2', clf),
                                                      ('svm3', clf),
                                                      # ('knn', knn),
                                                      ('dt', dt),
                                                      ('rf', rf),
                                                      #('boost', xgboost),
                                                      ('mn', mn),
                                                      ('mn2', mn),
                                                      ('mn3', mn),
                                                      # ('bag', bag)
                                                      ])

    bmc_scorer = make_scorer(balanced_accuracy_score)
    grid = GridSearchCV(estimators,  param_grid=tuned_parameters,  scoring=bmc_scorer, n_jobs=-1,cv = 10 )
    print("hheh")
    grid.fit(X_train,y_train)
    pred = grid.best_estimator_.predict(X_val)
    validation_score = balanced_accuracy_score(y_val,pred)
    print("Validation score:")
    print(validation_score)
    scores.append(validation_score)
    y_final = grid.predict(X_test)
    y_final = y_final.astype(int)

    y_finals[i] = y_final.reshape(num_test,1)

y_final = mode(y_finals,axis=0)
#y_final = y_final.astype(int)

"""
for train_ind, test_ind in kf.split(X_train):
    X_train_v , X_test_v = X_train[train_ind], X_train[test_ind]
    y_train_v, y_test_v = y_train[train_ind], y_train[test_ind]
    fitted = voting_class.fit(X_train_v,y_train_v)
    val_pred = fitted.predict(X_test_v)
    print("Score is: ")
    score = balanced_accuracy_score(y_test_v,val_pred)
    print(score)
    print()
    scores.append(score)
    if score > best_score:
        best_votingClass = fitted
        best_score = score
        print("NEW BEST")
"""


print("Total Variance is")
print(np.var(scores))

"""
y_final = best_votingClass.predict(X_test)
y_final = y_final.astype(int)
"""

path = "predictions.csv"

pd.DataFrame(y_final).to_csv(path)



print("Bonn17")
