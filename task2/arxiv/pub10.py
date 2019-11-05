import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

data_x = pd.read_csv("X_train.csv", delimiter=',')
data_y = pd.read_csv("y_train.csv", delimiter=',')

test_data = pd.read_csv("X_test.csv", delimiter=',')
test = np.array(test_data.iloc[:, 1:])

train = np.array(data_x.iloc[:, 1:])
label = np.array(data_y.iloc[:, 1:])
label= label.ravel()



# # scaling
# scaler = RobustScaler()
# train_scaled = scaler.fit_transform(train)
# test_scaled = scaler.fit_transform(test)

# np.savetxt('train_rob_sacled.csv', train_scaled, delimiter=',')
# np.savetxt('test_rob_scaled.csv', test_scaled, delimiter=',')

# train = np.loadtxt('train_rob_sacled.csv',  delimiter=',')
# test = np.loadtxt('test_rob_scaled.csv', delimiter=',')


#  feature selection
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import mutual_info_regression
# from sklearn.feature_selection import f_regression

# sel = SelectKBest(f_regression, k =550)
# train = sel.fit_transform(train, label)
# test =sel.transform(test)



# train, label = SMOTE().fit_resample(train, label)
# train, label = RandomUnderSampler().fit_resample(train, label)

def count(y_train):
    class_0 = 0
    class_1 = 0
    class_2 = 0
    for i in range(y_train.shape[0]):
        if y_train[i] == 0:
            class_0 = class_0 + 1
        elif y_train[i] == 1:
            class_1 = class_1 + 1
        elif y_train[i] == 2:
            class_2 = class_2 + 1
    # print("train #class 0 : {}       class 1: {}         class 2: {}".format(class_0, class_1, class_2))
print("shape of train:  {}".format(train.shape))
print("shape of label:  {}".format(label.shape))

#  feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import f_regression

# sel = SelectKBest(f_regression, k =700)
# train = sel.fit_transform(train, label)
# test =sel.transform(test)




for class_1_under in [900]:
    X_train, X_train_val, y_train, y_train_val = train_test_split(train, label,
                                                                  test_size=0.25, random_state=0, shuffle=True)

    class_0 = 0
    class_1 = 0
    class_2 = 0
    for i in range(y_train.shape[0]):
        if y_train[i] == 0:
            class_0 = class_0 + 1
        elif y_train[i] == 1:
            class_1 = class_1 + 1
        elif y_train[i] == 2:
            class_2 = class_2 + 1
    # print("train #class 0 : {}       class 1: {}         class 2: {}".format(class_0, class_1, class_2))

    print("class_1_under : {}".format(class_1_under))
    X_train, y_train = RandomUnderSampler(sampling_strategy={0: class_0, 1: class_1_under, 2: class_2}).fit_resample(X_train, y_train)
    X_train, y_train = RandomOverSampler(sampling_strategy='auto').fit_resample(X_train, y_train)

    print("after sampling")
    count(y_train)

    for ratio in [2]:
        print("ratio: {}".format(ratio))
        clf = SVC(C=10.0, kernel='rbf', degree=3, gamma='scale',
         coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200,
         class_weight={0:ratio,1:1.0,2:ratio}, verbose=False, max_iter=-1, decision_function_shape='ova',
          random_state=None)

        dt = DecisionTreeClassifier(criterion='gini', splitter='best',
        max_depth=None, min_samples_split=2, min_samples_leaf=1,
        min_weight_fraction_leaf=0.0, max_features=None, random_state=None,
        max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
         class_weight='balanced', presort=False)

        rf = RandomForestClassifier(n_estimators=100, criterion='gini',
        max_depth=None, min_samples_split=2, min_samples_leaf=1,
        min_weight_fraction_leaf=0.0, max_features='auto',
        max_leaf_nodes=None, min_impurity_decrease=0.0,
        min_impurity_split=None, bootstrap=True, oob_score=False,
        n_jobs=None, random_state=None, verbose=0, warm_start=False,
        class_weight={0: 6.0, 1: 1.0, 2: 6.0})

        knn = KNeighborsClassifier(n_neighbors=5, weights='uniform',
         algorithm='auto', leaf_size=30, p=2, metric='minkowski',
         metric_params=None, n_jobs=-1)

        xgboost =GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0)

        tuned_parameters = [{
            'svm__C':[0.5], 'svm__decision_function_shape':['ova'],
            # 'svm2__C':[0.1],
            # 'svm3__C':[0.4], 'svm__decision_function_shape':['ovo'],

                                                   # 'knn__n_neighbors':[1,2,3],
                                                   # 'dt__min_samples_leaf':[2],
                                                   # 'rf__min_samples_leaf':[1],
                                                # 'boost__learning_rate':[0.1]
                                                   }]

        estimators = VotingClassifier(estimators=[
            ('svm', clf),
            # ('svm2', clf),
            # ('svm3', clf),
            # ('knn', knn),
            # ('dt', dt),
            # ('rf',rf),
            # ('boost', xgboost)

            ])

        bmc_scorer = make_scorer(balanced_accuracy_score)
        grid = GridSearchCV(estimators ,  param_grid=tuned_parameters,  scoring=bmc_scorer , cv=5)
        grid.fit(X_train, y_train)

        best_est = grid.best_estimator_

        print("Best parameters set found on development set:")
        print()
        print(grid.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        print(grid.best_score_)

        pred_valid_y = best_est.predict(X_train_val)
        val_BMAC = balanced_accuracy_score(y_train_val, pred_valid_y)
        print("validation score: {}".format(val_BMAC))


        from sklearn.metrics import classification_report
        target_names = ['class 0', 'class 1', 'class 2']
        print(classification_report(y_train_val, pred_valid_y, target_names=target_names))

        y_pred_svr = best_est.predict(test)

        class_0 = 0
        class_1 = 0
        class_2 = 0
        for i in range(y_pred_svr.shape[0]):
            if y_pred_svr[i] == 0:
                class_0 = class_0 + 1
            elif y_pred_svr[i] == 1:
                class_1 = class_1 + 1
            elif y_pred_svr[i] == 2:
                class_2 = class_2 + 1

        print("#class 0 : {}       class 1: {}         class 2: {}".format(class_0, class_1, class_2))

        sample = pd.read_csv("sample.csv", delimiter=',')

        sample['y'] = y_pred_svr

        sample.to_csv("y_pred_900_2.csv", index = False)

        # print(y_pred_svr[:6])