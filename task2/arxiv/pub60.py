import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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


from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from xgboost import XGBClassifier
# from imbxgboost import imbx
from imxgboost.imbalance_xgb import imbalance_xgboost as imb_xgb
import matplotlib.pyplot as plt

data_x = pd.read_csv("X_train.csv", delimiter=',')
data_y = pd.read_csv("y_train.csv", delimiter=',')

test_data = pd.read_csv("X_test.csv", delimiter=',')
test = np.array(test_data.iloc[:, 1:])

train = np.array(data_x.iloc[:, 1:])
label = np.array(data_y.iloc[:, 1:])
label= label.ravel()
# label = label.astype(int)/

# # scaling
# scaler = RobustScaler()
# train_scaled = scaler.fit_transform(train)
# test_scaled = scaler.fit_transform(test)
# scaler = StandardScaler()
# train = scaler.fit_transform(train)
# test = scaler.fit_transform(test)

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
    print("train #class 0 : {}       class 1: {}         class 2: {}".format(class_0, class_1, class_2))
print("shape of train:  {}".format(train.shape))
print("shape of label:  {}".format(label.shape))

temp_k = 0
temp_sample = 0
temp_ratio = 0
best_valid_score = 0
#  feature selection

train_ori  = train
label_ori = label
test_ori = test
for k in [
    # 100,200,300,
    # 400,500,600,
    # 700,800,900,
    1000
]:
    print("k = {}".format(k))
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.feature_selection import f_classif
    from sklearn.feature_selection import f_regression

    # sel = SelectKBest(f_regression, k =k)
    # train = sel.fit_transform(train_ori, label_ori)
    # test =sel.transform(test_ori)

    from sklearn.feature_selection import VarianceThreshold

    # sel = VarianceThreshold(threshold=(0.3))
    # train = sel.fit_transform(train)
    # test = sel.transform(test)

    print("after feature selection")
    print(train.shape)

    for class_1_under in [900
        # 300,400,500,600,700,800,900,1000,1100,1200,
        ]:
        X_train, X_train_val, y_train, y_train_val = train_test_split(train, label,
                                                                      test_size=0.25, random_state=0, shuffle=True, stratify=label)

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
        print("train #class 0 : {}       class 1: {}         class 2: {}".format(class_0, class_1, class_2))

        print("class_1_under : {}".format(class_1_under))
        # X_train, y_train = RandomUnderSampler(sampling_strategy={0: class_0, 1: class_1_under, 2: class_2}).fit_resample(X_train, y_train)

        # X_train, y_train = RandomOverSampler(sampling_strategy='auto').fit_resample(X_train, y_train)
        #
        # X_train, y_train = RandomUnderSampler(
        #     sampling_strategy={0: class_0, 1: class_1_under, 2: class_2}).fit_resample(X_train, y_train)
        #
        # X_train, y_train = RandomOverSampler(sampling_strategy='auto').fit_resample(X_train, y_train)

        print("after sampling")
        count(y_train)

        for ratio in [6.0]:
            print("ratio: {}".format(ratio))
            clf = SVC(C=0.5, kernel='rbf', degree=3,
                      # gamma='scale',
                      gamma=1 / (1000 * X_train.std()),
             coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=500,
             class_weight=
                      # 'balanced',
                      {0: ratio, 1: 0.95, 2: ratio},
                      verbose=False, max_iter=-1, decision_function_shape='ova',
              random_state=None)

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
            xgboost = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=66, verbosity=1,
                                        objective='multi:softmax', booster='gbtree', tree_method='auto',
                                        n_jobs=-1, gpu_id=0, gamma=0, min_child_weight=1,
                                        max_delta_step=0, subsample=1, colsample_bytree=1,
                                        colsample_bylevel=1, colsample_bynode=1, reg_alpha=0,
                                        reg_lambda=1, scale_pos_weight=6, base_score=0.5,
                                        random_state=0, missing=None)

            imb_xgboost = imb_xgb(booster='gbtree', eta=0.3, eval_metric='logloss',
                  focal_gamma=None, imbalance_alpha=1, max_depth=10,
                  num_round=10, objective_func='multi:softmax',
                  silent_mode=True, special_objective='weighted')

            mn = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=0.0005, fit_intercept=True,
            intercept_scaling=1, class_weight={0: ratio, 1:1.0, 2: ratio}, random_state=None,
             max_iter=500, multi_class='ovr', solver = 'lbfgs',
                                    verbose=0, warm_start=False, n_jobs=-1, l1_ratio=None)

            bag = BaggingClassifier(base_estimator=clf, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True,
                              bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=-1,
                              random_state=None, verbose=0)

            nn = MLPClassifier(hidden_layer_sizes=(256,128), alpha=1, max_iter=1000)

            for n_estimators in [10]:
                print("n_neighbors {}".format(n_estimators))
                for c in [
                    1
                          ]:
                    for gamma in [0.000001]:
                        # print("c = {}".format(c))
                        # print("gamma = {}".format(gamma))
                        tuned_parameters = [{
                            'svm__C': [0.45],
                            # 'svm2__C':[0.6],
                            'svm3__C':[0.5], 'svm__decision_function_shape':['ovo'],
                            #
                            # 'knn__n_neighbors':[n_neighbor],
                            'dt__max_leaf_nodes':[5],
                            'rf__n_estimators':[20],
                             'boost__n_estimators':[20], 'boost__max_depth':[5],
                            # 'imb_boost__num_round':[10]
                            'mn__C':[0.0005],
                            'mn2__C': [0.0007], 'mn2__multi_class':['multinomial'],
                            'mn3__C': [0.0006], 'mn3__multi_class':['multinomial'],
                            # 'bag__n_estimators':[10]
                            }]



                        estimators = VotingClassifier(estimators=[
                            ('svm', clf),
                            ('svm2', clf),
                            ('svm3', clf),
                            # ('knn', knn),
                            ('dt', dt),
                            ('rf', rf),
                            ('boost', xgboost),
                            # ('imb_boost', imb_xgboost),
                            #
                            ('mn', mn),
                            ('mn2', mn),
                            ('mn3', mn),

                            ('nn', nn)
                            # ('bag', bag)
                            ] )

                        bmc_scorer = make_scorer(balanced_accuracy_score)
                        grid = GridSearchCV(estimators,  param_grid=tuned_parameters,  scoring=bmc_scorer, n_jobs=-1,cv = 5 )

                        print(y_train.shape)
                        grid.fit(X_train, y_train)

                        target_names = ['class 0', 'class 1', 'class 2']

                        best_est = grid.best_estimator_

                        # best_est = estimators.fit(X_train, y_train)
                        # best_est = best_est.fit(X_train, y_train)
                        y_train_predict = best_est.predict(X_train)
                        print(classification_report(y_train, y_train_predict, target_names=target_names))
                        count(y_train_predict)
                        train_BMAC = balanced_accuracy_score(y_train, y_train_predict)
                        print("train score: {}".format(train_BMAC))

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

                        if val_BMAC > best_valid_score:
                            best_valid_score = val_BMAC
                            temp_k = k
                            temp_sample = class_1_under
                            temp_ratio = ratio


                        target_names = ['class 0', 'class 1', 'class 2']
                        print(classification_report(y_train_val, pred_valid_y, target_names=target_names))
#
                        class_0 = 0
                        class_1 = 0
                        class_2 = 0
                        for i in range(pred_valid_y.shape[0]):
                            if pred_valid_y[i] == 0:
                                class_0 = class_0 + 1
                            elif pred_valid_y[i] == 1:
                                class_1 = class_1 + 1
                            elif pred_valid_y[i] == 2:
                                class_2 = class_2 + 1

                        print("validation set prediction #class 0 : {}       class 1: {}         class 2: {}".format(class_0, class_1, class_2))





                        y_pred_svr = best_est.predict(test)

                        # class_0 = 0
                        # class_1 = 0
                        # class_2 = 0
                        # for i in range(y_pred_svr.shape[0]):
                        #     if y_pred_svr[i] == 0:
                        #         class_0 = class_0 + 1
                        #     elif y_pred_svr[i] == 1:
                        #         class_1 = class_1 + 1
                        #     elif y_pred_svr[i] == 2:
                        #         class_2 = class_2 + 1
                        #
                        # print("#class 0 : {}       class 1: {}         class 2: {}".format(class_0, class_1, class_2))

                        sample = pd.read_csv("sample.csv", delimiter=',')

                        sample['y'] = y_pred_svr

                        file_name = "ous_y_pred_"+str(class_1_under)+"_"+str(ratio)+"_"+"svm_"+str(c)+".csv"

                        sample.to_csv(file_name, index = False)

                        print(y_pred_svr[:6])



print("print best score, k, sample. ratio")
print(best_valid_score)
print(temp_k)
print(temp_sample)
print(temp_ratio)


print("stored in " + file_name)
