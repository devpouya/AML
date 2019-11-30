import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import math




X2 = np.loadtxt("extracted_features_train.csv",delimiter=",")
X_test = np.loadtxt("extracted_features_test.csv",delimiter=",")
label_path = "y_train.csv"

labeldf = pd.read_csv(label_path)

labels = labeldf.values

labels = labels[:,1]

classifier2 = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=400, verbosity=1,
                                        objective='binary:logistic', booster='gbtree', tree_method='auto',
                                        n_jobs=-1, gpu_id=0, gamma=0, min_child_weight=3,
                                        max_delta_step=0, subsample=1, colsample_bytree=1,
                                        colsample_bylevel=1, colsample_bynode=1, reg_alpha=0,
                                        reg_lambda=1, base_score=0.5,
                                        missing=None)

bmc_scorer = make_scorer(f1_score, average = 'micro')

tuned_parameters2 = [{
                    'n_estimators':[400], 'max_depth':[5],
                    }]

grid2 = GridSearchCV(classifier2,  param_grid=tuned_parameters2,  scoring=bmc_scorer, n_jobs=-1,cv =5, verbose=3)


x_train, x_val, y_train, y_val = train_test_split(X2, labels, test_size=0.25, random_state=0, stratify=labels)


grid2.fit(x_train,y_train)
y_pred = grid2.predict(x_val)
F1 = f1_score(y_val, y_pred, average='micro')
print("VALIDATION: {}".format(F1))
print(grid2.cv_results_)
classifier2.fit(X2,labels)
final_pred = classifier2.predict(X_test)
sample = pd.read_csv("sample.csv", delimiter=',')
sample['y'] = final_pred
sample.to_csv("3000_xgboost.csv", index = False)
