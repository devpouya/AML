import numpy as np
import pandas as pd
import math
import sklearn
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel, DotProduct
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV
from sklearn.ensemble import IsolationForest, AdaBoostRegressor, BaggingRegressor, RandomForestRegressor, VotingRegressor, VotingClassifier
from sklearn.svm import SVR, OneClassSVM
from sklearn.linear_model import Lasso
from sklearn.covariance import EllipticEnvelope
from sklearn.kernel_ridge import KernelRidge



train_path = "X_train.csv"
test_path = "X_test.csv"
label_path = "y_train.csv"

df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
labeldf = pd.read_csv(label_path)


labels = labeldf.values
data = df.values
test_data = test_df.values



means = np.nanmean(data,axis=0).reshape(1,833)
meanstest = np.nanmean(data,axis=0).reshape(1,833)
X = np.nan_to_num(data,nan=means)
X_test = np.nan_to_num(test_data,nan=meanstest)

# drop columns where all values are the same

X = X[:, ~(X == X[0,:]).all(0)]
X_test = X_test[:, ~(X_test == X_test[0,:]).all(0)]

y = labels

scaler = preprocessing.RobustScaler()

X_test = scaler.fit_transform(X_test)
X = scaler.fit_transform(X)
y = y[:,1]



eps = 0.03
feature_selector = SelectKBest(f_regression,k=170).fit(X,y)
X = feature_selector.transform(X)
X_test = feature_selector.transform(X_test)




anamoly_detector = IsolationForest()
anamoly_voting = VotingClassifier([("f1",anamoly_detector),("f2",anamoly_detector),("f3",anamoly_detector)
                                    ("f4",anamoly_detector), ("f5",anamoly_detector), ("f6",anamoly_detector)])



X_anamoly = anamoly_detector.fit_predict(X)
X = X[X_anamoly==1]
(num_samples,_) = X.shape

y = y[np.argwhere(X_anamoly==1).reshape(num_samples,)]




X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=7)


# GP Linear + RBF

svr = SVR(kernel="rbf",C=100,gamma="scale",epsilon=0.01)
svr25 = SVR(kernel="rbf",C=25,gamma="scale",epsilon=0.01)
svr40 = SVR(kernel="rbf",C=40,gamma="scale",epsilon=0.01)
random_forest = RandomForestRegressor()
kernel = DotProduct() + RBF()
gp = GaussianProcessRegressor(kernel=kernel)

from sklearn.metrics import make_scorer

kf = KFold(n_splits=10,random_state=7,shuffle=True)
r2_scorer = make_scorer(r2_score)
svr_params = [{"C":[20,25,45,100]}]
random_forest_params = [{"n_estimators":[10,20,25,45]}]

#grid_svr = GridSearchCV(svr,param_grid=svr_params,scoring=r2_scorer,cv=10)
#grid_svr.fit(X_train,y_train)

#grid_randomforest = GridSearchCV(random_forest,param_grid=random_forest_params,scoring=r2_scorer,cv=10)
#grid_randomforest.fit(X_train,y_train)


#clf_svr = grid_svr.best_estimator_
#clf_randomforest = grid_randomforest.best_estimator_


model_averaging = VotingRegressor([("svr1",svr),("svr2",svr25),("svr3",svr40)
                                 ,("svr4",svr),("svr5",svr25),("svr6",svr40),("random_forest",random_forest),("gaussian_process",gp)])
best_group = VotingRegressor([("svr1",svr),("svr2",svr25),("svr3",svr40)
                                 ,("svr4",svr),("svr5",svr25),("svr6",svr40),("random_forest",random_forest),("gaussian_process",gp)])
best_score = -1000000

for train_ind, test_ind in kf.split(X_train):

    X_train_v , X_test_v = X_train[train_ind], X_train[test_ind]
    y_train_v, y_test_v = y_train[train_ind], y_train[test_ind]

    fitted = model_averaging.fit(X_train_v,y_train_v)
    val_pred = fitted.predict(X_test_v)
    print("Score is: ")
    score = r2_score(y_test_v,val_pred)
    print(score)
    print()
    if score > best_score:
        best_group = fitted
        best_score = score
        print("NEW BEST")


pred = best_group.predict(X_val)
validation_score = r2_score(y_val,pred)
print("VALIDATION SCORE:")
print(validation_score)
print()
"""
print("SVR Score Cross Val")
print()
print(grid_svr.best_score_)
print("RandomForestRegressor Score Cross Val")
print()
print(grid_randomforest.best_score_)

"""






#gp.fit(X,y.reshape(num_samples,))


y_final = best_group.predict(X_test)
y_final = y_final.astype(int)


path = "predictions.csv"

pd.DataFrame(y_final).to_csv(path)
