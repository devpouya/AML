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
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.ensemble import IsolationForest, AdaBoostRegressor, BaggingRegressor
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

X_anamoly = anamoly_detector.fit_predict(X)
X = X[X_anamoly==1]
(num_samples,_) = X.shape

y = y[np.argwhere(X_anamoly==1).reshape(num_samples,)]




X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=7)




regressor = SVR(kernel="rbf",C=35,gamma="scale",epsilon=0.01)
regressor.fit(X_train,y_train)
#regr = AdaBoostRegressor()
#regr.fit(X_train,y_train)

#scores = cross_val_score(regressor2,X_train,y_train,cv=10)
#pred2 = cross_val_predict(regressor2,X_val,y_val,cv=10)

#kernel_scores = cross_val_score(gp,X_train,y_train,cv=10)
#pred_ker = cross_val_predict(gp,X_val,y_val,cv=10)


#pred = regressor.predict(X_val)
pred2 = regressor.predict(X_val)
score2 = r2_score(y_val,pred2)
#score_kernel = r2_score(y_val,pred_ker)

print("SVR")
print(score2)


#gp.fit(X,y.reshape(num_samples,))


y_final = regressor.predict(X_test)
y_final = y_final.astype(int)


path = "predictions.csv"

pd.DataFrame(y_final).to_csv(path)
