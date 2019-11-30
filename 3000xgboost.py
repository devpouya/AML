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
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from imblearn.over_sampling import SVMSMOTE
from sklearn.model_selection import KFold,StratifiedKFold

import matplotlib.pyplot as plt
def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

X2 = np.loadtxt("extracted_features_train.csv",delimiter=",")
X_test = np.loadtxt("extracted_features_test.csv",delimiter=",")
label_path = "y_train.csv"

labeldf = pd.read_csv(label_path)

labels = labeldf.values

labels = labels[:,1]

classifier2 = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=2000, verbosity=1,
                                        objective='binary:logistic', booster='gbtree', tree_method='auto',
                                        n_jobs=-1, gpu_id=0, gamma=0, min_child_weight=3,
                                        max_delta_step=0, subsample=1, colsample_bytree=1,
                                        colsample_bylevel=1, colsample_bynode=1, reg_alpha=0,
                                        reg_lambda=1, base_score=0.5,
                                        missing=None)

bmc_scorer = make_scorer(f1_score, average = 'micro')

tuned_parameters2 = [{
                    'min_child_weight':[1,2,3],
                    }]

grid2 = GridSearchCV(classifier2,  param_grid=tuned_parameters2,  scoring=bmc_scorer, n_jobs=-1,cv =5, verbose=3)


x_train, x_val, y_train, y_val = train_test_split(X2, labels, test_size=0.25, random_state=0, stratify=labels)
smote = SVMSMOTE(sampling_strategy={0:2272,1:334,2:1105,3:334},random_state=42)
X_bal, y_bal = smote.fit_resample(x_train,y_train)

grid2.fit(X_bal,y_bal)
y_pred = grid2.predict(x_val)
F1 = f1_score(y_val, y_pred, average='micro')
print("VALIDATION: {}".format(F1))
print(grid2.cv_results_)

smote = SVMSMOTE(sampling_strategy={0:3030,1:443,2:1474,3:443},random_state=42)
X_bal, y_bal = smote.fit_resample(X2,labels)

grid2.best_estimator_.fit(X2,labels)
final_pred = gri2.best_estimator_.predict(X_test)
sample = pd.read_csv("sample.csv", delimiter=',')
sample['y'] = final_pred
sample.to_csv("3000_finalPred.csv", index = False)



# y_val_pred = np.argmax(grid.predict(x_val), axis = 1)
# y_train_pred = np.argmax(grid.predict(x_train), axis = 1)
#y_val_pred = grid.predict(x_val)
#y_train_pred = grid.predict(x_train)
y_final_val = y_pred
y_final_val = y_final_val.astype(int)
y_val = y_val.astype(int)
class_names = np.array([0,1])
#class_names = int(class_names)
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_val, y_final_val,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_val, y_final_val, normalize=True,
                      title='Normalized confusion matrix')



plt.show()
