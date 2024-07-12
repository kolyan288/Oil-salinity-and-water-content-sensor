import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import sys
sys.path.append('/home/linux-home/Pyprojects/AWW_2024')
from Data_preprocessing import DataPipeline

plt.style.use('ggplot')

SEED = 123

# Заглушка для модели
# array = np.random.rand(2047, 51)
# np.savetxt('заглушка.csv', array, delimiter = ',')

DataPipeline(minmax=False)

X = pd.read_csv('X_features.csv')
y = pd.read_excel('PQv2_for_pandas.xlsx').loc[:, 'Total_salts']
y = pd.to_numeric(y, errors = 'coerce', downcast = 'float')
y.fillna(y.mean(), inplace=True)

def convert_values(x):
    return 0 if x <= 50 else 1

y_new = y.apply(convert_values)

X_train, X_test, y_train, y_test = train_test_split(X, y_new, test_size= 0.2, random_state = SEED)

def MLPipeline(task):

    if task == 'GridSearch':

        # param_grid_XGBC = {
        # 'n_estimators': [100, 200, 300],
        # 'max_depth': [3, 6, 12],
        # 'learning_rate': [0.01, 0.05, 0.1]
        # }

        param_grid_XGBC = {
        'n_estimators': [200], # [100, 200, 300]
        'max_depth': [12],         # [3, 6, 12]
        'learning_rate': [0.1], # [0.01, 0.05, 0.1]
        'grow_policy': ['depthwise', 'lossguide'],
        'booster': ['gbtree', 'gblinear', 'dart'],
        'tree_method': ['exact', 'approx', 'hist']
        }

        clf = XGBClassifier(random_state=SEED, objective = 'binary:logistic') 

        search = GridSearchCV(clf, param_grid_XGBC, scoring = 'f1', cv = 5, verbose = 2)

        result = search.fit(X_train, y_train)
        print(search.best_params_)

        clf2 = XGBClassifier(random_state=SEED, objective = 'binary:logistic', **search.best_params_) 
        clf2.fit(X_train, y_train)
        clf2.save_model('XGBoost_classifier.json')

        y_pred_train1 = clf2.predict(X_train)
        y_pred_test1= clf2.predict(X_test)

        y_pred_train2 = search.predict(X_train)
        y_pred_test2 = search.predict(X_test)

        print(all(y_pred_train1 == y_pred_train2))
        print(all(y_pred_test1 == y_pred_test2))

        # BEST PARAMS {'booster': 'gbtree', 'grow_policy': 'depthwise', 'learning_rate': 0.1, 'max_depth': 12, 'n_estimators': 200, 'tree_method': 'exact'}
        return y_pred_train2, y_pred_test2
    
    elif task == 'BestParams':

        clf = XGBClassifier(random_state=SEED, objective = 'binary:logistic', booster = 'gbtree',
                                                                              grow_policy = 'depthwise',
                                                                              learning_rate = 0.1,
                                                                              max_depth = 12,
                                                                              n_estimators = 200,
                                                                              tree_method = 'exact') 

        clf.fit(X_train, y_train)
        importances = list(zip(X.columns, clf.feature_importances_))
        importances.sort(key=lambda x: x[1], reverse = True)

        for i, j in importances:
            print(f"{i} -->> {j}")

        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        clf.save_model('XGBoost_classifier.json')

        return y_pred_train, y_pred_test

y_pred_train, y_pred_test = MLPipeline(task = 'BestParams')

print('-' * 80)
print(f"f1 на тренировочных данных: {f1_score(y_train, y_pred_train, average = 'micro')}")
print(f"f1 на тестовых данных: {f1_score(y_test, y_pred_test, average = 'micro')}")
print('-' * 80)

lol = pd.DataFrame()
lol.loc[:, 'facts'] = y_test
lol.loc[:, 'predictions'] = np.round(y_pred_test)
lol.to_csv('itog.csv')