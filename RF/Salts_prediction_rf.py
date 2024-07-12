import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRFRegressor
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = SEED)

def MLPipeline(task):
    
    if task == 'GridSearch':

        param_grid_XGBRFR = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 12],
                'learning_rate': [0.01, 0.05, 0.1],
                }

        regr = XGBRFRegressor(random_state=SEED) 

        search = GridSearchCV(regr, param_grid_XGBRFR, scoring = 'neg_mean_squared_error', cv = 5, verbose = 2)

        result = search.fit(X_train, y_train)
        print(search.best_params_)

        y_pred_train = search.predict(X_train)
        y_pred_test = search.predict(X_test)

        ### BEST PARAMS
        return y_pred_train, y_pred_test
    
    elif task == 'BestParams':

        regr = XGBRFRegressor(random_state=SEED) 

        regr.fit(X_train, y_train)

        y_pred_train = regr.predict(X_train)
        y_pred_test = regr.predict(X_test)

        return y_pred_train, y_pred_test

y_pred_train, y_pred_test = MLPipeline(task = 'GridSearch')

print(f"RMSE на тренировочных данных: {root_mean_squared_error(y_train, y_pred_train)}")
print(f"RMSE на тестовых данных: {root_mean_squared_error(y_test, y_pred_test)}")

print(f"MAPE на тестовых данных: {mean_absolute_percentage_error(y_test, y_pred_test)}")

lol = pd.DataFrame()
lol.loc[:, 'facts'] = y_test
lol.loc[:, 'predictions'] = np.round(y_pred_test)
lol.to_csv('itog.csv')