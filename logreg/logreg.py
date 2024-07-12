import math
import torch
import numpy as np
import pandas as pd
from torch import nn
import torch.nn as nn
from tqdm.auto import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MeanSquaredError, F1Score
from torchmetrics.classification import BinaryF1Score
from sklearn.linear_model import LogisticRegression
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

X = pd.read_csv('X_features.csv', header = None)
y = pd.read_excel('PQv2_for_pandas.xlsx').loc[:, 'Total_salts']
y = pd.to_numeric(y, errors = 'coerce', downcast = 'float')
y.fillna(y.mean(), inplace=True)

def convert_values(x):
    return 0 if x <= 50 else 1

y_new = y.apply(convert_values)

X_train, X_test, y_train, y_test = train_test_split(X, y_new, test_size= 0.2, random_state = SEED)

param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 12],
        'learning_rate': [0.01, 0.05, 0.1],
}

clf = LogisticRegression(random_state=SEED) 


clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)


print(f"f1 на тренировочных данных: {f1_score(y_train, y_pred_train, average = 'micro')}")
print(f"f1 на тестовых данных: {f1_score(y_test, y_pred_test, average = 'micro')}")

lol = pd.DataFrame()
lol.loc[:, 'facts'] = y_test
lol.loc[:, 'predictions'] = np.round(y_pred_test)
lol.to_csv('itog.csv')