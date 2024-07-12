import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def DataPipeline(minmax = True):
    
    df = pd.read_excel('PQv2_for_pandas.xlsx')

    for i in df.columns[1:]:
        df[i] = pd.to_numeric(df.loc[:, i], errors = 'coerce', downcast = 'float')
        df[i] = df[i].fillna(df[i].mean())

    df.drop(['Data', 'Time', 'function', 'Total_%_water', 'Total_salts'], axis = 1, inplace = True)

    features_for_drop = ['PG_1_И', 
                         'PG_2_И', 
                         'Cons_stab_oil', 
                         'Cons_oil_into_furnace', 
                         'Pres_sump',
                         'Temp_hot_jet',
                         'Temp_raw_oil',
                         'Temp_ready_oil',
                         'Pres_Б',
                         'Temp_column_power',
                         'Temp_column_bottom',
                         'Temp_raw_oil_from_intake',
                         'Analysis_after_PG_2_%_water',
                         'Analysis_after_PG_2_salts',
                         'PG_2_А2',
                         'Cons_after',
                         ]

    df.drop(features_for_drop, axis = 1, inplace = True)
    
    for col in df.columns[1:]:
        
        # Шаг 1: Определение выбросов
        Q1 = df[col].quantile(0.15)
        Q3 = df[col].quantile(0.85)
        IQR = Q3 - Q1

        # Определяем границы для выбросов
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Шаг 2: Замена выбросов на np.nan
        df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = np.nan

        # Шаг 3: Вычисление среднего значения, исключая np.nan
        mean_value = df[col].median()

        # Шаг 4: Замена np.nan на среднее значение
        df[col].fillna(mean_value, inplace=True)

        df[col]

    if minmax:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(df)
        X = pd.DataFrame(X, columns = df.columns)
    else:
        X = df
    

    #np.savetxt('X_features.csv', X, delimiter = ',')
    X.to_csv('X_features.csv', index=None)

