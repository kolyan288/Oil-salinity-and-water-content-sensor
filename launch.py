import os
import random
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

mode = ['Catboost', 'XGB']
mode = mode[1]

if mode == 'Catboost':
    try:
        model = CatBoostClassifier()
        model.load_model('Catboost_classifier.cbm') 
    except:
        print('Что-то пошло не так при загрузке модели. Возможно, вам следует обучить модель с нуля')

elif mode == 'XGB':
    try:
        model = XGBClassifier()
        model.load_model('XGBoost_classifier.json') 
    except:
        print('Что-то пошло не так при загрузке модели. Возможно, вам следует обучить модель с нуля')

feature_vector = [
    random.uniform(-27, 34),
    random.uniform(259, 344),
    random.uniform(184, 292),
    random.uniform(173, 280),
    random.uniform(1, 4),
    random.uniform(7, 13),
    random.uniform(1, 2),
    random.uniform(2, 5),
    random.uniform(10, 15),
    random.uniform(6, 10),
    #
    random.uniform(2, 3),
    random.uniform(0, 2),
    random.uniform(3, 7),
    random.uniform(0, 25),
    random.uniform(0, 4),
    random.uniform(40, 70),
    random.uniform(140, 190),
    random.uniform(70, 95),
    random.uniform(80, 105),
    random.uniform(1, 7),
    #
    random.uniform(4, 7),
    random.uniform(10, 60),
    random.uniform(80, 140),
    random.uniform(880, 2480),
    random.uniform(0, 1),
    random.uniform(50, 420),
    random.uniform(0.01, 0.18),
    random.uniform(17, 126),
    random.uniform(0.02, 0.04),
    random.uniform(10, 64)

]

feature_vector = list(map(lambda x: round(x, 2), feature_vector))
prediction = model.predict([feature_vector])

cls = prediction[0]
os.system(f"python3 tg_bot.py '{cls}' '{feature_vector}'")