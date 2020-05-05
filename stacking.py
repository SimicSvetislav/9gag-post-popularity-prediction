# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics

import numpy as np
import pandas as pd

import csv

def random_forest_prediction_opt(X, y, ids, output_file, on_what):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    print("************** RANDOM FOREST OPT ************", end="\n\n")
    
    param_dist = {'n_estimators': range(10, 320),
                  'max_depth': range(2,50)}

    forest = RandomForestRegressor()
    rscv = RandomizedSearchCV(forest, param_dist, cv=10, n_iter=50, 
                              scoring='r2', n_jobs=4, verbose=1)
    
    rscv.fit(X_train, y_train)
    
    print("Best params :", rscv.best_params_)
    print("Best score :", rscv.best_score_, end="\n\n")
    
    y_pred = rscv.predict(X_test)
    
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    r2 = metrics.r2_score(y_test, y_pred)
    
    print('Mean Absolute Error:', mae)  
    print('Mean Squared Error:', mse)  
    print('Root Mean Squared Error:', rmse) 
    print("r^2 on test data :", r2)
    
    with open('stacking_results.csv', 'a', newline='') as results_file:
        writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow([on_what, round(mae,4), round(mse,4), round(rmse,4), round(r2,4)])
   
    y_full_pred = rscv.predict(X)
    
    with open(output_file, 'a', newline='') as predictions_file:
        writer = csv.writer(predictions_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        if len(y) != len(y_full_pred):
            raise
            
        for i in range(len(y_full_pred)):
            writer.writerow([ids[i], y[i], y_full_pred[i]])
    
    print("\n*********************************************", end="\n\n")
 
    
def prediction_objects():
    
    dataset = pd.read_csv('features_complete_v2.csv')
    
    X = dataset[['person', 'people', 'cat', 'dog', 'other animal', 'poster', 
                 # 'clothing', 
                 'car', 'toy', 'tree', 'glasses', 
                 'building', 'electronic device', 'airplane', 'guitar',
                 # Pattern included
                 'pattern']].values
    
    ids = dataset['id'].values
    
    print("Data shape :", X.shape)
    
    y = dataset['score'].values
    
    if len(X) != 6007:
        raise
    
    output_file = 'stacking_pred_obj.csv'
    
    with open(output_file, 'w', newline='') as results_file:
        writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(['id', 'ground truth', 'prediction'])
    
    random_forest_prediction_opt(X, y, ids, output_file, 'images')
   
    
def prediction_sentiment():
    dataset = pd.read_csv('sentiment.csv')
    
    X = dataset[['very positive', 'positive', 
                 'neutral',
                 'negative', 'very negative'
                 ]].values
    
    ids = dataset['id'].values
    
    print("Data shape :", X.shape)
    
    y = dataset['score'].values
    
    if len(X) != 6007:
        raise
    
    output_file = 'stacking_pred_sentiment.csv'
    
    with open(output_file, 'w', newline='') as results_file:
        writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(['id', 'ground truth', 'prediction'])
    
    random_forest_prediction_opt(X, y, ids, output_file, 'sentiment')


def prediction_keywords():
    
    dataset = pd.read_csv('keywords.csv')
    
    X = dataset[[
        
        ]].values
    
    ids = dataset['id'].values
    
    print("Data shape :", X.shape)
    
    y = dataset['score'].values
    
    if len(X) != 6007:
        raise
    
    output_file = 'stacking_pred_keywords.csv'
    
    with open(output_file, 'w', newline='') as results_file:
        writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(['id', 'ground truth', 'prediction'])
    
    random_forest_prediction_opt(X, y, ids, output_file, 'keywords')


if __name__=="__main__":
    
    prediction_objects()
    
    # prediction_sentiment()
    
    # prediction_keywords()