# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics

import numpy as np
import pandas as pd

import csv

import scipy.stats as stats

def random_forest_prediction_opt(X_train, X_test, y_train, y_test, output_file):
    
    print("************** RANDOM FOREST OPT ************", end="\n\n")
    
    param_dist = {'n_estimators': range(10, 320),
                  'max_depth': range(2,50)}

    forest = RandomForestRegressor()
    rscv = RandomizedSearchCV(forest, param_dist, cv=10, n_iter=100, 
                              scoring='r2', n_jobs=4, verbose=1)
    
    rscv.fit(X_train, y_train)
    
    print("Best params :", rscv.best_params_)
    print("Best score :", rscv.best_score_, end="\n\n")
    
    y_pred = rscv.predict(X_test)
    
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    r2 = metrics.r2_score(y_test, y_pred)
    
    rho, pval = stats.spearmanr(y_pred, y_test)
    
    print('Mean Absolute Error:', mae)  
    print('Mean Squared Error:', mse)  
    print('Root Mean Squared Error:', rmse) 
    print("r^2 on test data :", r2)
    print("Spearman rank :", rho)
    print("P-value :", pval)
    
    with open(output_file, 'a', newline='') as results_file:
        writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(['Random forest opt', round(mae,4), round(mse,4), round(rmse,4), round(r2,4)])
   
    
    print("\n*********************************************", end="\n\n")

if __name__ == "__main__":
    
    dataset = pd.read_csv('stacking_features_complete.csv')
    
    X = dataset[['image_pred', 
                 'sentiment_pred', 
                 'keywords_pred', 
                 'comments count', 'type'
                 ]].values
    
    print("Data shape :", X.shape)
    
    # y = dataset['score'].values
    y = dataset['log_score'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    
    if len(X) != 6007:
        raise
    
    random_forest_prediction_opt(X_train, X_test, y_train, y_test, 'stacking_final_results.csv')