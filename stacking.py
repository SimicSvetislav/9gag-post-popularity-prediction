# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics

import numpy as np
import pandas as pd

import csv


from math import log

import stacking_combine_features

from encode_words import WORDS_LIST_2
import stacking_combine_features as scf
import stacking_final_prediction as sfp

import scipy.stats as stats

def random_forest_prediction_opt(X, y, ids, output_file, on_what):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
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
    
    dataset = pd.read_csv('features_complete_v3.csv')
    
    X = dataset[['person', 'people', 'cat', 'dog', 'other animal', 'poster', 
                 # 'clothing', 
                 'car', 'toy', 'tree', 'glasses', 
                 'building', 'electronic device', 'airplane', 'guitar',
                 # Pattern included
                 'pattern', 'image_text']].values
    
    ids = dataset['id'].values
    
    print("Data shape :", X.shape)
    
    y = dataset['log_score'].values
    
    if len(X) != 6007 and len(X) != 2905:
        raise
    
    output_file = 'stacking_pred_obj.csv'
    
    with open(output_file, 'w', newline='') as results_file:
        writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(['id', 'ground truth', 'prediction'])
    
    random_forest_prediction_opt(X, y, ids, output_file, 'images')
   
    
def prepare_scores():
    
    ds_v2 = pd.read_csv('features_complete_v3.csv')
    
    scores_dict = {}
    
    for index, row in ds_v2.iterrows():
        scores_dict[row['id']] = row['score']
    
    ds_sent = pd.read_csv('percentage_sentiment_average.csv')
    
    
    not_found = 0
    
    with open('percentage_sentiment_average_with_scores.csv', 'w', newline='') as out_file:
    
        writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(['id', 'very_positive', 'positive', 'neutral',
                         'negative', 'very_negative', 'score'])
        
        for index, row in ds_sent.iterrows():
            try:
                writer.writerow([row['id'], row['very_positive'], 
                                 row['positive'], row['neutral'], 
                                 row['negative'],row['very_negative'],
                                 scores_dict[row['id']]])
            except:
                not_found += 1
    
    print("Not found :", not_found)

def prediction_sentiment():
    
    prepare_scores()
    
    dataset = pd.read_csv('percentage_sentiment_average_with_scores.csv')
    
    X = dataset[['very_positive', 'positive', 
                 'neutral',
                 'negative', 'very_negative'
                 ]].values
    
    ids = dataset['id'].values
    
    print("Data shape :", X.shape)
    
    y = dataset['score'].values
    
    y = list(map(lambda score: log(score+1), y))
    
    if len(X) != 6007 and len(X) != 2905:
        raise
    
    output_file = 'stacking_pred_sentiment.csv'
    
    with open(output_file, 'w', newline='') as results_file:
        writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(['id', 'ground truth', 'prediction'])
    
    random_forest_prediction_opt(X, y, ids, output_file, 'sentiment')

def prediction_keywords():
    
    dataset = pd.read_csv('features_complete_v3.csv')
    
    X = dataset[WORDS_LIST_2].values
    
    # print(X)
    
    ids = dataset['id'].values
    
    print("Data shape :", X.shape)
    
    y = dataset['log_score'].values
    
    if len(X) != 6007 and len(X) != 2905:
        raise
    
    output_file = 'stacking_pred_keywords.csv'
    
    with open(output_file, 'w', newline='') as results_file:
        writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(['id', 'ground truth', 'prediction'])
    
    random_forest_prediction_opt(X, y, ids, output_file, 'keywords')


def prediction_sentiment_post_average():
    
    prepare_scores()
    
    dataset = pd.read_csv('features_complete_v3.csv')
    
    X = dataset[['comments']].values
    
    ids = dataset['id'].values
    
    print("Data shape :", X.shape)
    
    y = dataset['score'].values
    
    y = list(map(lambda score: log(score+1), y))
    
    print("X len", len(X) )
    
    if len(X) != 6007 and len(X) != 2905 and len(X) != 2939:
        raise
    
    output_file = 'stacking_pred_sentiment.csv'
    
    with open(output_file, 'w', newline='') as results_file:
        writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(['id', 'ground truth', 'prediction'])
    
    random_forest_prediction_opt(X, y, ids, output_file, 'sentiment')


def prediction_sentiment_bert():
    
    prepare_scores()
    
    dataset = pd.read_csv('neuronske_mreze/comment_sentiment_language_model/comments_post_sentiment.csv')
    
    X = dataset[['average_sentiment'
                 ]].values
    
    ids = dataset['id'].values
    
    print("Data shape :", X.shape)
    
    y = dataset['score'].values
    
    y = list(map(lambda score: log(score+1), y))
    
    print("X len", len(X) )
    
    if len(X) != 6007 and len(X) != 2905 and len(X) != 2939:
        raise
    
    output_file = 'stacking_pred_sentiment.csv'
    
    with open(output_file, 'w', newline='') as results_file:
        writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(['id', 'ground truth', 'prediction'])
    
    random_forest_prediction_opt(X, y, ids, output_file, 'sentiment')


def stacking_end_to_end():
    
    prediction_keywords()
    
    prediction_objects()
    
    prediction_sentiment()
    # prediction_sentiment_bert()

    features_array = scf.combine_all_features()
    scf.write_all_features(features_array)
    
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
    
    sfp.random_forest_prediction_opt(X_train, X_test, y_train, y_test, 'stacking_final_results.csv')


if __name__=="__main__":
    
    stacking_end_to_end()
    
    # prediction_sentiment()
    
    # prediction_sentiment_post_average()
    
    # prediction_sentiment_bert()
    
    # prediction_objects()
    
    # prediction_keywords()