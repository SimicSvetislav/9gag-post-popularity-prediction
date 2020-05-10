# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression, ElasticNet, ElasticNetCV
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# import autosklearn.regression

import scipy.stats as stats

import csv

features_file = 'features_complete_v3.csv'
results_file_name = 'results_v3.csv'

ALPHA = 0.1
L1_RATIO = 0.7

N_ESTIMATORS = 242
MAX_DEPTH = 5

# When using all data
# USE_DATA = ['objects', 'pattern', 'comments', 'keywords']
USE_DATA = ['objects', 'pattern', 'image_text', 'comments']
# USE_DATA = []

# Database features
# using_features = ['comments count', 'section', 'type']
using_features = ['type', 'comments count']

def multi_linear_regression(X_train, X_test, y_train, y_test):
    
    print("****** MULTIVARIABLE LINEAR REGRESSION ******", end="\n\n")
    
    multi_linear_regressor = LinearRegression()  
    multi_linear_regressor.fit(X_train, y_train)
    
    y_pred = multi_linear_regressor.predict(X_test)
    
    evaluate('Linear regression', y_test, y_pred)
        
    print("\n*********************************************", end="\n\n")

def elastic_net_prediction(X_train, X_test, y_train, y_test):
    
    print("**************** ELASTIC NET ****************", end="\n\n")
    
    enet = ElasticNet(alpha=0.1, l1_ratio=0.7)
    enet.fit(X_train, y_train)
    
    y_pred = enet.predict(X_test)
    
    evaluate('Elastic net', y_test, y_pred)
        
    print("\n*********************************************", end="\n\n")
    
    return metrics.r2_score(y_test, y_pred)

def elastic_net_prediction_opt(X_train, X_test, y_train, y_test, enet_r2):
    
    print("************** ELASTIC NET OPT **************", end="\n\n")
    
    param_dist = {'alpha': stats.expon(0, 1),
                  'l1_ratio': stats.expon(0, 1)}
    
    enet = ElasticNet()
    model_cv = RandomizedSearchCV(enet, param_dist, cv=10, n_iter=50, 
                                  scoring='r2', n_jobs=4, verbose=1
                              )
    
    
    # candidates = np.linspace(0.0, 1.0, num=21)
    
    # param_dist = {'alpha': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    #              'l1_ratio': candidates}
    
    # model_cv = ElasticNetCV(l1_ratio=[0.05, 0.15, 0.5, 0.7, 0.9, 0.95, 0.99, 1], 
    #                        alphas=np.array([0.141, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]), n_jobs=4,
    #                        eps=0.1)
    
    
    
    model_cv.fit(X_train, y_train)
    
    # print("Best params : alpha={0}; l1_ratio={1}".format(model_cv.alpha_, model_cv.l1_ratio_), end="\n\n")
    print("Best params :", model_cv.best_params_)
    print("Best score :", model_cv.best_score_)
    
    y_pred = model_cv.predict(X_test)
    
    evaluate('Elastic net opt', y_test, y_pred)
        
    print("\n*********************************************", end="\n\n")
    
    if enet_r2 < metrics.r2_score(y_test, y_pred):
        alpha = model_cv.best_params_['alpha']
        l1_ratio = model_cv.best_params_['l1_ratio']
    else:
        alpha = ALPHA
        l1_ratio = L1_RATIO

    return alpha, l1_ratio

def svr_regression(X_train, X_test, y_train, y_test):
    
    print("******************** SVR ********************", end="\n\n")
    
    svr = SVR(kernel='rbf')
    svr.fit(X_train, y_train)
    
    y_pred = svr.predict(X_test)
    
    evaluate('SVR', y_test, y_pred)
    
    print("\n*********************************************", end="\n\n")
    
    
def svr_regression_opt(X_train, X_test, y_train, y_test):
    
    print("****************** SVR OPT ******************", end="\n\n")
    
    svr = SVR(kernel='rbf')
    
    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 200]
    gammas = [0.0005, 0.001, 0.01, 0.1, 1, 10, 100]
    # Reduced parameter search
    # Cs = [100, 200, 400]
    # gammas = [0.0001, 0.0005, 0.001, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    
    regressor = GridSearchCV(svr, param_grid, cv=5,
                              scoring='r2', n_jobs=4, verbose=1)
    
    regressor.fit(X_train, y_train)
    print("Best params :", regressor.best_params_)
    print("Best score :", regressor.best_score_)
    
    y_pred = regressor.predict(X_test)
    
    evaluate('SVR opt', y_test, y_pred)
    
    print("\n*********************************************", end="\n\n")

    

def random_forest_prediction(X_train, X_test, y_train, y_test):
    
    print("*************** RANDOM FOREST ***************", end="\n\n")
    
    forest = RandomForestRegressor(n_estimators=160, max_depth=80)
    y_pred = forest.fit(X_train, y_train).predict(X_test)
    
    y_pred = forest.predict(X_test)
    
    evaluate('Random forest', y_test, y_pred)
    
    print("\n*********************************************", end="\n\n")

def random_forest_prediction_opt(X_train, X_test, y_train, y_test):
    
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
    
    evaluate('Random forest opt', y_test, y_pred)
    
    print("\n*********************************************", end="\n\n")

    n_estimators = rscv.best_params_['n_estimators']
    max_depth = rscv.best_params_['max_depth']

    return n_estimators, max_depth

def baseline_prediction(y_test):
    
    print("******************* BASELINE *****************", end="\n\n")
    
    # y_mean = np.mean(y_test)
    
    # print("Mean score :", y_mean, end="\n\n")
    
    y_median = np.median(y_test)
    
    # y_pred = [y_mean] * len(y_test)
    
    y_pred = [y_median] * len(y_test)
    
    evaluate('Baseline', y_test, y_pred)
    
    print("\n*********************************************", end="\n\n")

def vote_prediction(X_train, X_test, y_train, y_test, alpha, l1_ratio, n_estimators, max_depth):
    
    print("******************* VOTING ******************", end="\n\n")
    
    # forest = RandomForestRegressor(n_estimators=242, max_depth=5)
    # elasic_net = ElasticNet(alpha=0.141, l1_ratio=1.0)
    forest = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    elasic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    linear_regressor = LinearRegression()
    
    voting_regressor = VotingRegressor(estimators=[('rf', forest), ('enet', elasic_net), ('lr', linear_regressor)])
    voting_regressor = voting_regressor.fit(X, y)
    
    y_pred = voting_regressor.predict(X_test)
    
    evaluate('Voting', y_test, y_pred)
    
    print("\n*********************************************", end="\n\n")

'''
def auto_prediction(X_train, X_test, y_train, y_test):
    
    feature_types = []

    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=3600,
        per_run_time_limit=60
    )
    automl.fit(X_train, y_train, dataset_name='9gag complete features',
               feat_type=feature_types)


    print(automl.show_models())
    y_pred = automl.predict(X_test)
    
    evaluate("Automatic model finder", y_test, y_pred)
'''
  
def evaluate(method_name, y_test, y_pred):
    
    
    
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
    
    with open(results_file_name, 'a', newline='') as results_file:
        writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow([method_name, round(mae,4), round(mse,4), round(rmse,4), round(r2,4), str(used_features)[1:-1].replace("'", ""), round(rho,4)])
   
    with open('predictions.csv', 'a', newline='') as results_file:
        writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        if len(y_test) != len(y_pred):
            raise
            
        for i in range(len(y_test)):
            writer.writerow([y_test[i], y_pred[i]])
    
if __name__=="__main__":
    
    dataset = pd.read_csv(features_file)
    
    print(dataset.describe())
    
    
     
    # Using all data
    '''
    X = dataset[['comments count', 'section', 'type', 'person', 'people', 'cat', 'dog', 
             'other animal', 'poster', 'clothing', 'car', 'toy', 'tree', 'glasses', 
             'building', 'electronic device', 'airplane', 'guitar', 'pattern', 
             'comments', 'kw_1', 'kw_2', 'kw_3', 'kw_4', 'kw_5', 'kw_6', 'kw_7', 
             'kw_8', 'kw_9', 'kw_10',]].values
    '''
    
    # Database features
    # using_features = ['comments count', 'section', 'type']
    using_features = ['type', 'comments count']
    
    used_features = using_features.copy()
        
    used_features.extend(USE_DATA)
    
    print('Using feature groups : ', end='')
    
    if 'objects' in USE_DATA:
        print('OBJECTS ', end='')
        using_features.extend(['person', 'people', 'cat', 'dog', 
             'other animal', 'poster', 'clothing', 'car', 'toy', 'tree', 'glasses', 
             'building', 'electronic device', 'airplane', 'guitar'])
    
    if 'pattern' in USE_DATA:
        print('PATTERN ', end='')
        using_features.append('pattern')
        
    if 'image_text' in USE_DATA:
        print('IMAGE_TEXT ', end='')
        using_features.append('image_text')
    
    if 'comments' in USE_DATA:
        print('COMMENTS ', end='')
        using_features.append('comments')
    
    if 'keywords' in USE_DATA:
        print('KEYWORDS ', end='')
        using_features.extend(['kw_1', 'kw_2', 'kw_3', 'kw_4', 'kw_5', 'kw_6', 'kw_7', 'kw_8', 'kw_9', 'kw_10'])
    
    print("\nUsing features :", using_features)
    X = dataset[using_features].values
    
    label_feature = 'log_score'
    
    y = dataset[label_feature].values
    
    print("\nLabel :", label_feature)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    if len(X) != 6007 and len(X) != 2905:
        raise

    with open('predictions.csv', 'w', newline='') as results_file:
        writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(['y', 'prediction'])

    baseline_prediction(y)

    multi_linear_regression(X_train, X_test, y_train, y_test)
     
    enet_r2 = elastic_net_prediction(X_train, X_test, y_train, y_test)
    
    alpha, l1_ratio = elastic_net_prediction_opt(X_train, X_test, y_train, y_test, enet_r2)
    
    svr_regression(X_train, X_test, y_train, y_test)
    
    svr_regression_opt(X_train, X_test, y_train, y_test)
    
    random_forest_prediction(X_train, X_test, y_train, y_test)
        
    n_estimators, max_depth = random_forest_prediction_opt(X_train, X_test, y_train, y_test)
    
    vote_prediction(X_train, X_test, y_train, y_test, alpha, l1_ratio, n_estimators, max_depth)

    # auto_prediction(X_train, X_test, y_train, y_test)
    
    with open(results_file_name, 'a', newline='') as results_file:
        writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(["Samples", len(X)])
        writer.writerow([])
    
