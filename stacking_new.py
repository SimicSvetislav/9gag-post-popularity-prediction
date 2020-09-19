# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold 
from sklearn.linear_model import LinearRegression, ElasticNet, ElasticNetCV, LogisticRegression
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from keras.utils import to_categorical
from sklearn.utils.multiclass import type_of_target


def stacking_v2(model, train, y, test, n_fold):
    folds=KFold(n_splits=n_fold,random_state=None, shuffle=False)
    test_pred=np.empty((0,1),float)
    train_pred=np.empty((0,1),float)
    for train_indices,val_indices in folds.split(train,y.values):
        x_train,x_val=train.iloc[train_indices],train.iloc[val_indices]
        y_train,y_val=y.iloc[train_indices],y.iloc[val_indices]

        # print(f'{val_indices[0]} - {val_indices[len(val_indices)-1]}')
        # print("train shape:", x_train.shape)
        # print("val shape:", x_val.shape)

        model.fit(X=x_train,y=y_train.values.ravel())
        
        x_val_pred = model.predict(x_val)
        train_pred = np.append(train_pred, x_val_pred)
        
    test_pred = np.append(test_pred, model.predict(test))
    
    return test_pred.reshape(-1,1),train_pred


IMAGE_COLS = ['person', 'people', 'cat', 'dog', 'other animal', 'poster', 
              'car', 'toy', 'tree', 'glasses', 
              'building', 'electronic device', 'airplane', 'guitar',
              'pattern', 'image_text']

METADATA_COLS = ['comments count', 'type']

COMMENT_COL = ['comments']

KEYWORD_COLS = ['get', 'like', 'go', 'make', 'one', 'quarantine', 'time', 
                'good', 'know', 'day', 'guy', 'see', 'look', 'new', 
                'people', 'right', 'old', 'say', 'year', 'still']

def experiment():
    
    dataset = pd.read_csv('features_complete_v4.csv')
    X = dataset[IMAGE_COLS + METADATA_COLS + COMMENT_COL + KEYWORD_COLS]
    y = dataset[['log_score']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
     
    param_dist = {'n_estimators': range(10, 320),
                  'max_depth': range(2,50)}
    
    ###
    
    # rf_model_1 = RandomForestRegressor(n_estimators=170, max_depth=4)
    
    forest_1 = RandomForestRegressor()
    rf_model_1 = RandomizedSearchCV(forest_1, param_dist, cv=10, n_iter=10, 
                              scoring='r2', n_jobs=4, verbose=1)
    
    X_train_1 = X_train[IMAGE_COLS]
    X_test_1 = X_test[IMAGE_COLS]
    
    test_pred_1, train_pred_1 = stacking_v2(model=rf_model_1,n_fold=10, train=X_train_1,test=X_test_1,y=y_train)
    
    train_pred_1=pd.DataFrame(train_pred_1)
    test_pred_1=pd.DataFrame(test_pred_1)
    
    # print(train_pred_1.shape)
    # print(test_pred_1.shape)
    
    print("Finished images")
    
    ###
    
    # rf_model_2 = RandomForestRegressor(n_estimators=130, max_depth=8)
    forest_2 = RandomForestRegressor()
    rf_model_2 = RandomizedSearchCV(forest_2, param_dist, cv=10, n_iter=10, 
                              scoring='r2', n_jobs=4, verbose=1)
    
    X_train_2 = X_train[COMMENT_COL]
    X_test_2 = X_test[COMMENT_COL]
    
    test_pred_2 ,train_pred_2 = stacking_v2(model=rf_model_2,n_fold=10,train=X_train_2,test=X_test_2,y=y_train)
    
    train_pred_2 = pd.DataFrame(train_pred_2)
    test_pred_2 = pd.DataFrame(test_pred_2)
    
    print("Finished comments")
    
    ###
    
    # rf_model_3 = RandomForestRegressor(n_estimators=125, max_depth=5)
    
    forest_3 = RandomForestRegressor()
    rf_model_3 = RandomizedSearchCV(forest_3, param_dist, cv=10, n_iter=10, 
                              scoring='r2', n_jobs=4, verbose=1)
    
    X_train_3 = X_train[KEYWORD_COLS]
    X_test_3 = X_test[KEYWORD_COLS]
    
    test_pred_3, train_pred_3 = stacking_v2(model=rf_model_3,n_fold=10,train=X_train_3,test=X_test_3,y=y_train)
    
    train_pred_3 = pd.DataFrame(train_pred_3)
    test_pred_3 = pd.DataFrame(test_pred_3)
    
    print("Finished keywords")
    
    ###
    
    # rf_model_4 = RandomForestRegressor(n_estimators=100, max_depth=4)
    forest_4 = RandomForestRegressor()
    rf_model_4 = RandomizedSearchCV(forest_4, param_dist, cv=10, n_iter=10, 
                              scoring='r2', n_jobs=4, verbose=1)
    
    X_train_4 = X_train[METADATA_COLS]
    X_test_4 = X_test[METADATA_COLS]
    
    test_pred_4, train_pred_4 = stacking_v2(model=rf_model_4,n_fold=10,train=X_train_4,test=X_test_4,y=y_train)
    
    train_pred_4 = pd.DataFrame(train_pred_4)
    test_pred_4 = pd.DataFrame(test_pred_4)
    
    print("Finished metadata")
    
    ###
    
    df = pd.concat([train_pred_1, train_pred_2, train_pred_3, train_pred_4], axis=1)
    df_test = pd.concat([test_pred_1, test_pred_2, test_pred_3, test_pred_4], axis=1)
    
    # model = RandomForestRegressor(n_estimators=150, max_depth=6)
    model = LinearRegression()
    model.fit(df, y_train)
    
    print(df.shape)
    print(df_test.shape)
    print(y_test.shape)
    
    scores = model.score(df_test, y_test)
    
    print(scores)


if __name__ == "__main__":    
    experiment()