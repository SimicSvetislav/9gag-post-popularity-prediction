# -*- coding: utf-8 -*-

import mysql.connector
from mysql.connector import Error

import numpy as np

import csv

import math

class Features:
    
    def __init__(self, post_id, comments_count, post_type, score, image_pred, sentiment_pred, keywords_pred):
        self.id = post_id
        self.comments_count = comments_count
        
        
        if post_type == 'Photo': 
            self.type = 1
        elif post_type == 'Animated':
            self.type = 2
        else:
            raise
        
        self.score = score
        self.log_score = math.log(score+1)
        
        self.image_pred = image_pred
        
        self.sentiment_pred = sentiment_pred
        self.keywords_pred = keywords_pred        
    
    def get_as_list(self):
        
        
        ret_list = []
        
        ret_list.append(self.id)
        
        ret_list.append(self.comments_count)
        ret_list.append(self.type)
        
        ret_list.append(self.image_pred)
        ret_list.append(self.sentiment_pred)
        ret_list.append(self.keywords_pred)
        
        ret_list.append(self.score)
        ret_list.append(self.log_score)
                
        # print(ret_list)
        
        
        if len(ret_list) != 8:
            raise
        
        return ret_list

headers = ['id', 'comments count', 'type',
           'image_pred',
           'sentiment_pred',
           'keywords_pred',
           'score',
           'log_score']

def database_features():
    
    features_array = []
    
    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='9gag',
                                             user='root',
                                             password='root')
        if connection.is_connected():
            db_Info = connection.get_server_info()
            print("Connected to MySQL Server version ", db_Info)
            cursor = connection.cursor()
            cursor.execute("select database();")
            record = cursor.fetchone()
            print("You're connected to database: ", record)
    
        try:
            sql_select_Query = "SELECT id, comments_count, section, type, down_vote_count, up_vote_count FROM actual_post"
            cursor = connection.cursor()
            cursor.execute(sql_select_Query)
            records = cursor.fetchall()
        
            for row in records:
                # score = int(row[5]) / int(row[4]) if int(row[4] != 0) else int(row[5])
                score = int(row[5]) / (int(row[4]+1))
                features = Features(row[0], row[1], row[3], score, None, None, None)
                features_array.append(features)
                
        except Error as e:
            print(e)
    
    except Error as e:
        print(e)
    finally:
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
            
    return features_array

def get_image_predictions(features_array):
    
    with open("stacking_pred_obj.csv", 'r', newline='') as images_pred_file:
        lines = list(csv.reader(images_pred_file))[1:]
        
        objects_dict = {}
        
        for line in lines:
            objects_dict[line[0]] = line[2]
            
        to_delete = []
            
        for features in features_array:
            try:
                detection_feature_vector = objects_dict[features.id]
                
                features.image_pred = detection_feature_vector
            
            except:
                to_delete.append(features)
            
        for del_feature in to_delete:
            features_array.remove(del_feature)
            
    
    return features_array

def get_sentiment_predictions(features_array):
    
    with open("stacking_pred_sentiment.csv", 'r', newline='') as images_pred_file:
        lines = list(csv.reader(images_pred_file))[1:]
        
        # Calculate average
        np_lines = np.array(lines)
        
        np_lines_values = np_lines[:, 1]
    
        mean_sentiment = np.mean(np_lines_values.astype(np.float))
        
        print("Mean sentiment for comments :", mean_sentiment) 
        
        objects_dict = {}
        
        for line in lines:
            objects_dict[line[0]] = line[2]
            
        # to_delete = []
        
        for features in features_array:
            try:
                detection_feature_vector = objects_dict[features.id]
                
                features.sentiment_pred = detection_feature_vector
            
            except:
                features.sentiment_pred = mean_sentiment
                # to_delete.append(features)
            
        # for del_feature in to_delete:
        #    features_array.remove(del_feature)
            
    
    return features_array

def get_keywords_predictions(features_array):
    
    with open("stacking_pred_keywords.csv", 'r', newline='') as keywords_pred_file:
        lines = list(csv.reader(keywords_pred_file))[1:]
        
        objects_dict = {}
        
        for line in lines:
            objects_dict[line[0]] = line[2]
            
        to_delete = []
            
        for features in features_array:
            try:
                detection_feature_vector = objects_dict[features.id]
                
                features.keywords_pred = detection_feature_vector
            
            except:
                to_delete.append(features)
            
        for del_feature in to_delete:
            features_array.remove(del_feature)
            
    
    return features_array

def combine_all_features():
    
    features_array = database_features()    
    
    features_array = get_image_predictions(features_array)
    
    features_array = get_sentiment_predictions(features_array)
    
    features_array = get_keywords_predictions(features_array)
    
    index = 10
    
    print(len(features_array))
    print(features_array[index].id)
    print(features_array[index].score)
    print(features_array[index].log_score)
    print(features_array[index].comments_count)
    print(features_array[index].image_pred)
    print(features_array[index].sentiment_pred)
    print(features_array[index].keywords_pred)

    return features_array

def write_all_features(features_array):
    
    with open('stacking_features_complete.csv', 'w', newline='') as features_file:
        
        writer = csv.writer(features_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(headers)
        
        for features in features_array:
            writer.writerow(features.get_as_list())

    
if __name__=="__main__":
    features_array = combine_all_features()
    write_all_features(features_array)
    