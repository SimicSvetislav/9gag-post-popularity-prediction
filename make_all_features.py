# -*- coding: utf-8 -*-

import mysql.connector
from mysql.connector import Error

import numpy as np

import csv

from math import log

import encode_words as ew

output_file = 'features_complete_v3.csv'

class Features:
    
    def __init__(self, post_id, comments_count, section, post_type, score, objects, pattern, image_text, keywords, comments):
        self.id = post_id
        self.comments_count = comments_count
        
        if section == 'hot':
            self.section = 1
        elif section == 'trending':
            self.section = 2
        elif section == 'fresh':
            self.section = 3
        else:
            raise
        
        if post_type == 'Photo':
            self.type = 1
        elif post_type == 'Animated':
            self.type = 2
        else:
            raise
        
        self.score = score
        self.log_score = log(score+1)
        
        self.objects = objects
        self.pattern = pattern
        
        self.keywords = None
        self.comments = comments        
    
    def get_as_list(self):
        
        
        ret_list = []
        
        ret_list.append(self.id)
        ret_list.append(self.comments_count)
        ret_list.append(self.section)
        ret_list.append(self.type)
        ret_list.extend(disassemble_binary(self.objects, 15))
        ret_list.append(self.pattern)
        ret_list.append(self.image_text)
        ret_list.append(self.comments)
        ret_list.extend(disassemble_binary(self.keywords, len(ew.WORDS_LIST_2)))
        ret_list.append(self.score)
        ret_list.append(self.log_score)
        
        # print(ret_list)
        
        if len(ret_list) != 24 + len(ew.WORDS_LIST_2):
            raise
        
        return ret_list

headers = ['id', 'comments count', 'section', 'type',
           # Objects
           'person', 'people', 'cat', 'dog', 'other animal', 'poster', 
           'clothing', 'car', 'toy', 'tree', 'glasses', 'building', 
           'electronic device', 'airplane', 'guitar',
           'pattern', 'image_text', 'comments',
           ]

# Keywords
headers.extend(ew.WORDS_LIST_2)

headers.extend(['score', 'log_score'])

# print(headers)

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
                score = int(row[5]) / int(row[4]) if int(row[4] != 0) else int(row[5])
                score = int(row[5]) / (int(row[4]+1))
                # total_votes = int(row[5]) + int(row[4])
                # score = int(row[5]) / total_votes if total_votes != 0 else 0.5
                features = Features(row[0], row[1], row[2], row[3], score, None, None, None, None, None)
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

def detection_features(features_array):
    
    with open("notebooks/mapped_detections.csv", 'r', newline='') as detected_objects_file:
        lines = list(csv.reader(detected_objects_file))
        
        objects_dict = {}
        
        for line in lines:
            objects_dict[line[0]] = line[1]
            
        to_delete = []
            
        for features in features_array:
            try:
                detection_feature_vector = objects_dict[features.id]
                
                features.objects = detection_feature_vector
            
            except:
                to_delete.append(features)
            
        for del_feature in to_delete:
            features_array.remove(del_feature)
            
    
    return features_array

def pattern_feature(features_array):
    with open("patterns_v3.csv", 'r', newline='') as patterns_file:
        lines = list(csv.reader(patterns_file))
        
        objects_dict = {}
        
        for line in lines:
            objects_dict[line[0]] = int(line[2])
            
        for features in features_array:
            pattern_feature = objects_dict[features.id]
                
            features.pattern = pattern_feature
            
    return features_array

def image_text_feature(features_array):
    
    
    with open('notebooks/image_text.csv', 'r', newline='') as image_text_file:
        
        lines = list(csv.reader(image_text_file))[1:]
        
        objects_dict = {}
        
        for line in lines:
            # 2 - text length; 3 - words
            # print(line[2])
            objects_dict[line[0].split('.')[0]] = int(line[2])
            
        for features in features_array:
            image_text_len = objects_dict[features.id]
                
            features.image_text = image_text_len
    
    return features_array

def comments_feature(features_array, missing):
    
    take_avg = 0
    
    with open("normalized_average_sentiment.csv", 'r', newline='') as comments_file:
        lines = list(csv.reader(comments_file))[1:]
        
        # Calculate average
        np_lines = np.array(lines)
        
        np_lines_values = np_lines[:, 1]
    
        mean_value = np.mean(np_lines_values.astype(np.float))
        
        print("Mean value for comments :", mean_value) 
        
        objects_dict = {}
        
        for line in lines:
            objects_dict[line[0]] = float(line[1])
            
        to_delete = []
        
        for features in features_array:
            try :
                comment_grade = objects_dict[features.id]
            except KeyError:
                
                take_avg += 1
                
                comment_grade = 0
                
                if missing == 1:
                    comment_grade = mean_value
                elif missing == 2:
                    pass
                elif missing == -1:
                    to_delete.append(features)    
                else:
                    raise
            
            features.comments = comment_grade
            
        for del_feature in to_delete:
            features_array.remove(del_feature)
            
    print("Comment grade not found for {0} posts".format(take_avg))
    
    return features_array

def keywords_feature(features_array):
    
    not_found = 0
    
    with open("keywords_encoded.csv", 'r', newline='') as comments_file:
        lines = list(csv.reader(comments_file))
        
        keywords_encoded = {}
        
        for line in lines:
            keywords_encoded[line[0]] = line[1]
        
        for features in features_array:
                
                if features.id in keywords_encoded:                   
                    encoding = keywords_encoded[features.id]
                    
                else:
                    encoding = '0'
                    not_found += 1
                
                
                features.keywords = encoding
                
    print("Not found keywords :", not_found)
    
    return features_array

def combine_all_features():
    
    features_array = database_features()    
    
    features_array = detection_features(features_array)
    
    features_array = pattern_feature(features_array)
    
    features_array = image_text_feature(features_array)
    
    features_array = comments_feature(features_array, 2)
    
    features_array = keywords_feature(features_array)
    
    index = 10
    
    print(len(features_array))
    print(features_array[index].id)
    print(features_array[index].score)
    print(features_array[index].objects)
    print("Pattern?", features_array[index].pattern)
    print(features_array[index].comments)
    # print(features_array[index].keywords)

    return features_array

def write_all_features(features_array):
    
    with open(output_file, 'w', newline='') as features_file:
        
        writer = csv.writer(features_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        writer.writerow(headers)
        
        for features in features_array:
            writer.writerow(features.get_as_list())
            
    print('Features written to file :', output_file)

def disassemble_binary(str_number, number_of_classes):
    
    bin_number = int(str_number, base=2)
    
    bin_list = [0] * number_of_classes
    mask = 0b1
    
    bin_length = bin_number.bit_length()
    
    # print("Length :", bin_length)
    
    for i in range(bin_length):
    
        bit = bin_number & mask
        
        bin_list[i] = bit
        
        bin_number = bin_number>>1
    
    # print(bin_list)
        
    if len(bin_list) != number_of_classes:
        raise
    
    return bin_list

'''
def disassemble_binary_number(bin_number):
    
    bin_list = [0] * 10
    mask = 0b1
    
    bin_length = bin_number.bit_length()
    
    # print("Length :", bin_length)
    
    for i in range(bin_length):
    
        bit = bin_number & mask
        
        bin_list[i] = bit
        
        bin_number = bin_number>>1
    
    # print(bin_list)
        
    if len(bin_list) != 10:
        raise
    
    return bin_list
'''
    
if __name__=="__main__":
    features_array = combine_all_features()
    write_all_features(features_array)
    
    # sample = '001000000100001'
    
    # disassemble_binary(sample)
    
    pass