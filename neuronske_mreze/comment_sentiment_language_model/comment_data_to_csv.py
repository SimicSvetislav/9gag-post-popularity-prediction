# -*- coding: utf-8 -*-

import csv
import mysql.connector
from mysql.connector import Error
import re

COMMENTS_FILE = 'comment_data.csv'

def extract_id(permalink):
    
    first_part = permalink.split('#')[0]
    
    # print(first_part)

    post_id = first_part[first_part.rfind('/')+1:]
    
    # print(post_id)

    return post_id

def remove_at_answer(text):
    is_answer = text.startswith('@')
    if is_answer:
        
        if ' ' in text:
            text = text[text.find(' '):]
        else:
            text = ''
    
    text = text.strip()
    
    return text

def connect_db(name):
    try:
        connection = mysql.connector.connect(host='localhost',
                                             database=name,
                                             user='root',
                                             password='root')
        if connection.is_connected():
            db_Info = connection.get_server_info()
            print("Connected to MySQL Server version ", db_Info)
            cursor = connection.cursor()
            cursor.execute("select database();")
            record = cursor.fetchone()
            print("You're connected to database: ", record)
            
            return connection
                                                                    
        else:
            raise Exception("Not connected to database!")
    
    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print("MySQL connection is closed")


def execute_query(db_name, query):
    try:
        
        connection = mysql.connector.connect(host='localhost',
                                             database=db_name,
                                             user='root',
                                             password='root')
    except Error as e:
        print("Error while connecting to MySQL", e)
    
    try:    
        cursor = connection.cursor()
        cursor.execute(query)
        records = cursor.fetchall()
        
        return records

    except Error as e:
        print("Error reading data from MySQL table", e)
    finally:
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

def transform():
    
    records = execute_query('9gag', "SELECT id, media_text, permalink FROM comment")    
    
    with open(COMMENTS_FILE, 'w', newline='', encoding='utf-8') as comments_file:
        writer = csv.writer(comments_file)
        writer.writerow(['commend_id', 'post_id', 'text_original', 'text'])
        
        for row in records:
            post_id = extract_id(row[2])
            writer.writerow([row[0], post_id, row[1], remove_at_answer(row[1])])

if __name__=="__main__":
    transform()    
    
    pass