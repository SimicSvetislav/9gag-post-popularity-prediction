# -*- coding: utf-8 -*-

import mysql.connector
from mysql.connector import Error

from PIL import Image
import requests
from io import BytesIO

import csv

import os

from shutil import copyfile

CATEGORY = 'Random'
CATEGORY_ID = 19

def extend_download():    
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
            sql_select_Query = "SELECT id, image_url FROM actual_post"
            cursor = connection.cursor()
            cursor.execute(sql_select_Query)
            records = cursor.fetchall()
        
            limit = 1500
        
            
            print("\nPrinting urls")
        
            metadata_file = open('metadata.csv', mode='a', newline='')
            writer = csv.writer(metadata_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)  
            
            for row in records:
                img_url = row[1].replace('\\', '')
                print("Image url: ", img_url)
                img_res = requests.get(img_url)
                try:
                    Image.open(BytesIO(img_res.content)).save('scraped/' + CATEGORY + row[0] + '/' + '.jpg')
                    
                    writer.writerow([row['id'], CATEGORY, CATEGORY_ID])
                    
                    limit -= 1
                    
                    if limit == 0:
                        break
                    
                except:
                    print("Error while getting image: ", img_url)
                
            print("Total number of image urls: ", cursor.rowcount)
                
        except Error as e:
            print("Error reading data from MySQL table", e)
    
    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

def extend_metadata(limit=-1):
    
    # total = 0
    
    with open('metadata.csv', mode='a', newline='') as metadata_file:
        
        writer = csv.writer(metadata_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL) 
        
        for file in os.listdir('scraped/' + CATEGORY):
            
            # print(file)
            # print(img_id)
            
            writer.writerow([file, CATEGORY, CATEGORY_ID])
            
            limit -= 1
            
            if limit == 0:
                break
            
            # total += 1
        
    # print(total)

def copy_images(images_folder='scraped_database_tags_new/'):
    
    with open('metadata.csv', mode='a', newline='') as metadata_file:
            
        lines = list(csv.reader(metadata_file))
        
        for line in lines:
            if line[1] == 'Random':
                img_path = images_folder + line[0]
                copyfile(img_path, 'scraped/Random')
        

    
if __name__=="__main__":
    
    # extend_download()
    
    extend_metadata(limit=1500)
    
    # copy_images()
    
    # copyfile('scraped_database_tags_new/')
    
    pass
