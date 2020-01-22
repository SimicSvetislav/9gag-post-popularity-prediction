# -*- coding: utf-8 -*-

import mysql.connector
from mysql.connector import Error

from PIL import Image
import requests
from io import BytesIO

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
    
        print("\nPrinting urls")
        for row in records:
            img_url = row[1]
            print("Image url: ", img_url)
            img_res = requests.get(img_url)
            Image.open(BytesIO(img_res.content)).save('scraped_database/' + row[0] + '.jpg')
            
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
