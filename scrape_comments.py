# -*- coding: utf-8 -*-

import mysql.connector
from mysql.connector import Error

from selenium import webdriver
from selenium.webdriver.common.by import By

import time

insert_sql = "INSERT INTO comments (post_id, comment_text, replyTo, media) VALUES (%s, %s, %s, %s)"

driver = webdriver.Chrome()

SCROLL_PAUSE_TIME = 0.7
SHORT_PAUSE_TIME = 0.2

def scrape_comments(url):
    driver.get(url)
    post_id = url[url.rfind('/')+1:]
    print(post_id)
    last_height = driver.execute_script("return document.body.scrollHeight")
    
    # Initial push
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight*0.5);")
    time.sleep(SHORT_PAUSE_TIME)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight*0.75);")
    time.sleep(SHORT_PAUSE_TIME)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight*0.85);")
    time.sleep(SHORT_PAUSE_TIME)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    
    while True:
        
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    
        time.sleep(SCROLL_PAUSE_TIME)
    
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
        
    print('END')
    
    comments_element = driver.find_element_by_css_selector('div.comment-embed')
    
    collapsed_comments = len(comments_element.find_elements(By.CLASS_NAME, 'collapsed-comment'))
    print('Collapsed: ', collapsed_comments)
    
    for i in range(collapsed_comments):
        print(i)
        driver.execute_script("document.getElementsByClassName('collapsed-comment')[0].click()")
        time.sleep(0.2)
    
    # Everythin expanded
    html_comments = driver.find_elements_by_css_selector('.comment-embed > div')[2]
    main_comments_list = html_comments.find_elements_by_xpath('.//*[@class="comment-entry"]')

    print("Main comments: ", len(main_comments_list))

    for entry in main_comments_list:
        content = entry.find_element_by_css_selector('.payload > .content')
        mc_text = content.get_attribute('innerHTML')
        
        media_url = None
        
        try:
            media_el = entry.find_element_by_css_selector('.payload > .media > a')
            media_url = media_el.get_attribute('href')
            # print("MEDIA: ", media_url.get_attribute('href'))
        except:
            pass
        
        # Writing main comment
        cursor.execute(insert_sql, (post_id, mc_text, None, media_url))
        connection.commit()
        print("Main comment inserted.")
        last_id = cursor.lastrowid
        # print(last_id)
        
        next_sibling = entry.find_element_by_xpath('./following-sibling::div')
        replies = next_sibling.find_elements_by_xpath('./div/div[@class="comment-entry indent-1"]')
        # print('REPLIES: ', len(replies))
        
        for reply in replies:
            # print(reply.get_attribute('innerHTML'))
            reply_content = reply.find_element_by_css_selector('.payload > .content')
            reply_text = reply_content.get_attribute('innerText')
            
            reply_media_url = None
            
            try:
                reply_media = reply.find_element_by_css_selector('.payload > .media > a')
                reply_media_url =  reply_media.get_attribute('href')
                # print("MEDIA: ", reply_media.get_attribute('href'))
            except:
                pass
            
            # Writing reply comment
            cursor.execute(insert_sql, (post_id, reply_text, last_id, reply_media_url))
            connection.commit()
            print("\tReply comment inserted.")
        
        connection.commit()
        print()


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
        sql_select_Query = "SELECT url, comments_count FROM actual_post LIMIT 10"
        cursor = connection.cursor()
        cursor.execute(sql_select_Query)
        records = cursor.fetchall()
        
        print("Total number of posts: ", cursor.rowcount)
        
        for row in records:
            url = row[0]
            comments = row[1]
            
            print('URL: ', url)
            
            scrape_comments(url)
            
            print('***************************************')
        
            
        driver.close()
        
    except Error as e:
        print("Error reading data from MySQL table", e)

except Error as e:
    print("Error while connecting to MySQL", e)
finally:
    if (connection.is_connected()):
        cursor.close()
        connection.close()
        print("MySQL connection is closed")

