# -*- coding: utf-8 -*-

import scrapy
from PIL import Image
import requests
from io import BytesIO
import csv

hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Connection': 'keep-alive'}


MEME_PATTERNS = ['Socially-Awkward-Awesome-Penguin',
                 'Ancient-Aliens',
                 'One-Does-Not-Simply',
                 'X-X-Everywhere',
                 'Futurama-Fry',
                 'Woman-Yelling-At-Cat',
                 'Distracted-Boyfriend',
                 'Drake-Hotline-Bling'
                 ]

SCRAPING_CATEGORIES = ['Socially-Awkward-Awesome-Penguin',
                       'Ancient-Aliens',
                       'One-Does-Not-Simply',
                       'X-X-Everywhere',
                       'Futurama-Fry',
                       'Woman-Yelling-At-Cat',
                       'Distracted-Boyfriend',
                       'Drake-Hotline-Bling'
                       ]

category_dict = dict(zip(MEME_PATTERNS, range(len(MEME_PATTERNS))))

class MemeSpider(scrapy.Spider):
    name = 'meme-pattern-spider'
    start_urls = ['https://imgflip.com/meme/' + SCRAPING_CATEGORIES[i]
                  for i in range(len(SCRAPING_CATEGORIES))]
    metadata_file = open('metadata.csv', mode='a', newline='')
    writer = csv.writer(metadata_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    def parse(self, response):
        
        i = response.request.url.rfind('?page=')
        if i != -1:
            page_num = int(response.request.url[i+6:])
            if page_num > 100:
                raise scrapy.exceptions.CloseSpider('Maximum number of pages scraped')
        
        for img_link in response.css('.base-img-link'):
            category = self.get_category(response)
            img_src = img_link.css('::attr(src)').get()
            ind = img_src.rfind('/')
            
            self.writer.writerow([img_src[ind+1:], category.replace('-', ' '), category_dict[category]])
            
            img_res = requests.get('https:' + img_src)
            Image.open(BytesIO(img_res.content)).save('scraped/' + img_src[ind+1:])
            
            yield {'src': img_src}

        for next_page in response.css('.pager-next'):
            yield response.follow(next_page, self.parse)
         
    def get_category(self, response):
        i = str(response).rfind('/')
        j = str(response).find('?')
        return str(response)[i+1:j]
    
    def __del__(self):
        self.metadata_file.close()