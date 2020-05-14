# -*- coding: utf-8 -*-

import csv

'''
WORDS_LIST = ['get', 'like', 'go', 'make', 'one', 'quarantine', 'good',
              'time', 
              # 'wa', 
              'know', 'day', 'guy', 'see', 'look', 'new',
              'people', 'right', 'old', 'say', 'year'
              ]  
'''

WORDS_LIST_2 = ['get', 'like', 'go', 'make', 'one', 'quarantine', 'time', 
                'good', 'know', 'day', 'guy', 'see', 'look', 'new', 'people', 
                'right', 'old', 'say', 'year', 'still']

KEYWORDS_FILE = 'notebooks/keywords.csv'
OUTPUT_FILE = 'keywords_encoded.csv'

# Dobija niz nizova reci
def encode(words_list):
    
    encoding_dictionary = {}
    
    mask = 0b1
    
    # Pravljenje recnika
    for words in words_list:
        # Reci  nizu jedne objave
        for word in words:
            # Pojedinacne reci
            if not word in encoding_dictionary:
                encoding_dictionary[word] = mask
                mask = mask << 1
    
    encodings = [0b0] * len(words_list)
    
    for i, words in enumerate(words_list):
        for word in words:
            encodings[i] = encodings[i] | encoding_dictionary[word]
    
    
    return encodings, len(encoding_dictionary)


def disassemble_binary(bin_number, no_encodings):
    
    # print(no_encodings)
    
    bin_list = [0] * no_encodings
    mask = 0b1
    
    bin_length = bin_number.bit_length()
    
    # print("Length :", bin_length)
    
    for i in range(bin_length):
    
        bit = bin_number & mask
        
        bin_list[i] = bit
        
        bin_number = bin_number>>1 
    
    # print(bin_list) =
        
    if len(bin_list) != no_encodings:
        raise
    
    bin_list.reverse()
    
    return bin_list


def process_encoding(words_list):
    
    encodings, no_encodings = encode(words_list)
    
    for encoding in encodings:
        print(format(encoding, '#06b'))
        
    encodings_list = []    
    
    for i, encoding in enumerate(encodings):
        encodings_list.append(disassemble_binary(encodings[i], no_encodings))
    
    return encodings_list

def make_encoding_dictionary(words):
    
    encoding_dictionary = {}
    
    mask = 0b1
    
    for word in words:
        encoding_dictionary[word] = mask
        mask = mask << 1
        
    return encoding_dictionary
    

def encode_keywords(post_keywords, encoding_dictionary):
    
    encoding = 0b0
    
    for keyword in post_keywords:
        if keyword in encoding_dictionary:
            encoding = encoding | encoding_dictionary[keyword]  
        
    return encoding

def encode_all(keywords_file_path=KEYWORDS_FILE, output_file_path=OUTPUT_FILE):
    
    encoding_dictionary = make_encoding_dictionary(WORDS_LIST_2)
    
    results_dict = {}
    
    with open(keywords_file_path, 'r') as keywords_file:
        lines = list(csv.reader(keywords_file))
        
        for line in lines:
            
            encoding =  encode_keywords(line[3:], encoding_dictionary)
            
            results_dict[line[0]] = encoding
            
    
    with open(output_file_path, 'w', newline='') as output_file:
        writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
        writer.writerow(['id', 'keywords_encoding'])
        
        for key, value in results_dict.items():
             writer.writerow([key, format(value,'#021b')])
    

if __name__=="__main__":
    # s1 = [['cat', 'dog'], ['cat', 'bird'], ['monkey']]

    # encodings_list = process_encoding(s1)
    
    '''
    encoding_dictionary = make_encoding_dictionary(words_list)
    
    print("Dictionary length :", len(encoding_dictionary))
    
    for key, value in encoding_dictionary.items():
        print(format(value, '#021b'))
        
    encoding = encode_keywords(['go', 'hello', 'old', 'friend', 'make', 'me', 'coffee', 'this', 'year'], encoding_dictionary)
    print(format(encoding, '#021b'))
    '''
    
    encode_all(KEYWORDS_FILE, OUTPUT_FILE)
    