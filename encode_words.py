# -*- coding: utf-8 -*-

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
    
    
    return encodings

if __name__=="__main__":
    s1 = [['cat', 'dog'], ['cat', 'bird'], ['monkey']]

    encodings = encode(s1)
    
    for encoding in encodings:
        print(format(encoding, '#06b'))
    
