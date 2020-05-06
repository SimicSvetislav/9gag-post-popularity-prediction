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
    
    # print(bin_list)
        
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

if __name__=="__main__":
    s1 = [['cat', 'dog'], ['cat', 'bird'], ['monkey']]

    encodings_list = process_encoding(s1)
    
    print(encodings_list)
    
    for encoding in encodings_list:
        print(encoding)
    
