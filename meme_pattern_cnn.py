# -*- coding: utf-8 -*-

import numpy as np
import os
import time
from vgg16 import VGG16
from keras import optimizers
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.models import Model
from keras.models import model_from_json
from keras.models import load_model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import skimage

import matplotlib.pyplot as plt

import csv
from pathlib import Path
import pandas as pd

image_name = ''
BATCH_SIZE = 16
EPOCHS = 10 

def get_labels(file='metadata.csv'):
    
    return_dict = {}
    
    with open(file, 'r') as metadata:
        lines = csv.reader(metadata)
        dataset = list(lines)

        for x in range(1, len(dataset)):
            img_path = 'scraped/' + dataset[x][1] + "/" + dataset[x][0]
            path_obj = Path(img_path)
            if not path_obj.is_file():
                continue
            return_dict[img_path] = dataset[x][2]
            
    print("Dictionary length :", len(return_dict), end="\n\n")
                
    return return_dict

def load_data():
    
    labels = []
    images = []
    
    labels_dict = get_labels()
    
    for key, value in labels_dict.items():
        # images.append(skimage.data.imread(key))
        labels.append(value)
        
        img = image.load_img(key, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # print('Input image shape:', x.shape)
        
        images.append(x)
        
    return images, labels

def predict_pattern(img_path):
    
    images,labels = load_data()
    
    img_data = np.array(images[(ITERATION-1)*LIMIT:ITERATION*LIMIT])
    print (img_data.shape)
    img_data=np.rollaxis(img_data, 1, 0)
    print (img_data.shape)
    img_data=img_data[0]
    print ("Final shape :", img_data.shape)
    
    classes = 19
    samples = img_data.shape[0]
    print("Number of images :", samples)
    
    Y = np_utils.to_categorical(labels[(ITERATION-1)*LIMIT:ITERATION*LIMIT], classes)
    
    print("Shuffling...")
    x,y = shuffle(img_data, Y, random_state=42)
    print("Done")

    print("Splitting...")
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print("Done")
    
    image_input = Input(shape=(224,224,3))
        
    model = VGG16(input_tensor=image_input, include_top=True, weights='imagenet')
    
    model.summary()
    
    last_layer = model.get_layer("fc2").output
    out = Dense(classes, activation='softmax', name='output')(last_layer)
    new_model = Model(image_input, out)
    # new_model.summary()
    
    for layer in new_model.layers[:-1]:
        layer.trainable = False
    
    # new_model.summary()
    
    new_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    print("Training...")
    start_time = time.time()
    hist = new_model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(X_test, y_test))
    
    print('Training time: {:.2f}'.format(time.time() - start_time))
    
    
    new_model_json = new_model.to_json()
    with open("new_model.json", "w") as json_file:
        json_file.write(new_model_json)
    new_model.save_weights('trained_model.h5')
    
    print("Model and weights saved!")
    
    (loss, accuracy) = new_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

    print("loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

if __name__ == "__main__":
    predict_pattern('scraped/' + image_name)
