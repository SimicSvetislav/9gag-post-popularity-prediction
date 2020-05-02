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
from math import ceil
import random

from os import listdir

image_name = ''
BATCH_SIZE = 32
EPOCHS = 5

def get_labels(file='metadata.csv', evaluation=False):
    
    return_dict = {}
    
    with open(file, 'r') as metadata:
        lines = csv.reader(metadata)
        dataset = list(lines)

        samples = len(dataset)
    
        if evaluation:
            samples = 2001

        for x in range(1, samples):
        # for x in range(1, INPUT_SIZE):
        # for x in range(1, len(dataset)):
            img_path = 'scraped/' + dataset[x][1] + "/" + dataset[x][0]
            path_obj = Path(img_path)
            if not path_obj.is_file():
                continue
            return_dict[img_path] = dataset[x][2]
            
    print("Dictionary length :", len(return_dict), end="\n\n")
                
    return return_dict

def get_data_for_generator(file='metadata.csv'):
    img_files = []
    labels = []
    
    with open(file, 'r') as metadata:
        lines = csv.reader(metadata)
        dataset = list(lines)

        for x in range(1, len(dataset)):
            img_path = 'scraped/' + dataset[x][1] + "/" + dataset[x][0]
            path_obj = Path(img_path)
            if not path_obj.is_file():
                continue
            img_files.append(img_path)
            labels.append(dataset[x][2])
            
    print("Total samples :", len(img_files), end="\n\n")
                
    return img_files, labels

def load_data(evaluation=False):
    
    labels = []
    images = []
    
    labels_dict = get_labels(evaluation=evaluation)
    
    print("Reading images...")
    
    for key, value in labels_dict.items():
        # images.append(skimage.data.imread(key))
        labels.append(value)
        
        img = image.load_img(key, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # print('Input image shape:', x.shape)
        
        images.append(x)
        
        if len(images) % 500 == 0:
            print(len(images))
        
    return np.array(images), labels

def data_generator(img_files, labels, batch_size, classes):
    batch_num = 0
    list_X = []
    categorical_y = np_utils.to_categorical(labels, classes)
    while True:
        i = 0
        batch_X = []
        batch_y = []
        for i in range(batch_size):
            
            current_index = batch_num*batch_size+i
            
            img = image.load_img(img_files[current_index], target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            list_X.append(x)
            batch_y.append(categorical_y[current_index])
            
        img_data = np.array(list_X)
        print (img_data.shape)
        img_data=np.rollaxis(img_data, 1, 0)
        print (img_data.shape)
        img_data=img_data[0]
        print ("Final shape :", img_data.shape)
                
        print("Yielding ", len(batch_X))
        batch_num += 1
        yield batch_X, batch_y

def visualize(hist):
    
    try:
        train_loss=hist.history['loss']
        val_loss=hist.history['val_loss']
        train_acc=hist.history['acc']
        val_acc=hist.history['val_acc']
        xc=range(EPOCHS)
        
        plt.figure(1,figsize=(7,5))
        plt.plot(xc,train_loss)
        plt.plot(xc,val_loss)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('train_loss vs val_loss')
        plt.grid(True)
        plt.legend(['train','val'])
        
        plt.style.use(['classic'])
        
        plt.figure(2,figsize=(7,5))
        plt.plot(xc,train_acc)
        plt.plot(xc,val_acc)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('train_acc vs val_acc')
        plt.grid(True)
        plt.legend(['train','val'],loc=4)
        
        plt.style.use(['classic'])
    except:
        print("Error occured while plotting histogram")

def train_network():
    
    img_data,labels = load_data()
    
    # img_data = np.array(images)
    print (img_data.shape)
    img_data=np.rollaxis(img_data, 1, 0)
    print (img_data.shape)
    img_data=img_data[0]
    print ("Final shape :", img_data.shape)
    
    classes = 20
    samples = img_data.shape[0]
    print("Number of images :", samples)
    
    Y = np_utils.to_categorical(labels, classes)
    
    # print("shuffling...")
    # x,y = shuffle(img_data, y, random_state=42)
    # print("done")

    print("Splitting...")
    X_train, X_test, y_train, y_test = train_test_split(img_data, Y, test_size=0.2, random_state=42)
    print("Done")
    
    image_input = Input(shape=(224,224,3))
        
    model = VGG16(input_tensor=image_input, include_top=True, weights='imagenet')
    
    # model.summary()
    
    last_layer = model.get_layer("fc2").output
    out = Dense(classes, activation='softmax', name='output')(last_layer)
    new_model = Model(image_input, out)
    # new_model.summary()
    
    for layer in new_model.layers[:-1]:
        layer.trainable = False
    
    new_model.summary()
    
    new_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    print("Training...")
    start_time = time.time()
    hist = new_model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, shuffle=True, validation_data=(X_test, y_test))
    # new_model.fit_generator(data_generator(img_files, labels_meta, BATCH_SIZE, classes), 
    #                        steps_per_epoch=ceil(len(labels_dict) / BATCH_SIZE),
    #                        epochs=10, 
    #                        verbose=1)
    
    print('Training time: {:.2f}'.format(time.time() - start_time))
    
    
    new_model_json = new_model.to_json()
    with open("meme_patterns_model.json", "w") as json_file:
        json_file.write(new_model_json)
    new_model.save_weights('meme_patterns_weights.h5')
    
    print("Model and weights saved!")
    
    (loss, accuracy) = new_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

    print("loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
    
    visualize(hist)
    

def evaluate():
    
    img_data,labels = load_data(evaluation=False)
    
    # Sanity check
    # random.shuffle(labels)
        
    img_data=np.rollaxis(img_data, 1, 0)
    img_data=img_data[0]
    classes = 20
    samples = img_data.shape[0]
    print("Number of images :", samples)
    print("Number of labels :", len(labels))
    print("Final shape :", img_data.shape)
    
    Y = np_utils.to_categorical(labels, classes)
    
    json_file = open('meme_patterns_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("meme_patterns_weights.h5")
    print("Loaded model from disk")
    
    loaded_model.summary()
    
    # evaluate loaded model on test data
    # loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # score = loaded_model.evaluate(img_data, Y, verbose=1)
    # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    
    y_pred = loaded_model.predict(img_data, verbose=1)
    smth = np.argmax(y_pred, axis=1)
    
    total = len(smth)
    false = 0
    for i in range(len(smth)):
        if smth[i] != int(labels[i]):
            false += 1
            print("Bad classification")
            print(smth[i], ' - ', labels[i], '(' + str(i) + ')')
            
    print("Manual accuracy : {} %".format((total-false)/total*100))
    
def load_post_images(file='scraped_database_tags_new/'):
    
    
    images = []
    ids = []
    
    for img_file in os.listdir(file):
        
        img_id = img_file[:img_file.rfind(".")]
        # print(img_id)
        
        ids.append(img_id)
        
        img = image.load_img(file + img_file, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # print('Input image shape:', x.shape)
        
        images.append(x)
        
        if len(images) % 500 == 0:
            print(len(images))
        
    return np.array(images), ids
    
    
def detect_patterns_posts():
    
    img_data, ids = load_post_images()
    
    img_data=np.rollaxis(img_data, 1, 0)
    img_data=img_data[0]
    samples = img_data.shape[0]
    print("Number of images :", samples)
    print("Final shape :", img_data.shape)
    
    json_file = open('meme_patterns_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("meme_patterns_weights.h5")
    print("Loaded model from disk")
    
    loaded_model.summary()
    
    predictions = loaded_model.predict(img_data, verbose=1)
    best_matches = np.argmax(predictions, axis=1)
    
    no_pattern = 0
    
    mapped_predictions = []
    
    for i in range(len(best_matches)):
        if best_matches[i] == 19:
            no_pattern += 1
            mapped_predictions.append(0)
        else:
            mapped_predictions.append(1)
        print('{0}, '.format(best_matches[i]), end='')
    
    print("\nPattern not detected on {0} of {1} images". format(no_pattern, len(best_matches)))
    
    
    print("Writing patterns data...")
    
    with open("patterns.csv", 'a', newline='') as patterns_file:
        writer = csv.writer(patterns_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        for i in range(len(ids)):
            writer.writerow([ids[i], best_matches[i], mapped_predictions[i]])
        
    
if __name__ == "__main__":
    # train_network()
    # evaluate()
    # predict_pattern('scraped/' + image_name)
    detect_patterns_posts() 
