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
import constant
from pathlib import Path

def get_labels(file='metadata.csv'):
    
    return_dict = {}
    
    with open(file, 'r') as metadata:
        lines = csv.reader(metadata)
        dataset = list(lines)

        # for x in range(len(dataset)):
        for x in range(constant.INPUT_SIZE):
            img_path = constant.DATASET_DIR + dataset[x][1] + '.jpg'
            path_obj = Path(img_path)
            if not path_obj.is_file():
                continue
            if dataset[x][2] == 'nv':
                return_dict[img_path] = 0
            elif dataset[x][2] == 'mel':
                return_dict[img_path] = 1
            
    print("Dictionary length :", len(return_dict), end="\n\n")
    # print("Random value :", return_dict['ISIC_0024306'])
                
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

def visualize(hist):
    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['acc']
    val_acc=hist.history['val_acc']
    xc=range(constant.EPOCHS)
    
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
    

def main(nn = 'softmax_classifier'):
    
    images,labels = load_data()
    
    img_data = np.array(images)
    print (img_data.shape)
    img_data=np.rollaxis(img_data, 1, 0)
    print (img_data.shape)
    img_data=img_data[0]
    print ("Final shape :", img_data.shape)
    
    classes = 2
    samples = img_data.shape[0]
    print("Number of images :", samples)
    
    Y = np_utils.to_categorical(labels, classes)
    
    print("Shuffling...")
    x,y = shuffle(img_data, Y, random_state=42)
    print("Done")

    print("Splitting...")
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print("Done")
    
    # Data augmentation
    aug = image.ImageDataGenerator(
		rotation_range=20,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
		fill_mode="nearest"
    )

    if nn == 'softmax_classifier':
        image_input = Input(shape=(224,224,3))
        
        model = VGG16(input_tensor=image_input, include_top=True, weights='imagenet')
        
        model.summary()
        
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
        # hist = new_model.fit(X_train, y_train, batch_size=constant.BATCH_SIZE, epochs=constant.EPOCHS, verbose=1, validation_data=(X_test, y_test))
        
        
        hist = new_model.fit_generator(aug.flow(X_train, y_train, 
                                                batch_size=constant.BATCH_SIZE),
                                                epochs=constant.EPOCHS, 
                                                verbose=1, 
                                                validation_data=(X_test, y_test))
        
        print('Training time: {:.2f}'.format(time.time() - start_time))
        
        
        new_model_json = new_model.to_json()
        with open("new_model.json", "w") as json_file:
            json_file.write(new_model_json)
        new_model.save_weights('trained_model.h5')
        
        (loss, accuracy) = new_model.evaluate(X_test, y_test, batch_size=10, verbose=1)
    
        print("loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
        
        
    elif nn == 'dense_layers':
        image_input = Input(shape=(224, 224, 3))

        model = VGG16(input_tensor=image_input, include_top=True, weights='imagenet')
        
        model.summary()
        
        last_layer = model.get_layer('block5_pool').output
        x = Flatten(name='flatten')(last_layer)
        x = Dense(128, activation='relu', name='fc1')(x)
        x = Dense(128, activation='relu', name='fc2')(x)
        output = Dense(classes, activation='softmax', name='output')(x)
        modified_model = Model(image_input, output)
        modified_model.summary()
        
        # Freeze layers that are not fully connected
        for layer in modified_model.layers[:-3]:
            layer.trainable = False
        
        modified_model.summary()
        
        # default lr is 0.001
        # rmsprop_opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
        
        # sgd_opt = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
        
        # adadelta_opt = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
        
        modified_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        
        start_time=time.time()
        hist = modified_model.fit(X_train, y_train, batch_size=constant.BATCH_SIZE, epochs=constant.EPOCHS, verbose=1, validation_data=(X_test, y_test))
        
        '''hist = modified_model.fit_generator(aug.flow(X_train, y_train, 
                                                batch_size=constant.BATCH_SIZE),
                                                epochs=constant.EPOCHS, 
                                                verbose=1, 
                                                validation_data=(X_test, y_test))
        
        '''
        print('Training time : {:.2f}'.format(time.time() - start_time), 'seconds')
        (loss, accuracy) = modified_model.evaluate(X_test, y_test, batch_size=10, verbose=1)
        
        print("loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
        
    else:
         raise Exception("Parameter not supported")        
    
    
    visualize(hist)

def test_softmax_classifier():

    with ('new_model.json', 'r') as json_file:
        model_json = json_file.read()
    
    model = model_from_json(model_json)
    
    model.load_weights("trained_model.h5")
    print("Loaded model from disk")
    
    # loaded_model.save('model_num.hdf5')
    # loaded_model=load_model('model_num.hdf5')
    
    

if __name__ == "__main__":
    main()
    # main('dense_layers')
    