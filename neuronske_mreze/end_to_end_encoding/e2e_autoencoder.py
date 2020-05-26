# -*- coding: utf-8 -*-
import numpy as np
import os
from skimage.io import imsave
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.callbacks import Callback
from keras.preprocessing import image
from  keras.utils import Sequence
from sklearn.model_selection import train_test_split
import pathlib
import matplotlib.pyplot as plt
import csv

IMAGES_FOLDER = '../../scraped_database_tags_new/'
INPUT_SHAPE = (512, 512, 3) 
INPUT = Input(shape=INPUT_SHAPE)
ENCODED_SHAPE = (32, 32, 8) 

EPOCHS = 10

BATCH_SIZE = 2
MODEL_NAME = 'model_c_r'
OPTIMIZER = 'adam'
IMAGES_OUTPUT_DIRECTORY = 'validation_images_gen/'
TEST_SIZE = 0.1
LOSS_FUNCTION = 'mse'

OUTPUT_FILE = 'encoded_images.csv'

LIMIT = -1

np.random.seed(42)

if not os.path.exists(IMAGES_OUTPUT_DIRECTORY):
    os.mkdir(IMAGES_OUTPUT_DIRECTORY)


class AutoencoderGenerator(Sequence):
    def __init__(self, image_filenames, batch_size, input_shape) :
        self.image_filenames = image_filenames
        self.batch_size = batch_size
        self.input_shape = input_shape
        
    def __len__(self) :
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  
    def __getitem__(self, idx) :
        
        batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        
        batch_filenames = ([file_name for file_name in batch_x])
        
        batch_images = [] 
        
        for filename in batch_filenames:
            img = image.load_img(IMAGES_FOLDER + filename, target_size=self.input_shape)
        
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            
            batch_images.append(x)
        
        # (resize(imread(IMAGES_FOLDER + str(file_name)), (128, , 3)))
    
        np_batch_images = np.array(batch_images)
    
        np_batch_images = np_batch_images.astype('float32') / 255.
        
        np_batch_images = np.rollaxis(np_batch_images, 1, 0)[0]
        
        # print("Batch images size :", np_batch_images.shape)
        
        return np_batch_images, np_batch_images


def load_images(input_shape):
    # images = [np.empty((1, input_shape[0], input_shape[1], input_shape[2]), float)]
    images = []
    
    for i, file in enumerate(os.listdir(IMAGES_FOLDER)):

        img = image.load_img(IMAGES_FOLDER + file, target_size=input_shape)

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        images.append(x)
        
        if len(images) % 500 == 0:
            print(len(images))
        
        if LIMIT != -1 and LIMIT < i:
            break
        
    np_images = np.array(images)
    
    return np_images 

def prepare_images():
    
    images = load_images(INPUT_SHAPE)
    print(images.shape)
    
    img_data = np.rollaxis(images, 1, 0)[0]
    print(img_data.shape)
    
    print("Splitting...")
    X_train, X_test, y_train, y_test = train_test_split(img_data, img_data, test_size=TEST_SIZE, random_state=42)
    print("Done")
    
    return X_train, X_test, y_train, y_test

def build_autoencoder_vgg():
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='encoder_block1_conv1')(INPUT)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='encoder_block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='encoder_block1_pool')(x)
    
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='encoder_block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='encoder_block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='encoder_block2_pool')(x)
    
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='encoder_block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='encoder_block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='encoder_block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='encoder_block3_pool')(x)
    
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='encoder_block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='encoder_block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='encoder_block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='encoder_block4_pool')(x)
    
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='encoder_block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='encoder_block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='encoder_block5_conv3')(x)
    
    encoded = MaxPooling2D((2, 2), strides=(2, 2), name='encoder_block5_pool')(x)
    print("Shape of encoded representation :", encoded.shape)
    
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='decoder_block1_conv1')(encoded)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='decoder_block1_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='decoder_block1_conv3')(x)
    x = UpSampling2D((2, 2), name='decoder_block1_upscale')(x)
    
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='decoder_block2_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='decoder_block2_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='decoder_block2_conv3')(x)
    x = UpSampling2D((2, 2), name='decoder_block2_upscale')(x)
                     
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='decoder_block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='decoder_block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='decoder_block3_conv3')(x)
    x = UpSampling2D((2, 2), name='decoder_block3_upscale')(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='decoder_block4_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='decoder_block4_conv2')(x)
    x = UpSampling2D((2, 2), name='decoder_block4_upscale')(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='decoder_block5_conv1')(x)
    x = Conv2D(64, (3, 3), activation='sigmoid', padding='same', name='decoder_block5_conv2')(x)
    x = UpSampling2D((2, 2), name='decoder_block5_upscale')(x)
    
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='decoder_output')(x)
    
    autoencoder = Model(INPUT, decoded)
    autoencoder.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION)
    
    print("Input shape :", INPUT.shape)
    print("Bottleneck shape :", encoded.shape)
    print("Output shape :", decoded.shape)
    
    encoder = Model(INPUT, encoded)
    
    encoded_input = Input(shape=(4,4,512))
    
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='decoder_block1_conv1')(encoded_input)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='decoder_block1_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='decoder_block1_conv3')(x)
    x = UpSampling2D((2, 2), name='decoder_block1_upscale')(x)
    
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='decoder_block2_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='decoder_block2_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='decoder_block2_conv3')(x)
    x = UpSampling2D((2, 2), name='decoder_block2_upscale')(x)
                     
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='decoder_block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='decoder_block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='decoder_block3_conv3')(x)
    x = UpSampling2D((2, 2), name='decoder_block3_upscale')(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='decoder_block4_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='decoder_block4_conv2')(x)
    x = UpSampling2D((2, 2), name='decoder_block4_upscale')(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='decoder_block5_conv1')(x)
    x = Conv2D(64, (3, 3), activation='sigmoid', padding='same', name='decoder_block5_conv2')(x)
    x = UpSampling2D((2, 2), name='decoder_block5_upscale')(x)
    
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='decoder_output')(x)

    return autoencoder, encoder, decoder

def build_autoencoder_simple_bigger_filters():
    
    input_img = Input(shape=INPUT_SHAPE)  # adapt this if using `channels_first` image data format

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    
    print("Shape of encoded representation :", encoded.shape)
    print(encoded.shape[1])
    
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    # autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION)
    
    encoder = Model(input_img, encoded)
    
    # encoding_dim = 128
    # encoded_input = Input(shape=(encoding_dim,))
    
    # encoded_shape = (float(encoded.shape[1]), float(encoded.shape[2]), float(encoded.shape[3]))
    encoded_input = Input(shape=(16, 16, 16))
    # encoded_input_shape = Input(shape=(encoded.shape[1], encoded.shape[2], encoded.shape[3]))
    
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded_input)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x) 
    
    decoder = Model(encoded_input, decoded)
    
    return autoencoder, encoder, decoder

def build_autoencoder_model_a():
    
    input_img = Input(shape=INPUT_SHAPE)  # adapt this if using `channels_first` image data format

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    
    print("Shape of encoded representation :", encoded.shape)
    print(encoded.shape[1])
    
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    # autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION)
    
    encoder = Model(input_img, encoded)
    
    # encoding_dim = 128
    # encoded_input = Input(shape=(encoding_dim,))
    
    # encoded_shape = (float(encoded.shape[1]), float(encoded.shape[2]), float(encoded.shape[3]))
    encoded_input = Input(shape=(8, 8, 16))
    # encoded_input_shape = Input(shape=(encoded.shape[1], encoded.shape[2], encoded.shape[3]))
    
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded_input)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x) 
    
    decoder = Model(encoded_input, decoded)
    
    return autoencoder, encoder, decoder

def build_autoencoder_model_b():
    
    input_img = Input(shape=INPUT_SHAPE)  # adapt this if using `channels_first` image data format

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    
    print("Shape of encoded representation :", encoded.shape)
    print(encoded.shape[1])
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    # autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION)
    
    encoder = Model(input_img, encoded)
    
    # encoding_dim = 128
    # encoded_input = Input(shape=(encoding_dim,))
    
    # encoded_shape = (float(encoded.shape[1]), float(encoded.shape[2]), float(encoded.shape[3]))
    encoded_input = Input(shape=(64, 64, 32))
    # encoded_input_shape = Input(shape=(encoded.shape[1], encoded.shape[2], encoded.shape[3]))
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded_input)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x) 
    
    decoder = Model(encoded_input, decoded)
    
    return autoencoder, encoder, decoder

def build_autoencoder_model_c():
    
    input_img = Input(shape=INPUT_SHAPE)  # adapt this if using `channels_first` image data format

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    
    print("Shape of encoded representation :", encoded.shape)
    print(encoded.shape[1])
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    # autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION)
    
    encoder = Model(input_img, encoded)
    
    # encoding_dim = 128
    # encoded_input = Input(shape=(encoding_dim,))
    
    # encoded_shape = (float(encoded.shape[1]), float(encoded.shape[2]), float(encoded.shape[3]))
    encoded_input = Input(shape=(32, 32, 64))
    # encoded_input_shape = Input(shape=(encoded.shape[1], encoded.shape[2], encoded.shape[3]))
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    decoder = Model(encoded_input, decoded)
    
    return autoencoder, encoder, decoder

def build_autoencoder_model_c_r():
    
    input_img = Input(shape=INPUT_SHAPE)  # adapt this if using `channels_first` image data format

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    
    print("Shape of encoded representation :", encoded.shape)
    print(encoded.shape[1])
    
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    # autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION)
    
    encoder = Model(input_img, encoded)
    
    # encoding_dim = 128
    # encoded_input = Input(shape=(encoding_dim,))
    
    # encoded_shape = (float(encoded.shape[1]), float(encoded.shape[2]), float(encoded.shape[3]))
    encoded_input = Input(shape=(32, 32, 8))
    # encoded_input_shape = Input(shape=(encoded.shape[1], encoded.shape[2], encoded.shape[3]))
    
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded_input)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x) 
    
    decoder = Model(encoded_input, decoded)
    
    return autoencoder, encoder, decoder

def build_autoencoder_simple():
    
    input_img = Input(shape=INPUT_SHAPE)  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    
    print("Shape of encoded representation :", encoded.shape)
    print(encoded.shape[1])
    
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    # autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION)
    
    encoder = Model(input_img, encoded)
    
    # encoding_dim = 128
    # encoded_input = Input(shape=(encoding_dim,))
    
    # encoded_shape = (float(encoded.shape[1]), float(encoded.shape[2]), float(encoded.shape[3]))
    encoded_input = Input(shape=ENCODED_SHAPE)
    # encoded_input_shape = Input(shape=(encoded.shape[1], encoded.shape[2], encoded.shape[3]))
    
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded_input)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    decoder = Model(encoded_input, decoded)
    
    return autoencoder, encoder, decoder

def build_autoencoder_simple_double_conv():
    
    input_img = Input(shape=INPUT_SHAPE)  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    
    print("Shape of encoded representation :", encoded.shape)
    print(encoded.shape[1])
    
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION)
    
    encoded_input = Input(shape=(32, 32, 8)) 
    
    encoder = Model(input_img, encoded)
    
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded_input)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    decoder = Model(encoded_input, decoded)
    
    return autoencoder, encoder, decoder


def visualize(history):
    
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    # train_acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']
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

def combine_images(inpt_imgs, outpt_imgs):
    num = inpt_imgs.shape[0]
    width = num
    height = 2
    shape = inpt_imgs.shape[1:]
    image = np.zeros((height * shape[0], width * shape[1], 3), dtype=inpt_imgs.dtype)
    combined = np.concatenate((inpt_imgs, outpt_imgs))
    for index, img in enumerate(combined):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = img
    return image

def generate_images(input_imgs, model):
    
    output_imgs = model.predict(input_imgs)
    # output_imgs = output_vect.reshape(output_vect.shape[0], 28, 28)
    
    input_imgs = (input_imgs * 255).astype('uint8')
    output_imgs = (output_imgs * 255).astype('uint8')
    
    combined_img = combine_images(input_imgs, output_imgs)

    imsave(IMAGES_OUTPUT_DIRECTORY + '{0}_e{1}_b{2}_{3}.jpg'.format(MODEL_NAME, EPOCHS, BATCH_SIZE, LOSS_FUNCTION), combined_img)

def evaluate(autoencoder, encoder, decoder, x_test):
    
    encoded_imgs = encoder.predict(x_test)
    
    decoded_imgs = decoder.predict(encoded_imgs)
    
    decoded_imgs = autoencoder.predict(x_test)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i+1)
        plt.imshow(x_test[i].reshape(INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        # display reconstruction
        ax = plt.subplot(2, n, i + n+1)
        plt.imshow(decoded_imgs[i].reshape(INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def get_images_filenames(img_filenames):
    test_images = [] 
    
    for filename in img_filenames:
        img = image.load_img(IMAGES_FOLDER + filename, target_size=INPUT_SHAPE)
    
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        test_images.append(x)
    
    # (resize(imread(IMAGES_FOLDER + str(file_name)), (128, , 3)))

    np_batch_images = np.array(test_images)

    np_batch_images = np_batch_images.astype('float32') / 255.
    
    np_batch_images = np.rollaxis(np_batch_images, 1, 0)[0]

    return np_batch_images

def generate_images_gen(img_filenames, model):
    
    input_imgs = get_images_filenames(img_filenames)
    
    output_imgs = model.predict(input_imgs)
    # output_imgs = output_vect.reshape(output_vect.shape[0], 28, 28)
    
    input_imgs = (input_imgs * 255).astype('uint8')
    output_imgs = (output_imgs * 255).astype('uint8')
    
    combined_img = combine_images(input_imgs, output_imgs)

    imsave(IMAGES_OUTPUT_DIRECTORY + '{0}_e{1}_b{2}_{3}_{4}.jpg'.format(MODEL_NAME, EPOCHS, BATCH_SIZE, LOSS_FUNCTION, INPUT_SHAPE[0]), combined_img)

def evaluate_gen(autoencoder, encoder, decoder, filenames_test):
       
    test_images = [] 
    
    for filename in filenames_test:
        img = image.load_img(IMAGES_FOLDER + filename, target_size=INPUT_SHAPE)
    
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        test_images.append(x)
    
    # (resize(imread(IMAGES_FOLDER + str(file_name)), (128, , 3)))

    np_batch_images = np.array(test_images)

    np_batch_images = np_batch_images.astype('float32') / 255.
    
    x_test = np.rollaxis(np_batch_images, 1, 0)[0]
    
    
    encoded_imgs = encoder.predict(x_test)
    
    decoded_imgs = decoder.predict(encoded_imgs)
    
    decoded_imgs = autoencoder.predict(x_test)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i+1)
        plt.imshow(x_test[i].reshape(INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        # display reconstruction
        ax = plt.subplot(2, n, i + n+1)
        plt.imshow(decoded_imgs[i].reshape(INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def encode_all_images(encoder, image_paths):
    
    results = []
    
    for ind, img_path in enumerate(image_paths):
        
        img = image.load_img(IMAGES_FOLDER + img_path, target_size=INPUT_SHAPE)

        img = image.img_to_array(img)
        
        img = np.expand_dims(img, axis=0)
        
        np_image = np.array(img)
    
        np_image = np_image.astype('float32') / 255.
        
        result = encoder.predict(np_image)
        
        results.append(result)
        
        if ind % 500 == 0:
            print(ind)

    with open(OUTPUT_FILE, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        
        writer.writerow(['image', 'encoding'])
        
        result_string = [",".join(item) for item in result.astype(str)]
        
        for img, result in zip(image_paths, results):
            writer.writerow([img, result_string])

def build_model(name):
    if name=='model_b':
        return build_autoencoder_model_b()
    elif name=='model_a':
        return build_autoencoder_model_a()
    elif name=='biffer_filters':
        return build_autoencoder_simple_double_conv()
    elif name=='vgg':
        return build_autoencoder_simple_bigger_filters()
    elif name=='vgg':
        return build_autoencoder_vgg()
    elif name=='simple':
        return build_autoencoder_simple()
    elif name=='model_c':
        return build_autoencoder_model_c()
    elif name=='model_c_r':
        return build_autoencoder_model_c_r()
    else:
        raise Exception("MODEL_NAME not valid")

def encoding():
    pass

if __name__=="__main__":
    
    autoencoder, encoder, decoder = build_model(MODEL_NAME)
    
    print(autoencoder.summary())
    
    '''
    x_train, x_test, y_train, y_test = prepare_images()
    
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    
    history = autoencoder.fit(x_train, x_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                verbose=1,
                validation_data=(x_test, x_test))
    '''
    
    # Using batch generator
    
    image_file_names = [file_name for file_name in os.listdir(IMAGES_FOLDER)]
    
    print(len(image_file_names))
    
    X_train, X_val, y_train, y_val = train_test_split(
        image_file_names, image_file_names, test_size=TEST_SIZE, random_state=42)
    
    training_batch_generator = AutoencoderGenerator(X_train, BATCH_SIZE, INPUT_SHAPE)
    validation_batch_generator = AutoencoderGenerator(X_val, BATCH_SIZE, INPUT_SHAPE)
    
    history = autoencoder.fit_generator(generator=training_batch_generator,
                   steps_per_epoch = int(len(image_file_names) // BATCH_SIZE),
                   epochs = EPOCHS,
                   verbose = 1,
                   validation_data = validation_batch_generator,
                   validation_steps = int(len(image_file_names) // BATCH_SIZE))
    
    visualize(history)
    
    X_val = np.array(X_val)
    idx = np.random.randint(X_val.shape[0], size=(10, ))
    input_imgs = X_val[idx]
    
    print("Input images for visualization shape :", input_imgs.shape)
    
    generate_images_gen(input_imgs, autoencoder)
    
    autoencoder_json = autoencoder.to_json()
    with open("autoencoder_model.json", "w") as json_file:
        json_file.write(autoencoder_json)
    autoencoder.save_weights('autoencoder_weights.h5')
    
    print("Model and weights saved!")
    
    evaluate_gen(autoencoder, encoder, decoder, X_val)
    
    # encode_all_images(encoder, image_file_names)
    
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    