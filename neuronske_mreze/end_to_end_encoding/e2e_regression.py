# -*- coding: utf-8 -*-

import numpy as np
import os
from keras.models import Model, model_from_json
from keras.layers import Input, Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.utils import Sequence
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam
from keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv

IMAGES_FOLDER = '../../scraped_database_tags_new/'
MODEL_NAME = 'model_c3'
INPUT_SHAPE = (256, 256, 3) 
RANDOM_SEED = 100
TEST_SIZE = 0.2
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 0.0001
LOSS_FUNCTION = 'MSE'

class RegressorGenerator(Sequence):
    def __init__(self, image_filenames, scores, batch_size, input_shape):
        self.image_filenames = image_filenames
        self.scores = scores
        self.batch_size = batch_size
        self.input_shape = input_shape
        
    def __len__(self) :
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  
    def __getitem__(self, idx):
        
        batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.scores[idx * self.batch_size : (idx+1) * self.batch_size]
        
        scores = [float(y) for y in batch_y] 
        
        batch_filenames = ([file_name for file_name in batch_x])
        
        batch_images = [] 
        
        for filename in batch_filenames:
            img = image.load_img(IMAGES_FOLDER + filename, target_size=self.input_shape)
        
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            
            batch_images.append(x)
    
        np_batch_images = np.array(batch_images)
    
        np_batch_images = np_batch_images.astype('float32') / 255.
        
        np_batch_images = np.rollaxis(np_batch_images, 1, 0)[0]
        
        return np_batch_images, scores

def visualize(history):
    
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    xc=range(EPOCHS)
    
    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('Epochs')
    plt.ylabel(f'Loss {LOSS_FUNCTION}')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])
    
    plt.style.use(['classic'])

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def load_encoder():
        
    model_json_file = open(f'encoder_model_{MODEL_NAME}.json', 'r')
    loaded_model_json = model_json_file.read()
    model_json_file.close()
    encoder = model_from_json(loaded_model_json)
    encoder.load_weights(f"encoder_weights_{MODEL_NAME}.h5")
    print("Encoder loaded")
    
    return encoder

def build_top(encoder_output):
    
    top_input = Input(shape=encoder_output.output_shape[1:])
    
    model = Flatten()(top_input)
    model = Dense(1024, activation='relu')(model)
    model = Dropout(rate=0.5)(model)
    model = Dense(512, activation='relu')(model)
    model = Dense(1, activation='linear')(model)
    
    regressor = Model(top_input, model, name="top")
    
    return regressor

def read_scores():
    
    with open("../../features_complete_v3.csv", "r") as features_file:
        dictReader = csv.DictReader(features_file)
        
        scores_dict = {}
        for row in dictReader:
            # print(row["id"], " ", row["log_score"])
            scores_dict[row["id"]] = row["log_score"]

    return scores_dict

def build_regression_model():
    
    encoder = load_encoder()
    
    for layer in encoder.layers:
        layer.trainable = False
    
    print(encoder.summary()) 
    
    print(encoder.layers[-1].output_shape[1:])
    
    encoder_output = encoder.layers[-1]
    
    model_top = build_top(encoder_output)
    print(model_top.summary()) 
    
    input_img = Input(shape=INPUT_SHAPE) 
    encoded = encoder(input_img)
    neural_regressor = model_top(encoded)
    full_regressor = Model(input_img, neural_regressor, name="regressor")
    
    print(full_regressor.summary())
    
    lr = 0.0001
    opt = Adam(learning_rate=lr)
    
    full_regressor.compile(optimizer=opt, loss=LOSS_FUNCTION, metrics=['mse', 'mae', coeff_determination])

    return full_regressor


def build_regression_model_one_piece():
    input_img = Input(shape=(256, 256, 3))  

    if MODEL_NAME == 'model_e':
        model = Conv2D(64, (3, 3), activation='relu', padding='same', trainable=False)(input_img)
        model = MaxPooling2D((2, 2), padding='same')(model)
        model = Conv2D(32, (3, 3), activation='relu', padding='same', trainable=False)(model)
        model = MaxPooling2D((2, 2), padding='same')(model)
        model = Conv2D(16, (3, 3), activation='relu', padding='same', trainable=False)(model)
        model = MaxPooling2D((2, 2), padding='same')(model)
    elif MODEL_NAME == 'model_c3':
        x = Conv2D(128, (3, 3), activation='relu', padding='same', trainable=False)(input_img)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', trainable=False)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', trainable=False)(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', trainable=False)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same', trainable=False)(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same', trainable=False)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same', trainable=False)(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same', trainable=False)(x)
        model = MaxPooling2D((2, 2), padding='same')(x)
    else:
        raise
    
    
    model = Flatten()(model)
    model = Dense(2048, activation='relu')(model)
    model = Dropout(rate=0.3)(model)
    model = Dense(1024, activation='relu')(model)
    model = Dense(1, activation='linear')(model)
    
    
    # Manually load weights
    encoder = load_encoder()
    
    regressor = Model(input_img, model, name="regressor")
    
    if MODEL_NAME == 'model_e':
        regressor.layers[1].set_weights(encoder.layers[1].get_weights())
        regressor.layers[3].set_weights(encoder.layers[3].get_weights())
        regressor.layers[5].set_weights(encoder.layers[5].get_weights())
    elif MODEL_NAME == 'model_c3':
        regressor.layers[1].set_weights(encoder.layers[1].get_weights())
        regressor.layers[2].set_weights(encoder.layers[2].get_weights())
        
        regressor.layers[4].set_weights(encoder.layers[4].get_weights())
        regressor.layers[5].set_weights(encoder.layers[5].get_weights())
        
        regressor.layers[7].set_weights(encoder.layers[7].get_weights())
        regressor.layers[8].set_weights(encoder.layers[8].get_weights())
        
        regressor.layers[10].set_weights(encoder.layers[10].get_weights())
        regressor.layers[11].set_weights(encoder.layers[11].get_weights())
    else:
        raise
    
    
    print("Regression model built")
    
    print(regressor.summary())
    
    plot_model(regressor, to_file=f'model_plot_{MODEL_NAME}.png', show_shapes=True)
    
    opt = Adam(learning_rate=LEARNING_RATE)
    
    regressor.compile(optimizer=opt, loss=LOSS_FUNCTION, metrics=['mse', 'mae', coeff_determination])
    
    return regressor 

def neural_regression():
    
    # regression_model = build_regression_model()
    regression_model = build_regression_model_one_piece()
    
    image_file_names = [file_name for file_name in os.listdir(IMAGES_FOLDER)]
    print(f"Images length : {len(image_file_names)}")

    scores_dict = read_scores()
    print("Number of loaded scores :", len(scores_dict))
    
    scores = []
    for fn in image_file_names:
        scores.append(scores_dict[fn[:fn.find(".")]])
        
    print("Scores list length :", len(scores))

    X_train, X_test, y_train, y_test = train_test_split(
        image_file_names, scores, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    
    train_generator = RegressorGenerator(X_train, y_train, BATCH_SIZE, INPUT_SHAPE)
    test_generator = RegressorGenerator(X_test, y_test, BATCH_SIZE, INPUT_SHAPE)
    
    history = regression_model.fit_generator(
                   generator=train_generator,
                   steps_per_epoch = int(len(X_train) // BATCH_SIZE),
                   epochs = EPOCHS,
                   verbose = 1,
                   validation_data = test_generator,
                   validation_steps = int(len(X_test) // BATCH_SIZE),
                   shuffle=True)
    
    visualize(history)
    
    model_json = regression_model.to_json()
    # with open(f"regression_model_{MODEL_NAME}.json", "w") as json_file:
    with open(f"regression_model_one_piece_{MODEL_NAME}.json", "w") as json_file:
        json_file.write(model_json)
    regression_model.save_weights(f'regression_model_weights_one_piece_{MODEL_NAME}.h5')
    
if __name__=="__main__":    
    
    neural_regression()
   