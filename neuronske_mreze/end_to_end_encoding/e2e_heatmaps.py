# -*- coding: utf-8 -*-

import numpy as np

from matplotlib import pyplot as plt
import matplotlib.cm as cm

from keras.preprocessing.image import img_to_array
from keras.models import model_from_json

from vis.utils import utils
from vis.visualization import overlay, visualize_cam


SAMPLES = ['a2WE3Ae.jpg', 'a2WEK2O.jpg', 'aBm9ovD.jpg', 'aGdKmX5.jpg',
           'a7WvGXr.jpg', 'aEPdgdn.jpg', 'aQd6Wow.jpg',
           'aj9QeBQ.jpg', 'aj9QpEQ.jpg', 
           'aeDKewO.jpg',
           ]
IMAGES_FOLDER = '../../scraped_database_tags_new/'

MODEL_NAME = 'model_c3'
IMAGE_SIZE = (256, 256)


def load_regression_model():
    
    model_json_file = open(f'regression_model_one_piece_{MODEL_NAME}.json', 'r')
    loaded_model_json = model_json_file.read()
    model_json_file.close()
    regression_model = model_from_json(loaded_model_json)
    regression_model.load_weights(f"regression_model_weights_one_piece_{MODEL_NAME}.h5")
    print("Regression model loaded")
    
    return regression_model


def load_image(img_name):
    
    img = utils.load_img(f'../../scraped_database_tags_new/{img_name}', target_size=IMAGE_SIZE)
    # plt.imshow(img)
    
    bgr_img = utils.bgr2rgb(img)
    
    return img, bgr_img


def visualize_attention_maps():
    
    model = load_regression_model()
    print(model.summary())
    plt.figure(figsize=(20, 4))
    for ind, img_name in enumerate(SAMPLES):    
        img, bgr_img = load_image(img_name)
       
        img_input = np.expand_dims(img_to_array(img), axis=0)
        pred = model.predict(img_input)[0][0]
        print('Predicted {}'.format(pred))
        
        
        heatmap = visualize_cam(model, layer_idx=-1, filter_indices=0, seed_input=bgr_img, grad_modifier='small_values')
        jet_heatmap = np.uint8(cm.jet(heatmap)[..., :3] * 255)

        ax = plt.subplot(2, len(SAMPLES), ind+1)
        plt.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        ax = plt.subplot(2, len(SAMPLES), ind + len(SAMPLES) + 1)
        plt.imshow(overlay(img, jet_heatmap, alpha=0.3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
    # plt.show()
    plt.savefig('heatmaps_c3.png', bbox_inches='tight')
    
    
if __name__=="__main__":
    
    visualize_attention_maps()
