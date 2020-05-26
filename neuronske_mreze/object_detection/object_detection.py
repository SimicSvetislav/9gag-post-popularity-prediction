# -*- coding: utf-8 -*-

import os
import csv
import copy
import urllib
import tarfile
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

MODEL_DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
MODEL_NAME = 'ssd_mobilenet_v2_oid_v4_2018_12_12'
MODEL_FILE = MODEL_NAME + '.tar.gz'

DESCRIPTIONS_FILE = 'class-descriptions-boxable.csv'
OID_DOWNLOAD_BASE = 'https://storage.googleapis.com/openimages/2018_04/'

FROZEN_GRAPH_FILE = 'frozen_inference_graph.pb'
PATH_TO_FROZEN_GRAPH = os.path.join(MODEL_NAME, FROZEN_GRAPH_FILE)

IMAGES_FOLDER = '../../scraped_database_tags_new'

IMAGE_TENSOR_KEY = 'image_tensor'

DETECTION_BOXES_KEY = 'detection_boxes'
DETECTION_SCORES_KEY = 'detection_scores'
DETECTION_CLASSES_KEY = 'detection_classes'

TENSOR_SUFFIX = ':0'

DETECTIONS_FILE = 'detections.csv'
MAPPED_DETECTIONS_FILE = 'detection_tags_new.csv'

def model_bootstrap():
    if os.path.exists(MODEL_FILE) is False:
        opener = urllib.request.URLopener()
        opener.retrieve(MODEL_DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    
    if os.path.exists(MODEL_NAME) is False:
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
            filename = os.path.basename(file.name)
            if FROZEN_GRAPH_FILE in filename:
                tar_file.extract(file, os.getcwd())

def dataset_bootstrap():
    if os.path.exists(DESCRIPTIONS_FILE) is False:
        opener = urllib.request.URLopener()
        opener.retrieve(OID_DOWNLOAD_BASE + DESCRIPTIONS_FILE, DESCRIPTIONS_FILE)


def prepare():
    model_bootstrap()
    dataset_bootstrap()

def read_description():
    df = pd.read_csv(DESCRIPTIONS_FILE, names=['id', 'class'])
    
    category_index = {}
    embedding_dict = {}
    
    for idx, row in df.iterrows():
        category_index[idx+1] = {'id': row['id'], 'name': row['class']}
        embedding_dict[row['class']] = 0
        
    return category_index, embedding_dict
        
def load_image_into_numpy_array(image):
    (width, height) = image.size
    return np.array(image.getdata()).reshape((height, width, 3)).astype(np.uint8)


def run_inference(graph, image_np):
    
    output_tensor_dict = {
        DETECTION_BOXES_KEY: DETECTION_BOXES_KEY + TENSOR_SUFFIX,
        DETECTION_SCORES_KEY: DETECTION_SCORES_KEY + TENSOR_SUFFIX,
        DETECTION_CLASSES_KEY: DETECTION_CLASSES_KEY + TENSOR_SUFFIX
    }

    with graph.as_default():
        with tf.Session() as sess:
            input_tensor = tf.get_default_graph().get_tensor_by_name(IMAGE_TENSOR_KEY + TENSOR_SUFFIX)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            input_tensor_dict = {input_tensor: image_np_expanded}
            output_dict = sess.run(output_tensor_dict, feed_dict=input_tensor_dict)

            return {
                DETECTION_BOXES_KEY: 
                    output_dict[DETECTION_BOXES_KEY][0],
                DETECTION_SCORES_KEY: 
                    output_dict[DETECTION_SCORES_KEY][0],
                DETECTION_CLASSES_KEY: 
                    output_dict[DETECTION_CLASSES_KEY][0].astype(np.int64)
            }
                                
def process_output(classes, scores, boxes, category_index):

    results = []

    for clazz, score, box in zip(classes, scores, boxes):
        if score > 0.0:
            label = category_index[clazz]['name']
            obj_result = type('result', (object,),{'label': label, 
                                            'score': score,
                                            'box': box}) ()
            results.append(obj_result)
    
    return results


def detect_objects(filename, graph, category_index, embedding_dict):
    
    print('Detection on image', filename)
    
    image_np = load_image_into_numpy_array(Image.open(filename))
    
    output_dict = run_inference(graph, image_np)

    results = process_output(output_dict[DETECTION_CLASSES_KEY],
                             output_dict[DETECTION_SCORES_KEY],
                             output_dict[DETECTION_BOXES_KEY],
                             category_index)
    
    help_dict = copy.deepcopy(embedding_dict)

    objects_set = set()

    for item in results:
        if item.score > 0.4:
            objects_set.add(item.label)
            help_dict[item.label] += 1

    with open(DETECTIONS_FILE, 'a', newline='') as detections_file:
        writer = csv.writer(detections_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        feature_vector = str(help_dict.values())[13:-2].replace(", ", "")
        writer.writerow([filename, feature_vector])

    for set_item in objects_set:
        print(set_item, ":", help_dict[set_item])


def load_graph():
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='')

    return graph


PERSON =             0b000000000000001
PEOPLE =             0b000000000000010
CAT =                0b000000000000100
DOG =                0b000000000001000
OTHER_ANIMAL =       0b000000000010000
POSTER =             0b000000000100000
CLOTHING =           0b000000001000000
CAR =                0b000000010000000
TOY =                0b000000100000000
TREE =               0b000001000000000
GLASSES =            0b000010000000000
BUILDING =           0b000100000000000
ELECTRONIC_DEVICE =  0b001000000000000
AIRPLANE =           0b010000000000000
GUITAR =             0b100000000000000

PERSON_NEGATE = 0b111111111111110


def mapping(category_index):
    
    with open(DETECTIONS_FILE, 'r', newline='') as detections_file, open(MAPPED_DETECTIONS_FILE, 'w', newline='') as mapped_detections_file:
        
        lines = list(csv.reader(detections_file))
        writer = csv.writer(mapped_detections_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL) 
        
        for x in range(1, len(lines)):
            
            feature_vector = lines[x][1]
            
            image_path = lines[x][0]
            
            post_id = image_path[image_path.rfind('/') + 1 : image_path.rfind('.jpg')]

            
            person = False
            people = False
            human_faces = 0
            
            nfv = 0b000000000000000
            
            for ind, val in enumerate(feature_vector):
                
                if int(val)!=0:
                    category = category_index[ind+1]['name']
                    
                    if (not people) and (category == "Person" or category == "Man" or category == "Woman" or category == "Boy" or category == "Girl"):
                            
                        if person:
                            nfv = nfv & PERSON_NEGATE
                            nfv = nfv | PEOPLE
                            people = True
                        elif int(val) > 1:
                            nfv = nfv & PERSON_NEGATE
                            nfv = nfv | PEOPLE
                            people = True
                        else:
                            nfv = nfv | PERSON
                        
                        person = True
                    
                    elif (not people) and category == "Human face":
                        
                        human_faces += int(val)
                        
                        if int(val) == 1 and not person:
                            nfv = nfv | PERSON
                        
                        if human_faces > 1:
                            nfv = nfv & PERSON_NEGATE
                            nfv = nfv | PEOPLE
                            people = True
                    
                    elif category == "Cat":
                        nfv = nfv | CAT
                        
                    elif category == "Dog":
                        nfv = nfv | DOG
                    
                    elif category == "Bird" or category == "Monkey" or category == "Horse" or category == "Animal" or category == "Penguin" or category == "Duck" or category == "Tiger" or category == "Cattle" or category == "Rabbit" or category == "Chicken":
                        nfv = nfv | OTHER_ANIMAL
                    
                    elif category == "Poster":
                        nfv = nfv | POSTER
                        
                    elif category == "Clothing" or category == "Suit" or category == "Dress" or category == "Jeans":
                        nfv = nfv | CLOTHING
                        
                    elif category == "Car":
                        nfv = nfv | CAR
                        
                    elif category == "Toy":
                        nfv = nfv | TOY
                        
                    elif category == "Tree":
                        nfv = nfv | TREE
                    
                    elif category == "Glasses" or category == "Sunglasses":
                        nfv = nfv | GLASSES
                    
                    elif category == "House" or category == "Building" or category == "Window":
                        nfv = nfv | BUILDING
                        
                    elif category == "Mobile phone" or category == "Computer monitor" or category == "Computer keyboard" or category == "Laptop" or category == 'Television':
                        nfv = nfv | ELECTRONIC_DEVICE
                    
                    elif category == "Airplane":
                        nfv = nfv | AIRPLANE
                    
                    elif category == "Guitar":
                        nfv = nfv | GUITAR
                    
                    # print(category_index[ind+1][NAME_KEY], ":", v)
            
            print(post_id, ":", format(nfv, '#017b'))
            writer.writerow([post_id, format(nfv, '#017b')])
            
            if nfv & PERSON & PEOPLE != 0:
                raise


def complete_pipeline():
    prepare()
    
    category_index, embedding_dict = read_description()
    
    graph = load_graph()
    
    # limit = 100
    
    for ind, img in enumerate(os.listdir(IMAGES_FOLDER)):
        
        if ind%100==0:
            print("Checkpoint on", ind)
            
        # if limit <= ind:
        #    break
        
        detect_objects(IMAGES_FOLDER + "/" + img, graph, category_index, embedding_dict)
    
    mapping(category_index)
    
    
if __name__=="__main__":
    complete_pipeline()