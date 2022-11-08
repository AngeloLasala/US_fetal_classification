"""
Support module for classification task. 
Functions for the models and preprocessing.
"""
import os
import numpy as np 
import pandas as pd 
from PIL import Image
import imageio
from makedir import *
# from image_utils import *
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, RandomTranslation, RandomCrop


def dataset_folder(data, attribute, 
				   name_folder = 'Images_classification',
				   images_path = 'FETAL_PLANES_ZENODO/Images'
				   ):
	"""
	Create a folder ready to the classification task based on classes of selected attributes
	
	Parameters
	----------
	"""
	
	classes = data[attribute].unique()
	for clas in classes:
		smart_makedir(name_folder + '_' + attribute +'/train' + '/'+ clas)
		smart_makedir(name_folder + '_' + attribute +'/test' + '/'+ clas)

	train_index = data[data['Train ']==1].index
	test_index = data[data['Train ']==0].index

		
	for index in range(data.shape[0]):
		print(index)
		image_path = images_path +'/'+ data.iloc[index]['Image_name']+'.png'
		img_class = data.iloc[index][attribute]
		img = Image.open(image_path)
		if index in train_index:
			img.save(name_folder + '_' + attribute +'/train'+'/'+img_class+'/'+ data.iloc[index]['Image_name']+'.png')
		if index in test_index:
			img.save(name_folder + '_' + attribute +'/test'+'/'+img_class+'/'+ data.iloc[index]['Image_name']+'.png')

def data_augmenter():
    """
    Create a Sequential model composed of 4 layers

    Returns
	------- 
	data_augumentation: tf.keras.Sequential
    """
    data_augmentation = tf.keras.Sequential()
    data_augmentation.add(RandomFlip())
    data_augmentation.add(RandomRotation(0.05)) # 15 degrees
    data_augmentation.add(RandomTranslation(height_factor=(-0.05, 0.05), width_factor=(-0.05, 0.05))) # 10 pixel
    # data_augmentation.add(RandomCrop(224-int(0.1*224),224-int(0.1*224)))
    
    return data_augmentation

def preprocess_input_model():
	"""
	Preprocessing for selected pretrined model

	UPDATE: argument = selected model
	"""
	return tf.keras.applications.resnet.preprocess_input

def plane_model(image_shape, data_augmentation):
    """
	Classification model for Plane attribute.
	Last layer is Dense(6)
    """
    
    
    input_shape = image_shape + (3,)
    
    base_model = tf.keras.applications.resnet50.ResNet50(input_shape=input_shape,
                                                   include_top=False, # <== Important!!!!
                                                   weights='imagenet') # From imageNet
    
    # Freeze the base model by making it non trainable
    base_model.trainable = False
    
    # create the input layer (Same as the imageNetv2 input size)
    inputs = tf.keras.Input(shape=input_shape) 
    
    # apply data augmentation to the inputs
    x = data_augmentation(inputs)
    
    # data preprocessing using the same weights the model was trained on
    # Already Done -> preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    preprocess_input = tf.keras.applications.resnet.preprocess_input	
    x = preprocess_input(x) 
    
    # set training to False to avoid keeping track of statistics in the batch norm layer
    x = base_model(x, training=False) 
    
    # Add the new Binary classification layers
    # use global avg pooling to summarize the info in each channel
    x = tfl.GlobalAveragePooling2D()(x) 
    #include dropout with probability of 0.2 to avoid overfitting
    x = tfl.Dropout(0.2)(x)
        
    # create a prediction layer with one neuron (as a classifier only needs one)
    prediction_layer = tfl.Dense(6)
    
    outputs = prediction_layer(x) 
    model = tf.keras.Model(inputs, outputs)
    
    return model