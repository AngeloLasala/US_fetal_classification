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

    if attribute == 'Brain_plane':
        df_train = data[data['Train ']==1]
        df_test = data[data['Train ']==0]

        df_train_brain = df_train[df_train['Plane']=='Fetal brain']
        df_test_brain = df_test[df_test['Plane']=='Fetal brain']

        df_train_brain_classification = df_train_brain[df_train_brain['Brain_plane']!='Other']
        df_test_brain_classification = df_test_brain[df_test_brain['Brain_plane']!='Other']
        
        classes =  df_train_brain_classification[attribute].unique()

        for clas in classes:
            smart_makedir(name_folder + '_' + attribute +'/train' + '/'+ clas)
            smart_makedir(name_folder + '_' + attribute +'/test' + '/'+ clas)

        print('MAKE TRAIN FOLDER')
        for index in df_train_brain_classification.index:
            print(index)
            image_path = images_path +'/'+ data.iloc[index]['Image_name']+'.png'
            img_class = data.iloc[index][attribute]
            img = Image.open(image_path)
            img.save(name_folder + '_' + attribute +'/train'+'/'+img_class+'/'+ data.iloc[index]['Image_name']+'.png')

        smart_makedir(name_folder + '_' + attribute +'/train' + '/'+ 'Not A Brain')
        data_not_brain = data[data['Brain_plane']=='Not A Brain']
        data_not_brain = data_not_brain.sample(n=500)

        print('MAKE NOT A BRAIN FOLDER')
        for index in data_not_brain.index:
            print(index)
            image_path = images_path +'/'+ data.iloc[index]['Image_name']+'.png'
            img_class = data.iloc[index][attribute]
            img = Image.open(image_path)
            img.save(name_folder + '_' + attribute +'/train'+'/'+'Not A Brain'+'/'+ data.iloc[index]['Image_name']+'.png')

        print('MAKE TEST FOLDER')
        for index in df_test_brain_classification.index:
            print(index)
            image_path = images_path +'/'+ data.iloc[index]['Image_name']+'.png'
            img_class = data.iloc[index][attribute]
            img = Image.open(image_path)
            img.save(name_folder + '_' + attribute +'/test'+'/'+img_class+'/'+ data.iloc[index]['Image_name']+'.png')

    else:
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

def model_parameter(model_name,
                    BACH_SIZE = 32,
	                IMG_SIZE = (224,224),
	                val_split = 0.1,
                    imagenet_pretrain = True,
                    learning_rate = 0.001,
                    epochs = 15):
    """
    Return the dictionary of model's parameters for preprocessing and learning

    Parameters
    ----------
    model_name : string
        name of model's architecture, the possible choise are:
        'MobileNetV2' = Mobile Network form tensorflow
        'VGG_16' = VGG-16 form tensorflow
        'DenseNet_169' = .....

    Return
    ------
    model_dict : dictionary
        model architecture parameters dictionary
    """
    model_dict = {}

    ## Load image parameters
    model_dict['BACH_SIZE'] = BACH_SIZE
    model_dict['IMG_SIZE'] = IMG_SIZE
    model_dict['val_split'] = val_split

    ## Preprocessing 
    if model_name == 'MobileNetV2':
        model_dict['preprocessing'] = tf.keras.applications.mobilenet_v2.preprocess_input
        model_dict['base_model'] = tf.keras.applications.mobilenet_v2.MobileNetV2

    if model_name == 'VGG_16':
        model_dict['preprocessing'] = tf.keras.applications.vgg16.preprocess_input
        model_dict['base_model'] = tf.keras.applications.vgg16.VGG16

    if model_name == 'DenseNet_169':
        model_dict['preprocessing'] = tf.keras.applications.densenet.preprocess_input
        model_dict['base_model'] = tf.keras.applications.densenet.DenseNet169

    ## Training hyperparameters
    model_dict['imagenet_pretrain'] = imagenet_pretrain
    model_dict['learning_rate'] = learning_rate
    model_dict['epochs'] = epochs
    return model_dict

def preprocess_input_model():
	"""
	Preprocessing for selected pretrined model

	UPDATE: argument = selected model
	"""
	return tf.keras.applications.resnet.preprocess_input

def classification_model(model_dict, data_augmentation, attribute):
    """
	Classification model for Plane attribute.
	Last layer is Dense(6)

    Parameters
    ----------
    model_dict : dictionary
        dictionary with the model's parameters. See model_parameters function

    data_augumentation : function
        Sequential layer of data augumentation. See data_augmenter() function

    attribute : string
        Attribute to classification. 'Plane' or 'Brain_plane'

    Return
    ------
    model : tensorflow sequential model
    """
    
    
    input_shape = model_dict['IMG_SIZE']+ (3,)
    

    base_model = model_dict['base_model'](input_shape=input_shape,
                                                   include_top=False, # <== Important!!!!
                                                   weights='imagenet') # From imageNet
  
    
    # Freeze the base model by making it non trainable
    base_model.trainable = model_dict['imagenet_pretrain'] 
    
    # create the input layer (Same as the imageNetv2 input size)
    inputs = tf.keras.Input(shape=input_shape) 
    
    # apply data augmentation to the inputs
    x = data_augmentation(inputs)
    
    # data preprocessing using the same weights the model was trained on
    # Already Done -> preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    preprocess_input = model_dict['preprocessing']
    x = preprocess_input(x) 
    
    # set training to False to avoid keeping track of statistics in the batch norm layer
    x = base_model(x, training=False) 
    
    # Add the new Binary classification layers
    # use global avg pooling to summarize the info in each channel
    x = tfl.GlobalAveragePooling2D()(x) 
    #include dropout with probability of 0.2 to avoid overfitting
    x = tfl.Dropout(0.2)(x)
        
    # create a prediction layer with 6 neuron 
    if attribute == 'Plane': class_nodes = 6
    if attribute == 'Brain_plane': class_nodes = 4
    
    prediction_layer = tfl.Dense(class_nodes, activation='sigmoid')
    
    outputs = prediction_layer(x) 
    model = tf.keras.Model(inputs, outputs)
    
    return model

def evlatuation_model(model, test_set):
    """
    Evaluate the model on the teat set and return the main 
    prediction estimation

    Parameters
    ----------
    model : tensorflow model
        trained model on the train set

    test_set: tensorflow.Dataset
        test set

    Returns
    -------
    """

    evaluation = model.eval(test_set)
    prediction = model.predict(test_set)

    return evaluation, prediction

def statistical_analysis(model, test_set):
    """
    Statistical analysis of trained model
    """