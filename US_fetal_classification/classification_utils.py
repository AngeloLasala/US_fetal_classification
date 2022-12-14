"""
Support module for classification task. 
Functions for the models and preprocessing.
"""
import os
import numpy as np 
import pandas as pd 
from PIL import Image
from sklearn.metrics import classification_report
from makedir import *
from image_utils import smart_plot, get_img_array, normalization

import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, RandomTranslation, RandomCrop

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
                    retraining = True,
                    frozen_layers = -1,
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

    frozen_layers : integer
        number of frozen leyers stating for the lower level leyers.
        default = -1 means all the layer of base_model are frozen (equal to retraininf = False)
        frozen_layers = 0 means all the parameters are re_trained 

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
    model_dict['retraining'] = retraining
    model_dict['learning_rate'] = learning_rate
    model_dict['frozen_layers'] = frozen_layers
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
    base_model.trainable = model_dict['retraining'] 
    if model_dict['retraining']:
        print("Number of layers in the base model: ", len(base_model.layers))
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:model_dict['frozen_layers']]:
            print('Layer ' + layer.name + ' frozen.')
            layer.trainable = False
    
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
    
    prediction_layer = tfl.Dense(class_nodes, activation='softmax')
    
    outputs = prediction_layer(x) 
    model = tf.keras.Model(inputs, outputs)
    
    return model

def true_labels(test_path, test_data):
    """
    True label of slected path for classification

    Parameters
    ----------
    test_path : string
        test's path for classification problem

    test_data : tf.Dataset
        dataset from test path

    Returns
    -------
    true_label : list
        list of true labels
    """

    count = 0
    for direct in os.listdir(test_path):
        count += len(os.listdir(test_path + '/' + direct))
    
    true_lab = []
    for images, labels in test_data.take(count):
        # print(labels)
        true_lab.append(labels[0].numpy())

    return true_lab

def evaluation_model(model, test_set):
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

def statistical_analysis(y_test, y_pred, test_dataset):
    """
    Statistical analysis of trained model. For each classes compute
    the precision, recall and F1-score

    Parameters
    ----------
    y_test : array,
        true labels of test set
    
    y_pred : array,
        predicted labels from model.predict()

    test_dataset : tf.Dataset
        test dataset

    Returns
    -------
    task_report : dict
        dictionary of statistical metrics

    """
    classes = test_dataset.class_names
    # from sklearn.metrics import classification_report
    print('\nClassification Report\n')
    print(classification_report(y_test, y_pred, target_names=classes))
    task_report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)

    return task_report

def make_class_weight(count_dir):
    """
    Return the class weight for unbalanced data to sutable 
    train a model. w_i = (n_sample / n_classes) * (1/n_i)

    Parameters
    ----------
    count_dir : directory
        dir with number of sample of each class. keys=class value=n_i

    Return
    ------
    class_weights : directory
        dir with class weights
    """
    keys_class = count_dir.keys()
    n_sample = 0
    for k in keys_class:
        n_sample += count_dir[k] 
	
    class_weight = {}
    for k in keys_class:
        class_weight[k] = (n_sample/len(keys_class)) * (1/count_dir[k]) 
	
    return class_weight

## CAM SECTION ################################################################################
def cam_heatmap(preds, class_weights, last_conv_layer_output):
    """
    Heatmap of Class Activation Map regarding the predicted class

    Parameters
    ----------
    preds: numpy or tensorflow tensor
        array of predictions

    class_weight: numpy or tensorflow tensor
        class weight of the moste rate class prediction. From load_model():
        class_weights = model.layers[-1].get_weights()[0]

    last_conv_layer_output: tensor
        convolutional layer used for the heatmap

    Return
    ------
    heatmap : numpy array
        normalized version of calss activation map in range 0-1, The shape
        depends on the last convulution layer of the model

    """
    predicted_class = tf.argmax(preds[0])
    class_weights = class_weights[:,predicted_class]
    class_weights = np.expand_dims(class_weights, axis=-1)

    last_conv_layer_output = last_conv_layer_output[0,:,:,:]

    heatmap = last_conv_layer_output @ class_weights
    heatmap = np.squeeze(heatmap)
    heatmap = normalization(heatmap, 0, 1)
    return heatmap

def save_cam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4, save_it=True):
    """
    Save the CAM for selectec image. See cam_easy.py

    Parameter
    ---------
    img_path : string
        original image path

    heatmap: numpy array
        class activation map

    cam_pat : string
        directory used to save

    alpha : float
        shadowing of heatmap

    save_it : bool
        actual save the cam. default=True

    Return
    ------
    jet_heatmap : tensor
        calss activation maps of input image
    
    img : tensor
        imput image
    """
    # Load the original image
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    if save_it : superimposed_img.save(cam_path)
    return jet_heatmap, img

def save_array_as_image(array, path, mode=None, format_img='.png'):
    """
    Save the input array in a selecsted format in the selected path
    
    Parameters
    ----------
    array: numpy array
        image array

    path: string
        path's folder when save the image
    
    mode: string
        mode og image, i.e. 'RGB'. Default = None
    format_img : string (optional)
        format of image. Default='.png'
    """
    img = Image.fromarray(array.astype(np.uint8), mode = mode)
    img.save(path + format_img)

