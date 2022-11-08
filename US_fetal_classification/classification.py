"""
Main code for Classification task about Plane and Brain_plane
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation

from classification_utils import *

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Main for classification problem of US fetal image')
	parser.add_argument("attribute", type=str, help="Attributo to cliification task: 'Plane' or 'Brain_plane'")
	parser.add_argument("model_name", type=str, help="""Model name: 'MobileNetV2'""")
	
	args = parser.parse_args()

	## MODEL STRUCTURE
	model_name = args.model_name
	model_par = model_parameter(model_name, epochs=1)

	## Model dictianory of parameters
	BACH_SIZE = model_par['BACH_SIZE']
	IMG_SIZE = model_par['IMG_SIZE']
	val_split = model_par['val_split']

	images_path = 'Images_classification_'+args.attribute + '/train/'

	train_dataset = image_dataset_from_directory(images_path,
                                             shuffle=True,
                                             batch_size = BACH_SIZE,
                                             image_size=IMG_SIZE,
											 interpolation='bilinear',
                                             validation_split=val_split,
                                             subset='training',
                                             seed=42)

	validation_dataset = image_dataset_from_directory(images_path,
                                             shuffle=True,
                                             batch_size = BACH_SIZE,
                                             image_size=IMG_SIZE,
											 interpolation='bilinear',
                                             validation_split=val_split,
                                             subset='validation',
                                             seed=42)

	##  Prefetch step for optimaze the memory 
	AUTOTUNE = tf.data.experimental.AUTOTUNE
	train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

	## DATA AGUMENTATION - PREPROCESSING
	data_augmentation = data_augmenter()

	## MODEL
	model = classification_model(model_par, data_augmentation)

	## TRAINING
	learning_rate = model_par['learning_rate']
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              	   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
	
	epochs = model_par['epochs']
	history = model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs)

	## SAVE MODEL and PARAMETERS FILE
	classification_path = 'Images_classification_'+ args.attribute
	models_path = 'Images_classification_'+ args.attribute + '/models'
	model.save(models_path + '/' + model_name, save_format='h5')

	with open(models_path + '/' + model_name +'_summary.txt', 'w', encoding='utf-8') as file:
		file.write(f'\n Model Name: {model_name} \n ')
		model.summary(print_fn=lambda x: file.write(x + '\n'))

		for par in model_par.keys():
			file.write(f'\n {par}: {model_par[par]} \n ')


	