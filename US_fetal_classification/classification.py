"""
Main code for Classification task about Plane and Brain_plane
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from tensorflow.keras.preprocessing import image_dataset_from_directory

from classification_utils import *

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Main for classification problem of US fetal image')
	parser.add_argument("attribute", type=str, help="Attributo to cliification task: 'Plane' or 'Brain_plane'")
	parser.add_argument("model_name", type=str, help="""Model name: 'MobileNetV2'; 'VGG_16'; 'DenseNet_169""")
	
	args = parser.parse_args()

	## MODEL STRUCTURE
	model_name = args.model_name
	model_par = model_parameter(model_name)

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
	model = classification_model(model_par, data_augmentation, args.attribute)
	print(model.summary())

	## TRAINING
	learning_rate = model_par['learning_rate']
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              	   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
	
	epochs = model_par['epochs']
	history = model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs)

	## SAVE MODEL and PARAMETERS FILE
	classification_path = 'Images_classification_'+ args.attribute
	models_path = 'Images_classification_' + args.attribute +'/models/' + args.model_name + '_'

	if args.model_name + '_' in os.listdir('Images_classification_' + args.attribute +'/models'):
		pass
	else: smart_makedir(models_path, trial=True)

	## check other model are trined
	if len(os.listdir(models_path)) == 0 :
		model_folder = models_path + '/train_1'
		smart_makedir(model_folder)
	else: 
		count = len(os.listdir(models_path))
		model_folder = models_path + '/train_' + str(count+1)
		smart_makedir(model_folder)
		
	model.save(model_folder + '/' + model_name, save_format='h5')

	with open(model_folder + '/' + model_name +'_summary.txt', 'w', encoding='utf-8') as file:
		file.write(f'\n Model Name: {model_name} \n ')
		model.summary(print_fn=lambda x: file.write(x + '\n'))


	