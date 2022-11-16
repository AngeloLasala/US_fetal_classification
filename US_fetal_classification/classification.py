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
	parser.add_argument("-frozen", default=-1, type=int, help="Number of frosez leyers in model_base")
	args = parser.parse_args()
	
	## MODEL STRUCTURE
	model_name = args.model_name
	model_par = model_parameter(model_name, learning_rate=0.0001, epochs=15, frozen_layers=args.frozen)

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

	callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              	   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
	
	epochs = model_par['epochs']
	history = model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs, callbacks=[callback])

	hist_accuracy = [0.] + history.history['accuracy']
	hist_val_accuracy = [0.] + history.history['val_accuracy']
	hist_loss = history.history['loss']
	hist_val_loss = history.history['val_loss']
	epochs_train = len(history.history['loss'])
	
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
	np.save(model_folder + '/history_accuracy', np.array(hist_accuracy))
	np.save(model_folder + '/history_val_accuracy', np.array(hist_val_accuracy))
	np.save(model_folder + '/history_loss', np.array(hist_loss))
	np.save(model_folder + '/history_val_loss', np.array(hist_val_loss))
	np.save(model_folder + '/epoch_train', np.array(len(history.history['loss'])))

	with open(model_folder + '/' + model_name +'_summary.txt', 'w', encoding='utf-8') as file:
		file.write(f'\n Model Name: {model_name} \n ')
		model.summary(print_fn=lambda x: file.write(x + '\n'))

		for par in model_par.keys():
			file.write(f'\n {par}: {model_par[par]} \n ')

	