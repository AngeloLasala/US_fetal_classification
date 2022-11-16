"""
Grid search of the best hyperparameters
"""
"""
Main code for Classification task about Plane and Brain_plane
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.model_selection import GridSearchCV

from classification_utils import *

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Main for classification problem of US fetal image')
	parser.add_argument("attribute", type=str, help="Attributo to cliification task: 'Plane' or 'Brain_plane'")
	parser.add_argument("model_name", type=str, help="""Model name: 'MobileNetV2'; 'VGG_16'; 'DenseNet_169""")
	
	args = parser.parse_args()

	## MODEL STRUCTURE
	model_name = args.model_name
	model_par = model_parameter(model_name, learning_rate=0.001, frozen_layers=-1)

	## Model dictianory of parameters
	BACH_SIZE = model_par['BACH_SIZE']
	IMG_SIZE = model_par['IMG_SIZE']
	val_split = model_par['val_split']

	images_path = 'Images_classification_' + args.attribute + '/train/'
	N_img=0
	for clas in os.listdir(images_path):
		n_img = len(os.listdir(images_path+'/'+clas))
		N_img += n_img
	
	train_dataset = image_dataset_from_directory(images_path,
                                             shuffle=True,
                                             batch_size = N_img,
                                             image_size=IMG_SIZE,
											 interpolation='bilinear',
                                            )

	
	##  Prefetch step for optimaze the memory 
	AUTOTUNE = tf.data.experimental.AUTOTUNE
	train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

	## DATA AGUMENTATION - PREPROCESSING
	data_augmentation = data_augmenter()

	## MODEL
	model = classification_model(model_par, data_augmentation, args.attribute)
	base_model = tf.keras.applications.vgg16.VGG16()
	print(model.summary())
	print("Number of layers in the total model: ", len(model.layers))

	## FINE-TUNING
	learning_rate = model_par['learning_rate']
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              	   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
	
	# epochs = model_par['epochs']
	# param_grid = dict(batch_size=batch_size, epochs=epochs)
	param_grid = {'epochs':[1,3,5,7]}
	grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring = "accuracy", n_jobs=-1, cv=3)
	grid_result = grid.fit(train_dataset, verbose=1)

	# ## SAVE MODEL and PARAMETERS FILE
	# classification_path = 'Images_classification_'+ args.attribute
	# models_path = 'Images_classification_' + args.attribute +'/models_from_drive/' + args.model_name + '_'
	# print(models_path)
	# if args.model_name + '_' in os.listdir('Images_classification_' + args.attribute +'/models_from_drive'):
	# 	pass
	# else: smart_makedir(models_path)

	# ## check other model are trined
	# if len(os.listdir(models_path)) == 0 :
	# 	print('CREO IL FOLDER train_1')
	# 	model_folder = models_path + '/train_1'
	# 	smart_makedir(model_folder)
	# else: 
	# 	print('CREO UN NUOVO FOLDER')
	# 	count = len(os.listdir(models_path))
	# 	model_folder = models_path + '/train_' + str(count+1)
	# 	smart_makedir(model_folder)
		
	# # model.save(model_folder + '/' + model_name, save_format='h5')
	# a = [2,3,5]
	# np.save(model_folder + '/a', np.array(a))

	# with open(model_folder + '/' + model_name +'_summary.txt', 'w', encoding='utf-8') as file:
	# 	file.write(f'\n Model Name: {model_name} \n ')
	# 	model.summary(print_fn=lambda x: file.write(x + '\n'))

	# 	for par in model_par.keys():
	# 		file.write(f'\n {par}: {model_par[par]} \n ')