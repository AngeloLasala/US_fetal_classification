import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

from classification_utils import *

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Main for classification problem of US fetal image')
	parser.add_argument("attribute", type=str, help="Attributo to cliification task: 'Plane' or 'Brain_plane'")
	parser.add_argument("model_name", type=str, help="""Model name: 'MobileNetV2'; 'VGG_16'; 'DenseNet_169""")
	
	args = parser.parse_args()

	## MODEL STRUCTURE
	model_name = args.model_name

	model_par = model_parameter(model_name, learning_rate=0.0001, epochs=15, frozen_layers=5)

	## Model dictianory of parameters
	BACH_SIZE = model_par['BACH_SIZE']
	IMG_SIZE = model_par['IMG_SIZE']
	val_split = model_par['val_split']
	
	images_path = 'Images_classification_' + args.attribute + '/train/'
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
	print('INIZIO A FARE I NUMPY ARRAY')
	train_images = np.concatenate(list(train_dataset.map(lambda x, y:x)))
	train_labels = np.concatenate(list(train_dataset.map(lambda x, y:y)))

	val_images = np.concatenate(list(validation_dataset.map(lambda x, y:x)))
	val_labels = np.concatenate(list(validation_dataset.map(lambda x, y:y)))

	inputs = np.concatenate((train_images, val_images), axis=0)
	targets = np.concatenate((train_labels, val_labels), axis=0)
	print('FINE!!!!!\n\n\n\n')

	## MAKE FOLDER
	models_path = 'Images_classification_' + args.attribute + '/CrossValidation/' + args.model_name + '_' 
	if args.model_name + '_' in os.listdir('Images_classification_' + args.attribute +'/CrossValidation' ):
		pass
	else: smart_makedir(models_path)

	## check other model are trined
	if len(os.listdir(models_path)) == 0 :
		model_folder = models_path + '/cv_1'
		smart_makedir(model_folder)
	else: 
		count = len(os.listdir(models_path))
		model_folder = models_path + '/cv_' + str(count+1)
		smart_makedir(model_folder)
	
	## K-fold CROSS VALIDATION
	kfold = KFold(n_splits=3, shuffle=True)
	lr_list = np.logspace(-4,-3, 5)
	accuracy_list = []

	for lr in lr_list:
		lr_accuracy = []
		for train, test in kfold.split(inputs, targets):
			##  Prefetch step for optimaze the memory 
			AUTOTUNE = tf.data.experimental.AUTOTUNE
			train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

			## DATA AGUMENTATION - PREPROCESSING
			data_augmentation = data_augmenter()

			## MODEL
			model_par = model_parameter(model_name, learning_rate=lr, epochs=15, frozen_layers=0)
			model = classification_model(model_par, data_augmentation, args.attribute)
			print(model.summary())

			## TRAINING
			learning_rate = model_par['learning_rate']

			model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
						loss='sparse_categorical_crossentropy',
						metrics=['accuracy'])
			
			epochs = model_par['epochs']
			history = model.fit(inputs[train], targets[train], epochs=epochs)

			## EVALUATION			
			scores = model.evaluate(inputs[test], targets[test], verbose=1)
			lr_accuracy.append(scores[1])
			# prediction = model.predict(inputs[test], verbose=1)
			# predicted_labels = np.argmax(prediction, axis=-1)
			# classification_report(targets[test], predicted_labels, output_dict=True)
		
		lr_accuracy = np.array(lr_accuracy)
		accuracy_list.append(lr_accuracy.mean())
	
	print(accuracy_list)
	np.save(model_folder + '/accuracy', np.array(accuracy_list))
	np.save(model_folder + '/learning_rate', lr_list)

	