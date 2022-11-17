import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

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

	train_images = np.concatenate(list(train_dataset.map(lambda x, y:x)))
	train_labels = np.concatenate(list(train_dataset.map(lambda x, y:y)))

	val_images = np.concatenate(list(validation_dataset.map(lambda x, y:x)))
	val_labels = np.concatenate(list(validation_dataset.map(lambda x, y:y)))

	inputs = np.concatenate((train_images, val_images), axis=0)
	targets = np.concatenate((train_labels, val_labels), axis=0)
	
    ## K-fold CROSS VALIDATION
	kfold = KFold(n_splits=3, shuffle=True)

	for train, test in kfold.split(inputs, targets):
  
		model = tf.keras.Sequential([
		tf.keras.layers.Rescaling(1./255, input_shape=(180, 180, 3)),
		tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
		tf.keras.layers.MaxPooling2D(),
		tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
		tf.keras.layers.MaxPooling2D(),
		tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
		tf.keras.layers.MaxPooling2D(),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(128, activation='relu'),
		tf.keras.layers.Dense(5)])

		model.compile(optimizer='adam',
					loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
					metrics=['accuracy'])
		history = model.fit(inputs[train], targets[train],
					batch_size=batch_size,
					epochs=2)
		scores = model.evaluate(inputs[test], targets[test], verbose=0)