"""
Main code for Classification task about Plane and Brain_plane
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation

from classification_utils import data_augmenter, plane_model

if __name__ == '__main__':

	## LOAD TRAIN SET - preprocessing parameters
	BACH_SIZE = 32
	IMG_SIZE = (224,224)
	val_split = 0.1

	images_path = 'Images_classification_Plane/train/'

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
	model2 = plane_model(IMG_SIZE, data_augmentation)

	## TRAINING
	learning_rate = 0.01
	model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              	   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
	
	
	initial_epochs = 5
	history = model2.fit(train_dataset, validation_data=validation_dataset, epochs=initial_epochs)
	