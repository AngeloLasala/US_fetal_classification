"""
Pix2Pix file for US fetal brain
"""
import os
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

from gan_utils import *
from image_utils import normalization

@tf.function()
def random_jitter(input_image, real_image):
	"""
	Complete image preprocessing for GAN

	Parameters
	----------
	input_image : tensorflow tensor
		input imgage, i.e. CAM 

	real_image : tensorflow tensor
		real image, i.e. US image

	Returns
	-------
	input_image : tensorflow tensor
		cropped CAM 

	real_image : tensorflow tensor
		cropped US image
	"""
	# Resizing to 286x286
	input_image, real_image = resize(input_image, real_image, 286, 286)

	# Random cropping back to 256x256
	input_image, real_image = random_crop(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)

	if tf.random.uniform(()) > 0.5:
		# Random mirroring
		input_image = tf.image.flip_left_right(input_image)
		real_image = tf.image.flip_left_right(real_image)

	return input_image, real_image

def load_image_train(sample_path):
	"""
	Load and preproces train_file

	Parameters
	----------
	image_file : string
		image's path

	Returns
	-------
	input_image : tensorflow tensor
		preprocessed CAM 

	real_image : tensorflow tensor
		preprocessed US image
	"""
	input_image, real_image = load_image(sample_path)
	input_image, real_image = random_jitter(input_image, real_image)
	input_image, real_image = normalize(input_image, real_image)

	return input_image, real_image


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Pix2Pix GAN')
	parser.add_argument("attribute", type=str, help="Attribute to classification task: 'Plane' or 'Brain_plane'")
	parser.add_argument("clas", type=str, help="Class to classification task: example 'Fetal brain' or 'Trans-cerebellum'")
	args = parser.parse_args()

	# IMAGES PATH
	main_path = 'GAN/'+ args.attribute + '/' + args.clas + '/train'

	input_image, real_image = load_image(main_path + '/sample_8.png')
	
	## PREPROCESSING
	BUFFER_SIZE = len(os.listdir(main_path))   # The facade training set consist of 400 images
	BATCH_SIZE = 1       # The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
	IMG_WIDTH = 256      # Each image is 256x256 in size
	IMG_HEIGHT = 256
	
	input_image, real_image = load_image_train(main_path + '/sample_8.png')

	## MAKE tf.Dataset
	train_dataset = tf.data.Dataset.list_files(main_path +  '/*.png')	
	train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.AUTOTUNE)
	print(BUFFER_SIZE)
	print(train_dataset)
	plt.show()




	
	
