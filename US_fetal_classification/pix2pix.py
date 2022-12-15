"""
Pix2Pix file for US fetal brain
"""
import os
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.utils.vis_utils import plot_model
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

def load_image_test(image_file):
	"""
	Load and preproces test_file

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
	input_image, real_image = load_image(image_file)
	input_image, real_image = resize(input_image, real_image,
									IMG_HEIGHT, IMG_WIDTH)
	input_image, real_image = normalize(input_image, real_image)

	return input_image, real_image



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Pix2Pix GAN')
	parser.add_argument("attribute", type=str, help="Attribute to classification task: 'Plane' or 'Brain_plane'")
	parser.add_argument("clas", type=str, help="Class to classification task: example 'Fetal brain' or 'Trans-cerebellum'")
	args = parser.parse_args()

	# IMAGES PATH
	main_path_train = 'GAN/'+ args.attribute + '/' + args.clas + '/train'
	main_path_test = 'GAN/'+ args.attribute + '/' + args.clas + '/test'

	input_image, real_image = load_image(main_path_train + '/sample_8.png')
	
	## PREPROCESSING
	BUFFER_SIZE = len(os.listdir(main_path_train))   # The facade training set consist of 400 images
	BATCH_SIZE = 1       # The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
	IMG_WIDTH = 256      # Each image is 256x256 in size
	IMG_HEIGHT = 256
	
	input_image, real_image = load_image_train(main_path_train + '/sample_8.png')

	## MAKE tf.Dataset
	train_dataset = tf.data.Dataset.list_files(main_path_train + '/*.png')	
	train_dataset = train_dataset.map(load_image_train,
                                	  num_parallel_calls=tf.data.AUTOTUNE)
	train_dataset = train_dataset.shuffle(BUFFER_SIZE)
	train_dataset = train_dataset.batch(BATCH_SIZE)
	
	test_dataset = tf.data.Dataset.list_files(main_path_test + '/*.png')
	test_dataset = test_dataset.map(load_image_test,)
	test_dataset = test_dataset.batch(BATCH_SIZE)

	## MODEL
	OUTPUT_CHANNELS = 3
	LAMBDA = 100
	
	## TEST GENERATOR MODEL
	generator = Generator(OUTPUT_CHANNELS)
	# plot_model(generator, show_shapes=True, show_layer_names=True)
	# plt.figure()
	# plt.imshow(normalization(input_image,0,1))

	# gen_output = generator(input_image[tf.newaxis, ...], training=False)
	# plt.figure()
	# plt.imshow(normalization(gen_output[0, ...],0,1))
	
	## TEST DISCRIMINATOR MODEL
	discriminator = Discriminator()
	# disc_out = discriminator([input_image[tf.newaxis, ...], gen_output], training=False)
	# plt.figure()
	# plt.imshow(disc_out[0, ..., -1],  cmap='RdBu_r')
	# plt.colorbar()
	# plt.show()

	## TRAINING
	EPOCHS = 100 * BUFFER_SIZE
	TIME_EPOCHS = BUFFER_SIZE
	CKP_EPOCHS = 10 * BUFFER_SIZE

	generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
	discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
	
	checkpoint_dir = 'GAN/'+ args.attribute + '/' + args.clas + '/training_checkpoints'
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
	checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
									discriminator_optimizer=discriminator_optimizer,
									generator=generator,
									discriminator=discriminator)
	
	fit(train_dataset, test_dataset, steps= EPOCHS, 
		generator=generator, discriminator=discriminator,
		generator_optimizer=generator_optimizer, 
		discriminator_optimizer=discriminator_optimizer,
		checkpoint = checkpoint,
		name = f'gen_image_step',
		save_path = 'GAN/'+ args.attribute + '/' + args.clas +'/GAN_real_time',
		checkpoint_prefix = checkpoint_prefix,
		time_steps = TIME_EPOCHS,
		checkpoint_steps = CKP_EPOCHS)



	
	
