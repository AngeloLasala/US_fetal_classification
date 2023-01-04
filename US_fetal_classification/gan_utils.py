"""
Module with usefull function for GAN
"""
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython import display
from image_utils import normalization
from tensorflow.keras.preprocessing import image_dataset_from_directory
import datetime
import time

def load_image(sample_path):
		"""
		Load a SINGLE couple of input_image and real_image for GAN

		Parameter
		---------
		main_path : string
			main path of folder

		sample_name : string
			image's path

		Returns
		------
		input_image : tensorflow tensor
			input imgage, i.e. CAM 

		real_image : tensorflow tensor
			real image, i.e. US image
		"""
		
		raw_image = tf.io.read_file(sample_path)
		image = tf.image.decode_png(raw_image, channels=3)

		w = tf.shape(image)[0]
		w = w // 2
		input_image = image[:w, :, :]
		real_image = image[w:, :, :]

		input_image = tf.cast(input_image, tf.float32)
		real_image = tf.cast(real_image, tf.float32)

		return input_image, real_image

def resize(input_image, real_image, height, width):
	"""
	Resize the input and the real image for gan

	Parameters
	----------
	input_image : tensorflow tensor
		input imgage, i.e. CAM 

	real_image : tensorflow tensor
		real image, i.e. US image

	height : integer
		height of resized image 

	width : integer
		width of resized image

	Returns
	-------
	input_image : tensorflow tensor
		resized CAM 

	real_image : tensorflow tensor
		resized US image
	"""
	input_image = tf.image.resize(input_image, [height, width],
								method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	real_image = tf.image.resize(real_image, [height, width],
								method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

	return input_image, real_image

def random_crop(input_image, real_image, height, width):
	"""
	Random cropping of input and the real image for gan

	Parameters
	----------
	input_image : tensorflow tensor
		input imgage, i.e. CAM 

	real_image : tensorflow tensor
		real image, i.e. US image

	height : integer
		height of cropped image 

	width : integer
		width of cropped image
	Returns
	-------
	cropped_image : tensorflow tensor
		cropped CAM 

	cropped_image : tensorflow tensor
		cropped US image
	"""
	stacked_image = tf.stack([input_image, real_image], axis=0)
	cropped_image = tf.image.random_crop(
		stacked_image, size=[2, height, width, 3])

	return cropped_image[0], cropped_image[1]

def normalize(input_image, real_image):
	input_image = (input_image / 127.5) - 1
	real_image = (real_image / 127.5) - 1

	return input_image, real_image


## MODEL FUNCTION ##############################################################################

def downsample(filters, size, apply_batchnorm=True):
	"""
	U-Net downsampling section. It is composed by:
	Convolution -> Batch normalization -> Leaky ReLU

	Parameters
	----------
	filters : integer
		filter for the convolution

	size : integer
		size parameter for the convolution

	Returns
	-------
	result : tensorflow model
		basic element model for the downsampling encoder 
	"""
	initializer = tf.random_normal_initializer(0., 0.02)

	result = tf.keras.Sequential()
	result.add(
		tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
								kernel_initializer=initializer, use_bias=False))

	if apply_batchnorm:
		result.add(tf.keras.layers.BatchNormalization())

	result.add(tf.keras.layers.LeakyReLU())

	return result

def upsample(filters, size, apply_dropout=False):
	"""
	U-Net upsample section. It is composed by:
	Convolution -> Batch normalization -> Leaky ReLU

	Parameters
	----------
	filters : integer
		filter for the convolution

	size : integer
		size parameter for the convolution

	Returns
	-------
	result : tensorflow model
		basic element model for the upsampling encoder 
	"""
	initializer = tf.random_normal_initializer(0., 0.02)

	result = tf.keras.Sequential()
	result.add(
		tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
										padding='same',
										kernel_initializer=initializer,
										use_bias=False))

	result.add(tf.keras.layers.BatchNormalization())

	if apply_dropout:
		result.add(tf.keras.layers.Dropout(0.5))

	result.add(tf.keras.layers.ReLU())

	return result

def Generator(output_channel):
	"""
	Generetor of Pix2Pix condictional Gan

	Parameters
	----------
	output_channel : integer
		number of channel in generated image
	Return
	------
	model : tensorflow model
		Generetor model
	"""
	inputs = tf.keras.layers.Input(shape=[256, 256, 3])

	down_stack = [
		downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
		downsample(128, 4),  # (batch_size, 64, 64, 128)
		downsample(256, 4),  # (batch_size, 32, 32, 256)
		downsample(512, 4),  # (batch_size, 16, 16, 512)
		downsample(512, 4),  # (batch_size, 8, 8, 512)
		downsample(512, 4),  # (batch_size, 4, 4, 512)
		downsample(512, 4),  # (batch_size, 2, 2, 512)
		downsample(512, 4),  # (batch_size, 1, 1, 512)
	]

	up_stack = [
		upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
		upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
		upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
		upsample(512, 4),  # (batch_size, 16, 16, 1024)
		upsample(256, 4),  # (batch_size, 32, 32, 512)
		upsample(128, 4),  # (batch_size, 64, 64, 256)
		upsample(64, 4),  # (batch_size, 128, 128, 128)
	]

	initializer = tf.random_normal_initializer(0., 0.02)
	last = tf.keras.layers.Conv2DTranspose(output_channel, 4,
											strides=2,
											padding='same',
											kernel_initializer=initializer,
											activation='tanh')  # (batch_size, 256, 256, 3)

	x = inputs

	# Downsampling through the model
	skips = []
	for down in down_stack:
		x = down(x)
		skips.append(x)

	skips = reversed(skips[:-1])

	# Upsampling and establishing the skip connections
	for up, skip in zip(up_stack, skips):
		x = up(x)
		x = tf.keras.layers.Concatenate()([x, skip])

	x = last(x)

	return tf.keras.Model(inputs=inputs, outputs=x)

def generator_loss(disc_generated_output, gen_output, target, lambda_gan=100):
	"""
	Generator loss of Pix2Pix conditional GAN, it learns a structured loss that penalizes a possible structure 
	that differs from the network output and the target image

	Parameters
	----------
	disc_generated_output : tensorflow object
		output of discriminator. This is tha part of classic GAN loss

	gen_output : tensorflow object
		generator image
		
	target : tensorflow object
		target image, real US image

	lambda_gen : integer
		weight of L1 loss faction. suggest lambda=100
	"""
	loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

	gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

	# Mean absolute error
	l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

	total_gen_loss = gan_loss + (lambda_gan * l1_loss)

	return total_gen_loss, gan_loss, l1_loss

	
def Discriminator():
	"""
	Discriminator of Pix2Pix condictional Gan

	Return
	------
	model : tensorflow model
		Generetor model
	"""
	initializer = tf.random_normal_initializer(0., 0.02)

	inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
	tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

	x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

	down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
	down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
	down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

	zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
	conv = tf.keras.layers.Conv2D(512, 4, strides=1,
									kernel_initializer=initializer,
									use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

	batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

	leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

	zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

	last = tf.keras.layers.Conv2D(1, 4, strides=1,
									kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

	return tf.keras.Model(inputs=[inp, tar], outputs=last)

def discriminator_loss(disc_real_output, disc_generated_output):
	"""
	Discriminator loss of Pix2Pix conditional GAN

	Parameters
	----------
	disc_real_output : tensorflow object
		real image

	disc_generated_output : tensorflow object
		generator image

	Returns
	-------
	loss
	"""
	loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

	real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

	generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

	total_disc_loss = real_loss + generated_loss

	return total_disc_loss
################################################################################################

## TRAINING FUNCTION ###########################################################################
@tf.function
def train_step(generator, discriminator, 
			   input_image, target, step, 
			   generator_optimizer, discriminator_optimizer):
	"""
	Single training step of Pix2Pix GAN. COmpute the gradiente for 
	the generetor anf the discriminator for a single presentation of images

	Parameters
	----------
	input_image : array
		input image, i.e. CAM

	target : array
		target image: i.e. US image
	
	step : integer
		step
	"""
	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		gen_output = generator(input_image, training=True)

		disc_real_output = discriminator([input_image, target], training=True)
		disc_generated_output = discriminator([input_image, gen_output], training=True)

		gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
		disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

	generator_gradients = gen_tape.gradient(gen_total_loss,
											generator.trainable_variables)
	discriminator_gradients = disc_tape.gradient(disc_loss,
												discriminator.trainable_variables)

	generator_optimizer.apply_gradients(zip(generator_gradients,
											generator.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
												discriminator.trainable_variables))

def fit(train_ds, test_ds, steps, 
		generator, discriminator, 
		generator_optimizer, discriminator_optimizer,
		checkpoint,
		name, save_path, 
		checkpoint_prefix,
		time_steps = 1000,
		checkpoint_steps = 5000):
	"""
	Fit function for Pix2Pix gan model

	Parameters
	----------
	train_dataset : tensorflow dataset
		train dataset

	test_ds : tensorflow dataset
		test dataset

	steps : integer
		number of training steps
	
	generator : tansorflow model
		generator model

	discriminator : tansorflow model
		discriminator model

	generator_optimazer : tensorflow optimazer
		generator's optimazer

	discriminator_optimazer : tensorflow optimazer
		discriminator's optimazer

	checkpoint : tensorflow checkpoint
		object to save the checkpoint. tf.train.Checkpoint

	name : string
		name of output image

	save_path : string
		folder's path where the image are saved

	checkpoint_prefix : string
		folder's path for the checkpoints

	time_steps : integer
		number of step to compute the computational time

	checkpoint_steps : integer
		number of step to save the checkpoint
	"""
	example_input, example_target = next(iter(test_ds.take(1)))
	start = time.time()

	print(time_steps, checkpoint_steps)
	for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
		if (step) % time_steps == 0:
			display.clear_output(wait=True)

			if step != 0:
				print(f'Time taken for {time_steps} steps: {time.time()-start:.2f} sec\n')

			start = time.time()

			generate_images(generator, example_input, example_target,
							name = name + f'_{step.numpy()}',
							save_path = save_path)
			print(f"Step: {step//1000}k")

		train_step(generator, discriminator, input_image, target, step, generator_optimizer, discriminator_optimizer)

		# Training step
		if (step+1) % 10 == 0:
			print('.', end='', flush=True)


		# Save (checkpoint) the model every 5k steps
		if (step + 1) % checkpoint_steps == 0:
			checkpoint.save(file_prefix=checkpoint_prefix)


################################################################################################

## FUNCTION FOR THE PLOT #######################################################################
def generate_images(model, test_input, tar,
					name='name',
					save_path='GAN_real_time'):
	"""
	Generate images during the traning of Pix2Pix Gan 

	Parameters
	----------
	model : tensorflow model

	test_input : array
		input image of the model

	tar : array
		ground truth image
	"""
	prediction = model(test_input, training=True)

	display_list = [test_input[0], tar[0], prediction[0]]
	title = ['Input Image', 'Ground Truth', 'Predicted Image']

	plt.figure(figsize=(15, 15), num=name)
	for i in range(3):
		plt.subplot(1, 3, i+1)
		plt.title(title[i])
		# Getting the pixel values in the [0, 1] range to plot.
		plt.imshow(display_list[i] * 0.5 + 0.5)
		plt.axis('off')

	# os.makedirs(save_path)
	plt.savefig(save_path + '/'+ name + '.png')

def generate_images_test(model, inp, tar, height = 224, weight = 224):
	"""
	Return generated image grom GAN generator and the real image 
	from test dataset

	Parameters
	----------
	model : tensorflow model
		model of generator

	inp : tensorflow image
		input image for the generator
	
	tar : tensorflow image
		real image

	height : integer
		height of output images
	
	weight : integer
		weight of output image

	Returns
	-------
	genereted_image : tensorflow image
		genereted image ready to classification

	real_image : tensorflow image
		real image ready for the classification
	"""
	
	generated_image = model(inp, training=True)
		
	generated_image = normalization(generated_image[0,...],0,255)
	generated_image = tf.image.resize(generated_image, [height, weight])
	generated_image = tf.expand_dims(generated_image, axis=0)

	real_image = normalization(tar[0,...],0,255)
	real_image = tf.image.resize(real_image, [height, weight])
	real_image = tf.expand_dims(real_image, axis=0)
	
	return generated_image, real_image

## Deep Convolution GAN ###########################################################################

def Generator_plane(input_shape = 256, output_channel=1):
	"""
	Return the Generator model for DCGAN starting for random noise 
	instead of cam
	"""
	inputs = tf.keras.layers.Input(shape=[input_shape, input_shape, 1])

	down_stack = [
		downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
		downsample(128, 4),  # (batch_size, 64, 64, 128)
		downsample(256, 4),  # (batch_size, 32, 32, 256)
		downsample(512, 4),  # (batch_size, 16, 16, 512)
		downsample(512, 4),  # (batch_size, 8, 8, 512)
		downsample(512, 4),  # (batch_size, 4, 4, 512)
		downsample(512, 4),  # (batch_size, 2, 2, 512)
		downsample(512, 4),  # (batch_size, 1, 1, 512)
	]

	up_stack = [
		upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
		upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
		upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
		upsample(512, 4),  # (batch_size, 16, 16, 1024)
		upsample(256, 4),  # (batch_size, 32, 32, 512)
		upsample(128, 4),  # (batch_size, 64, 64, 256)
		upsample(64, 4),  # (batch_size, 128, 128, 128)
	]

	initializer = tf.random_normal_initializer(0., 0.02)
	last = tf.keras.layers.Conv2DTranspose(output_channel, 4,
											strides=2,
											padding='same',
											kernel_initializer=initializer,
											activation='tanh')  # (batch_size, 256, 256, 1)

	x = inputs

	# Downsampling through the model
	skips = []
	for down in down_stack:
		x = down(x)
		skips.append(x)

	skips = reversed(skips[:-1])

	# Upsampling and establishing the skip connections
	for up, skip in zip(up_stack, skips):
		x = up(x)
		x = tf.keras.layers.Concatenate()([x, skip])

	x = last(x)

	return tf.keras.Model(inputs=inputs, outputs=x)

def Discriminator_plane(input_shape = 256):
	"""
	Return the discriminator model for DCGAN, with only image input
	"""
	initializer = tf.random_normal_initializer(0., 0.02)

	tar = tf.keras.layers.Input(shape=[input_shape , input_shape, 1], name='target_image')

	down1 = downsample(64, 4, False)(tar)  # (batch_size, 128, 128, 64)
	down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
	down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

	zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
	conv = tf.keras.layers.Conv2D(512, 4, strides=1,
									kernel_initializer=initializer,
									use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

	batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

	leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

	zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

	last = tf.keras.layers.Conv2D(1, 4, strides=1,
									kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

	flatten = tf.keras.layers.Flatten()(last)

	output = tf.keras.layers.Dense(1)(flatten)

	return tf.keras.Model(inputs=tar, outputs=output)

def generator_loss_plane(fake_output):
	"""
	return the generator loss for DCGAN
	"""
	cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
	return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss_plane(real_output, fake_output):
	"""
	return the discriminator loss for DCGAN
	"""
	cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
	real_loss = cross_entropy(tf.ones_like(real_output), real_output)
	fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
	total_loss = real_loss + fake_loss
	return total_loss

@tf.function
def train_step_plane(images, batch_size,
					 generator, discriminator,
					 generator_optimizer, discriminator_optimizer
					 ):
	noise = tf.random.normal([batch_size, 256, 256, 1])

	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		generated_images = generator(noise, training=True)

		real_output = discriminator(images, training=True)
		fake_output = discriminator(generated_images, training=True)

		gen_loss = generator_loss_plane(fake_output)
		disc_loss = discriminator_loss_plane(real_output, fake_output)

	gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
	gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

	generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs, batch_size,
		generator, discriminator,
		generator_optimizer, discriminator_optimizer,
		checkpoint,
		checkpoint_prefix, 
		epochs_ckp, 
		path):
	seed = tf.random.normal([16, 256, 256, 1])

	for epoch in range(epochs):
		start = time.time()
		print(f'epoch: {epoch+1}')

		for image_batch in dataset:
			train_step_plane(image_batch, batch_size,
							generator, discriminator,
					        generator_optimizer, discriminator_optimizer, )
			print('.', end='', flush=True)


		# Generate at the end of the epoch
		display.clear_output(wait=True)
		generate_and_save_images(generator, epoch+1, seed, path)

		# Save the model every epochs_ckp epocks
		if (epoch + 1) % epochs_ckp == 0:
			checkpoint.save(file_prefix = checkpoint_prefix)

		print (f'Time for epoch {epoch + 1} is {time.time()-start:.2f} sec')

		

def generate_and_save_images(model, epoch, test_input, path):
	# Notice `training` is set to False.
	# This is so all layers run in inference mode (batchnorm).
	predictions = model(test_input, training=False)
	print(predictions.shape)
	fig = plt.figure(figsize=(4, 4))

	for i in range(predictions.shape[0]):
		plt.subplot(4, 4, i+1)
		plt.imshow(predictions[i, :, :, :] * 127.5 + 127.5, cmap='gray')
		plt.axis('off')

	plt.savefig(path + '/image_at_epoch_{:04d}.png'.format(epoch))

## GAN EVALUATION ##################################################################################

def feature_extraction(model, image_path, input_shape = (224,224)):
	"""
	Return the features regardindi the t-SNE. The features come from 
	the last layers pf selected model

	Parameters
	----------
	model : tensorflow model
		model adopted to feature selection

	image_path : string
		path of images, the structure must be compatible with image_dataset_from_directory

	input_shape : tuple (optional)
		shape of image, defoult=(224*224)

	Returns
	-------
	features : numpy array
		array of features. shape (num_of_image, num_of_feature)

	class_dataset : tensorflow dataset
		dateset of imgase and labels
	"""
	
	class_dataset = image_dataset_from_directory(image_path,
                                                shuffle=False,
                                                batch_size = 1,
                                                image_size=input_shape,
                                                interpolation='bilinear')
	features = model.predict(class_dataset, verbose=1)
	# features = features.reshape((features.shape[0],features.shape[1]*features.shape[2]*features.shape[3]))
	
	return features, class_dataset