"""
Deep Convolution Genarative Adversarial Network (DCGAN) for abaltion study 
"""
import os
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

from makedir import *
from gan_utils import *
from image_utils import normalization

def dataset_numpy(path, size=256):
	"""
	Return the preprocesed dataset in numpy array

	Parameters
	----------
	path : string
		main path of train dataset

	size : integer (optional)
		size of image

	Returns
	-------
	numpy_dataset : numpy darray
		numpy dataset
	"""

	list_images = os.listdir(path)
	numpy_dataset = []
	for img_path in list_images:
		cam, image = load_image(main_path_train + '/' + img_path)
		image = tf.image.resize(image, [size,size],
								method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		image = (image / 127.5) - 1
		image = image[:,:,0]
		numpy_dataset.append(image)
	return np.array(numpy_dataset)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Pix2Pix GAN')
	parser.add_argument("attribute", type=str, help="Attribute to classification task: 'Plane' or 'Brain_plane'")
	parser.add_argument("clas", type=str, help="Class to classification task: example 'Fetal brain' or 'Trans-cerebellum'")
	args = parser.parse_args()

	# IMAGES PATH
	main_path_train = 'GAN/'+ args.attribute + '/' + args.clas + '/train'
	dcgan_path = 'GAN/' + args.attribute + '/' + args.clas + '/GAN_real_time_dcgan'
	smart_makedir(dcgan_path)

	## TRAIN DATASET
	SIZE = 256
	BUFFER_SIZE = int(len(os.listdir(main_path_train)))
	BATCH_SIZE = 16
	numpy_dataset = dataset_numpy(main_path_train, size=SIZE)

	train_images = numpy_dataset.reshape(numpy_dataset.shape[0], SIZE, SIZE, 1).astype('float32')
	train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
	print(train_dataset)

	#MODEL 
	generator = Generator_plane(output_channel=1)
	# print(generator.summary())
	# noise = tf.random.normal([256,256,1])
	# generated_image = generator(noise[tf.newaxis, ...], training=False)
	# print(generated_image.shape)

	# plt.figure()
	# plt.imshow(normalization(generated_image[0, :, :, :],0,1), cmap='gray')
	# plt.show()
	
	discriminator = Discriminator_plane()
	# decision = discriminator(generated_image)
	# print(decision)
	# print(discriminator.summary())

	## LOSSES and TRAIN
	cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
	generator_optimizer = tf.keras.optimizers.Adam(1e-4)
	discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

	checkpoint_dir = 'GAN/'+ args.attribute + '/' + args.clas + '/training_checkpoints_dcgan'
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
	checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
									discriminator_optimizer=discriminator_optimizer,
									generator=generator,
									discriminator=discriminator)

	EPOCHS = 50
	EPCS_CKP = 50
	num_examples_to_generate = 16

	train(train_dataset, EPOCHS, BATCH_SIZE,
		generator, discriminator, 
		generator_optimizer, discriminator_optimizer, 
		checkpoint = checkpoint,
		checkpoint_prefix = checkpoint_prefix,
		epochs_ckp = EPCS_CKP,
		path = dcgan_path)