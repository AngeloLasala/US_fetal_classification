"""
Make the folder with real and synthetic images ready to be evaluated by
a clinitian. The folder will be contain:
- all the generated images from the GAN test set
- the same number of images from the GAN training set, randomly selected
- true answer of the ansawer: Is the image fake or real?
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import random

from makedir import *
from gan_utils import *

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


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Dataset for clinical evaluation of GAN')
	parser.add_argument("attribute", type=str, help="Attribute to classification task: 'Plane' or 'Brain_plane'")
	parser.add_argument("clas", type=str, help="Class to classification task: example 'Fetal brain' or 'Trans-cerebellum'")
	args = parser.parse_args()
	
	## Load Generative model
	OUTPUT_CHANNELS = 3
	generator = Generator(OUTPUT_CHANNELS)
	discriminator = Discriminator()
	generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
	discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
	
	checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
									discriminator_optimizer=discriminator_optimizer,
									generator=generator,
									discriminator=discriminator)

	checkpoint_dir = 'GAN/'+ args.attribute + '/' + args.clas +'/training_checkpoints'
	checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
	checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

	## LOAD SYNTHETIC AND REAL IMAGE
	OUTPUT_CHANNELS = 3
	BATCH_SIZE = 1 
	IMG_WIDTH = 256      
	IMG_HEIGHT = 256
	evaluation_path = 'GAN/'+ args.attribute + '/' + args.clas + '/evaluation'
	smart_makedir(evaluation_path)

	## Synthetic image
	main_path_test = 'GAN/'+ args.attribute + '/' + args.clas + '/test'
	syntetic_sample_n = [int(sample.split('.')[0].split('_')[1]) for sample in os.listdir(main_path_test)]
	print(syntetic_sample_n)

	for n in syntetic_sample_n:
		test_dataset = tf.data.Dataset.list_files(main_path_test + f'/sample_{n}.png')
		test_dataset = test_dataset.map(load_image_test)
		test_dataset = test_dataset.batch(BATCH_SIZE)
		for inp, tar in test_dataset.take(1):
			generated_image, real_image = generate_images_test(generator, inp, tar, 
														height = 224, weight = 224)

			plt.figure(num=f'sample_evaluation_{n}')
			plt.imshow(normalization(generated_image[0],0,1))
			plt.axis('off')
			plt.savefig(evaluation_path + f'/sample_evaluation_{n}' + '.png')
	
	# ## Real image 
	# # I deal with the train test as the test test to avoid bias due to the prepocessing
	main_path_train = 'GAN/'+ args.attribute + '/' + args.clas + '/train'
	real_sample_n = [int(sample.split('.')[0].split('_')[1]) for sample in os.listdir(main_path_train)]
	real_sample_n = random.SystemRandom().sample(real_sample_n, 13)
	print(real_sample_n)

	for n in real_sample_n:
		train_dataset = tf.data.Dataset.list_files(main_path_train + f'/sample_{n}.png')
		train_dataset = train_dataset.map(load_image_test)
		train_dataset = train_dataset.batch(BATCH_SIZE)

		for ii, (inp, tar) in enumerate(train_dataset.take(1)):
			real_image = normalization(tar[0,...],0,255)
			real_image = tf.image.resize(real_image, [224, 224])
			real_image = tf.expand_dims(real_image, axis=0)

			plt.figure(num=f'sample_evaluation_{n}')
			plt.imshow(normalization(real_image[0,...],0,1))
			plt.axis('off')
			plt.savefig(evaluation_path + f'/sample_evaluation_{n}' + '.png')

	with open(evaluation_path + '/right_answer.txt', 'w', encoding='utf-8') as file:
		for n in syntetic_sample_n:
			file.write(f'sample_evaluation_{n}: synthetic \n ')

		for n in real_sample_n:
			file.write(f'sample_evaluation_{n}: real \n ')
	