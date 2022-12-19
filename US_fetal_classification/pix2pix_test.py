"""
Test the trained generator of Pix2Pix conditional GAN fro pix2pix.py
"""
import os 
import imageio
import argparse
import glob
import tensorflow as tf
from PIL import Image
from gan_utils import *
from tensorflow.keras.preprocessing import image_dataset_from_directory

from image_utils import normalization


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
	parser = argparse.ArgumentParser(description='Pix2Pix GAN - test file')
	parser.add_argument("attribute", type=str, help="Attribute to classification task: 'Plane' or 'Brain_plane'")
	parser.add_argument("clas", type=str, help="Class to classification task: example 'Fetal brain' or 'Trans-cerebellum'")
	args = parser.parse_args()
	
	# MAKE GIF 
	gan_path = 'GAN/'+ args.attribute + '/' + args.clas
	frames = [Image.open(image) for image in glob.glob(gan_path + '/GAN_real_time' +  "/*.png")]
	frame_one = frames[0]
	frame_one.save(gan_path + '/my_awesome.gif', format="GIF", append_images=frames,save_all=True, duration=100, loop=0)

	## LOAD LAST CHECKPOINT MODEL
	OUTPUT_CHANNELS = 3
	BATCH_SIZE = 1 
	IMG_WIDTH = 256      
	IMG_HEIGHT = 256

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

	## EVALUATE ON TEST SET
	main_path_test = 'GAN/'+ args.attribute + '/' + args.clas + '/test'
	test_dataset = tf.data.Dataset.list_files(main_path_test + '/*.png')
	test_dataset = test_dataset.map(load_image_test)
	test_dataset = test_dataset.batch(BATCH_SIZE)

	# for n, (inp, tar) in enumerate(test_dataset.take(13)):
	# 	generate_images(generator, inp, tar, name=f'testing_{n}', save_path='GAN/'+ args.attribute + '/' + args.clas)

	## CLASSIFICATION OF SYNTETIC IMAGE
	models_path = 'Images_classification_' + args.attribute +'/models/' + 'VGG_16' + '_'
	models_path = models_path + '/' + 'train_11'
	model = tf.keras.models.load_model(models_path + '/VGG_16')

	for n, (inp, tar) in enumerate(test_dataset.take(13)):
		generated_image, real_image = generate_images_test(generator, inp, tar, 
													   height = 224, weight = 224)
		prediction_real = model.predict(real_image, verbose=1)
		pred_real = np.argmax(prediction_real, axis=-1)[0]

		prediction_generate = model.predict(generated_image, verbose=1)
		pred_generate = np.argmax(prediction_generate, axis=-1)[0]

		print(f'REAL: {pred_real} - {prediction_real[0]*100}')
		print(f'GENERATE: {pred_generate}- {prediction_generate[0]*100}')

		## PLOTS
		display_list = [inp[0], real_image[0], generated_image[0]]
		title = ['INPUT', 'REAL IMAGE', 'GENERATE IMAGE']
		prediction_title = ['', f' - Classification: {pred_real}, {prediction_real[0][pred_real]*100:.1f}%', 
								f' - Classification: {pred_generate}, {prediction_generate[0][pred_generate]*100:.1f}%']
		plt.figure(figsize=(15,8), num=f'Prova_{n}')
		for i in range(3):
			plt.subplot(1, 3, i+1)
			
			plt.title(title[i] + prediction_title[i])
			# Getting the pixel values in the [0, 1] range to plot.
			plt.imshow(normalization(display_list[i],0,1))
			plt.axis('off')
		# plt.figure()
		# plt.imshow(real_image[0,...]/255)
		# plt.figure()
		# plt.imshow(generated_image[0,...]/255)
		plt.savefig('GAN/'+ args.attribute + '/' + args.clas + f'/testing{n}.png')
		plt.show()



