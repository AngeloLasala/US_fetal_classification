"""
Frechet Inception Distance (FID) of two images's distribuction to assert
the quality of sythetic images
"""
import os
import argparse
import numpy as np
from scipy import linalg
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

from makedir import *
from gan_utils import *

def splitting_real_syn(dataset, count):
	"""
	Split the dataset in real and syntetic

	Parameters
	----------
	dataset : tf.dataset
		dataset with real and synthetic image

	count : integer
		number of image to take account

	Returns
	-------
	real_indeces : list
		indeces of real images

	syn_indeces : list
		indeces of synthetic image
	"""

	real_indeces, syn_indeces = [], []
	
	for n, (image, label) in dataset.take(count).enumerate():
		if label.numpy() == 0 :
			real_indeces.append(n.numpy())
		if label.numpy() == 1:
			syn_indeces.append(n.numpy())
	return real_indeces, syn_indeces

def calculate_fid(real_embeddings, generated_embeddings):
	"""
	compute the Frechet Inception Distance (FID) between real and 
	generate set of data

	Parameters
	----------
	real_embeddings : numpy array
		embedding image of real sample. The features come from the Incemption v3

	generate_embeddings : numpy array
		embedding image of syntetic sample. The features come from the Incemption v3
	
	Returns
	-------
	fid : float
		FID
	"""
	# calculate mean and covariance statistics
	mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
	mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings,  rowvar=False)
	
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = linalg.sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	
	# calculate score
	fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='t-SNE evaluation of Cam-cGan model')
	parser.add_argument("attribute", type=str, help="Attribute to classification task: 'Plane' or 'Brain_plane'")
	parser.add_argument("clas", type=str, help="Class to classification task: example 'Fetal brain' or 'Trans-cerebellum'")
	args = parser.parse_args()

	## MAIN PATH	
	gan_path = 'GAN/' + args.attribute + '/' + args.clas 
	fid_path = gan_path + '/fid'
	

	## SYNTHETIC AND REAL DATASET 
	INPUT_SHAPE = (224,224)
	total_dataset = image_dataset_from_directory(fid_path,
													shuffle=False,
                                                	batch_size = 1,
                                                	image_size = INPUT_SHAPE,
                                                	interpolation='bilinear')
													
	n_real_image = len(os.listdir(fid_path + '/' + os.listdir(gan_path + '/fid')[-1]))
	n_syn_image = len(os.listdir(fid_path + '/synthetic'))
	real_pos, syn_pos = splitting_real_syn(total_dataset, n_real_image + n_syn_image)

	## LOAD INCEPTION MODEL
	inception_model = tf.keras.applications.InceptionV3(include_top=False, 
                              weights="imagenet", 
                              pooling='avg')

	## EMBEDDING IMAGE AND FID
	embedding_image = inception_model.predict(total_dataset, verbose=1)
	embedding_real = embedding_image[real_pos]
	embedding_syn = embedding_image[syn_pos]

	fid = calculate_fid(embedding_real, embedding_syn)
	print(f'FID : {fid:.2f}')

	
