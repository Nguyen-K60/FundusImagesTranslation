import os
from os import listdir
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import asarray

import numpy as np

import tensorflow as tf


def normalize(image):
	image = (image / 127.5) - 1
	return image
def load(image_file, size):
	image = load_img(image_file, target_size=size, interpolation='bicubic')
	image = img_to_array(image)
	image = normalize(image)
	return asarray(image)
def load_dataset(directory, img_size):
	data_list = list()
	names = []
	for filename in sorted(listdir(directory)):
		img = load(directory + filename, img_size)
		data_list.append(img)
		names.append(filename)
	return asarray(data_list), names
def random_crop(img1, img2, patch_height, patch_width, num_patches):
	stacked_img = np.concatenate((img1, img2), axis=2)
	patches1 = list()
	patches2 = list()
	for i in range(num_patches):
		cropped_img = tf.image.random_crop(stacked_img, size=[patch_height, patch_width, 6])
		patch1 = cropped_img[:, :, 0:3]
		patch2 = cropped_img[:, :, 3:6]
		patches1.append(patch1)
		patches2.append(patch2)
	return asarray(patches1), asarray(patches2)
def load_patches(directory1, directory2, img_size, patch_height, patch_width, num_patches):
	patches1 = list()
	patches2 = list()
	count = 1
	for filename1, filename2 in zip(sorted(listdir(directory1)), sorted(listdir(directory2))):
		print('loading image ', count)
		count = count + 1
		img1 = load(directory1 + filename1, img_size)
		img1 = normalize(img1)
		img2 = load(directory2 + filename2, img_size)
		img2 = normalize(img2)
		patch1, patch2 = random_crop(img1, img2, patch_height, patch_width, num_patches)
		patches1.append(patch1)
		patches2.append(patch2)
	patches1 = asarray(patches1)
	patches2 = asarray(patches2)
	patches1 = np.reshape(patches1, (patches1.shape[0]*patches1.shape[1], patches1.shape[2], patches1.shape[3], patches1.shape[4]))
	patches2 = np.reshape(patches2, (patches2.shape[0]*patches2.shape[1], patches2.shape[2], patches2.shape[3], patches2.shape[4]))
	return patches1, patches2
def extract_patches(img, patch_height, patch_width): # img shape (bs, height, width, 3)
  num_rows = int(img.shape[1]/patch_height)
  num_cols = int(img.shape[2]/patch_width)
  patches = list()
  for height in range (num_rows):
    for width in range (num_cols):
      patch = img[:, height*patch_height:(height+1)*patch_height, width*patch_width:(width+1)*patch_width, :]
      patches.append(patch)
  return patches
def assemble_patches(patches, num_rows, num_cols):
  rows = list()
  for height in range (num_rows):
    row = list()
    for width in range (num_cols):
      row.append(patches[height*num_cols+width])
    r = row[0]
    for i in range(1, num_cols):
      r = np.concatenate((r, row[i]), axis=2)
    rows.append(r)
  img = rows[0]
  for i in range(1, num_rows):
    img = np.concatenate((img, rows[i]), axis=1)
  return img

def MakeFolder(path):
	if os.path.exists(path) == False:
		os.makedirs(path)