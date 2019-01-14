#!/usr/bin/env python3

# script for app functions
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sb
from utils import classify_model


def process_image(image):
	''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

	# TODO: Process a PIL image for use in a PyTorch model

	# load image
	img = Image.open(image)

	# check and resize image
	if img.size[0] > img.size[1]:
		img.thumbnail((4000, 256))
	else:
		img.thumbnail((256, 4000))

	# crop out the center

	left = (img.width - 224) / 2
	right = left + 224
	bottom = (img.height - 224) / 2
	top = bottom + 224
	img = img.crop((left, bottom, right, top))

	# Normalize
	img = np.array(img) / 255
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	img = (img - mean) / std

	img = img.transpose((2, 0, 1))

	return img


def imshow(image, ax=None, title=None):
	if ax is None:
		fig, ax = plt.subplots()

	# PyTorch tensors assume the color channel is the first dimension
	# but matplotlib assumes is the third dimension
	image = image.transpose((1, 2, 0))

	# Undo preprocessing
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	image = std * image + mean

	# Image needs to be clipped between 0 and 1 or it looks like noise when displayed
	image = np.clip(image, 0, 1)

	ax.imshow(image)

	return ax


def predict(image_path, model, cat_to_name, topk=5):
	''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

	# TODO: Implement the code to predict the class from an image file

	processed_image = process_image(image_path)
	image_to_tensor = torch.from_numpy(processed_image).type(torch.FloatTensor)

	input_to = image_to_tensor.unsqueeze(0)
	log_softmax_results = model.forward(input_to)
	probablitity_results = torch.exp(log_softmax_results)
	probs, label_ids = probablitity_results.topk(topk)

	probs = probs.detach().numpy().tolist()[0]
	label_ids = label_ids.detach().numpy().tolist()[0]

	idx_to_class = {val: key for key, val in
					model.class_to_idx.items()}
	labels = [idx_to_class[label_id] for label_id in label_ids]
	flower_names = [cat_to_name[idx_to_class[label_id]] for label_id in label_ids]

	return probs, label_ids, flower_names


def sanity_checking(image_path, model, cat_to_name):
	plt.figure(figsize=(6, 10))
	ax = plt.subplot(2, 1, 1)

	flower_num = image_path.split('/')[2]
	title_ = cat_to_name[flower_num]

	img = process_image(image_path)
	imshow(img, ax, title=title_);

	probs, labs, flowers = predict(image_path, model)
	return probs, labs, flowers
	#plt.subplot(2, 1, 2)
	#sb.barplot(x=probs, y=flowers, color=sb.color_palette()[0]);
	#plt.show()


def model_property_arrange(model, classifier, checkpoint):
	model.classifier = classifier
	model.class_to_idx = checkpoint['class_to_idx']
	model.load_state_dict(checkpoint['state_dict'])
	return model


def process_show_image(image_path):
	img = process_image(image_path)
	imshow(img)


def load_checkpoint(filepath):
	print("loading checkpoint ... ")
	checkpoint = torch.load(filepath)
	split_units = checkpoint['hidden_units'].split(',')
	print("split_units in array form", split_units)
	hdu = [int(i) for i in split_units]
	model = classify_model(hdu, checkpoint['arch'], checkpoint['final_nodes'], True)
	model.load_state_dict(checkpoint['state_dict'])
	model.class_to_idx = checkpoint['class_to_idx']

	return model
