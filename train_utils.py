#!/usr/bin/env python3

# script for app functions
import copy
import time

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import os, os.path
from utils import classify_model


def check_command_line_arguments(in_arg):
	if in_arg is None:
		print()
	else:
		print()


def perform_transformations(data_dir):
	print('Performing Transformations')
	train_dir = data_dir + '/train'
	valid_dir = data_dir + '/valid'
	test_dir = data_dir + '/test'

	data_transforms = {
		'train': transforms.Compose([
			transforms.RandomRotation(45),
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406],
								 [0.229, 0.224, 0.225])
		]),
		'valid': transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406],
								 [0.229, 0.224, 0.225])
		]),
		'test': transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406],
								 [0.229, 0.224, 0.225])
		]),
	}

	hash_dirs = {'train': train_dir, 'valid': valid_dir, 'test': test_dir}
	image_datasets = {
		x: datasets.ImageFolder(hash_dirs[x], transform=data_transforms[x])
		for x in ['train', 'valid', 'test']
	}

	dataloaders = {}
	for x in ['train', 'valid', 'test']:
		if x != 'train':
			dataloaders[x] = torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True)
		else:
			dataloaders[x] = torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True)




	dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}

	return image_datasets, dataloaders, dataset_sizes


def get_count_images_folder(data_dir):
	print('Getting Images from Folder....', data_dir)
	files = folders = 0

	for _, dirnames, filenames in os.walk(data_dir):
		# ^ this idiom means "we won't be using this value"
		files += len(filenames)
		folders += len(dirnames)
	return folders


def check_create_folder(folder_name):
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)


def classify_optimize_model(hidden_units, architecture, final_nodes, drop_out, pretrained=True):
	print('Classifying models and optimization ... ')
	return classify_model(hidden_units, architecture, final_nodes, float(drop_out), pretrained)


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs, device='cuda'):
	since = time.time()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		# Each epoch has a training and validation phase
		for phase in ['train', 'valid']:
			if phase == 'train':
				scheduler.step()
				model.train()  # Set model to training mode
			else:
				model.eval()  # Set model to evaluate mode

			running_loss = 0.0
			running_corrects = 0

			# Iterate over data.
			for inputs, labels in dataloaders[phase]:
				inputs = inputs.to(device)
				labels = labels.to(device)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(inputs)
					_, preds = torch.max(outputs, 1)
					loss = criterion(outputs, labels)

					# backward + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()

				# statistics
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects.double() / dataset_sizes[phase]

			print('{} Loss: {:.4f} Acc: {:.4f}'.format(
				phase, epoch_loss, epoch_acc))

			# deep copy the model
			if phase == 'valid' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model


def do_deep_learning(model, trainloader, validloader, epochs, print_every, criterion, optimizer, device='cpu'):
	epochs = int(epochs)
	print_every = print_every
	steps = 0

	# change to devise
	model.to(device)

	for e in range(epochs):
		running_loss = 0
		model.train()
		for ii, (inputs, labels) in enumerate(trainloader):
			steps += 1

			inputs, labels = inputs.to(device), labels.to(device)

			optimizer.zero_grad()

			# Forward and backward passes
			outputs = model.forward(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()

			if steps % print_every == 0:
				model.eval()

				# Turn off gradients for validation, saves memory and computations
				with torch.no_grad():
					test_loss, accuracy = validation(model, validloader, criterion, device)

				print(
					'Epoch: {}/{}.. '.format(e + 1, epochs),
					'Training Loss: {:.3f}.. '.format(running_loss / print_every),
					'Validation Loss: {:.3f}.. '.format(test_loss / len(validloader)),
					'Validation Accuracy: {:.3f}'.format(accuracy / len(validloader))
				)

				running_loss = 0

				# Make sure training is back on
				model.train()


def validation(model, validloader, criterion, device):
	test_loss = 0
	accuracy = 0
	for images, labels in validloader:
		images = images.to(device)
		labels = labels.to(device)
		output = model.forward(images)
		test_loss += criterion(output, labels).item()

		ps = torch.exp(output)
		equality = (labels.data == ps.max(dim=1)[1])
		accuracy += equality.type(torch.FloatTensor).mean()
	return test_loss, accuracy


def check_accuracy_on_test(model, testloader, device='cuda'):
	correct = 0
	total = 0
	model.eval()
	model.to(device)
	with torch.no_grad():
		for data in testloader:
			images, labels = data
			images = images.cuda()
			labels = labels.cuda()
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	print('Accuracy of the network on the  test images: %d %%' % (100 * correct / total))


def save_check_point(checkpoint_prop, filepath, save_dir):
	check_create_folder(save_dir)
	torch.save(checkpoint_prop, save_dir + "/" + filepath)
