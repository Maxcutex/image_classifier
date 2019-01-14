#!/usr/bin/env python3

# script for app functions
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict


def check_command_line_arguments(in_arg):
	if in_arg is None:
		print()
	else:
		print()


def perform_transformations(data_dir):
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
	image_datasets = {x: datasets.ImageFolder(hash_dirs[x], transform=data_transforms[x]) for x in
					  ['train', 'valid', 'test']}

	dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in
				   ['train', 'valid', 'test']}

	return image_datasets, dataloaders


def select_model(model_type):
	if model_type == 'vgg19':
		model = models.vgg19(pretrained=True)
	if model_type == 'vgg16':
		model = models.vgg16(pretrained=True)

	for param in model.parameters():
		param.requires_grad = False
	return model


def classify_optimize_model(model, hidden_units, arch, final_nodes):
	new_str_unit = hidden_units.split(',')

	a = [2345]
	hu = a + [int(i) for i in new_str_unit]
	od = []
	if arch == "vgg19":
		in_features = model.classifier[0].Linear.in_features
		in_array = [int(in_features)] + [int(i) for i in new_str_unit] + [int(final_nodes)]
		zipped_list = zip(hu[:-1], hu[1:])
		for x, y in zipped_list:
			od.append((nn.Linear(x, y)))
			od.append(('relu', nn.ReLU()))

		od.pop()
		od.append(('output', nn.LogSoftmax(dim=1)))

	classifier = nn.Sequential(OrderedDict(od))

	model.classifier = classifier

	criterion = nn.NLLLoss()
	optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

	return criterion, optimizer


def do_deep_learning(model, trainloader, epochs, print_every, criterion, optimizer, device='cpu'):
	epochs = epochs
	print_every = print_every
	steps = 0

	# change to devise
	model.to(device)

	for e in range(epochs):
		running_loss = 0
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
				print("Epoch: {}/{}... ".format(e + 1, epochs),
					  "Loss: {:.4f}".format(running_loss / print_every))

				running_loss = 0


def validation(model, validloader, criterion):
	test_loss = 0
	accuracy = 0
	model.eval()
	for images, labels in validloader:
		output = model.forward(images)
		test_loss += criterion(output, labels).item()

		ps = torch.exp(output)
		equality = (labels.data == ps.max(dim=1)[1])
		accuracy += equality.type(torch.FloatTensor).mean()
	model.train()
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
	model.train()
	print('Accuracy of the network on the  test images: %d %%' % (100 * correct / total))


def save_check_point(model, checkpoint_prop, filepath):
	torch.save(checkpoint_prop, filepath)


def load_checkpoint(filepath):
	checkpoint = torch.load(filepath)
	model = models.vgg19(pretrained=True)
	for param in model.parameters():
		param.requires_grad = False
	model.class_to_idx = checkpoint['class_to_idx']
	classifier = nn.Sequential(OrderedDict([
		('fc1', nn.Linear(25088, 4096)),
		('relu', nn.ReLU()),
		('fc2', nn.Linear(4096, 102)),
		('output', nn.LogSoftmax(dim=1))
	]))
	model.classifier = classifier

	model.load_state_dict(checkpoint['state_dict'])

	return model
