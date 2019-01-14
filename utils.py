#!/usr/bin/env python3

# script for app functions
from torch import nn
from torchvision import models
from collections import OrderedDict


def build_layers(n_in, hidden_units, n_out, dropout=0.5):
	hidden_layers = [
		nn.Linear(n_in, hidden_units[0]),
		nn.ReLU(),
		nn.Dropout(dropout)
	]  # 1st hidden layer

	for hl1, hl2 in zip(hidden_units[:-1], hidden_units[1:]):
		hidden_layers += [
			nn.Linear(hl1, hl2), nn.ReLU(), nn.Dropout(dropout)
		]

	hidden_layers += [nn.Linear(hidden_units[-1], n_out), nn.LogSoftmax(dim=1)]
	hidden_layers = [("fc{}".format(i), layer) for i, layer in enumerate(hidden_layers)]

	return hidden_layers


def classify_model(hidden_units, architecture, final_nodes, dropout, pretrained=True):
	if architecture == 'alexnet':
		print('[ Building alexnet]')
		model = models.alexnet(pretrained=pretrained)
	elif architecture == 'resnet18':
		print('[Building resnet18]')
		model = models.resnet18(pretrained=pretrained)
	elif architecture == 'resnet34':
		print('[Building resnet34]')
		model = models.resnet34(pretrained=pretrained)
	elif architecture == 'resnet50':
		print('[Building resnet50]')
		model = models.resnet50(pretrained=pretrained)
	elif architecture == 'resnet101':
		print('[Building resnet101]')
		model = models.resnet101(pretrained=pretrained)
	elif architecture == 'resnet152':
		print('[Building resnet152]')
		model = models.resnet152(pretrained=pretrained)
	elif architecture == 'densenet121':
		print('[Building densenet121]')
		model = models.densenet121(pretrained=pretrained)
	elif architecture == 'densenet169':
		print('[Building densenet169]')
		model = models.densenet169(pretrained=pretrained)
	elif architecture == 'densenet201':
		print('[Building densenet201]')
		model = models.densenet201(pretrained=pretrained)
	elif architecture == 'squeezenet1_0':
		print('[Building squeezenet1_0]')
		model = models.squeezenet1_0(pretrained=pretrained)
	elif architecture == 'squeezenet1_1':
		print('[Building squeezenet1_1]')
		model = models.squeezenet1_1(pretrained=pretrained)
	elif architecture == 'vgg11':
		print('[Building vgg11]')
		model = models.vgg11(pretrained=pretrained)
	elif architecture == 'vgg13':
		print('[Building vgg13]')
		model = models.vgg13(pretrained=pretrained)
	elif architecture == 'vgg16':
		print('[Building vgg16]')
		model = models.vgg16(pretrained=pretrained)
	elif architecture == 'vgg19':
		print('[Building vgg19]')
		model = models.vgg19(pretrained=pretrained)
	elif architecture == 'vgg11_bn':
		print('[Building vgg11_bn]')
		model = models.vgg11_bn(pretrained=pretrained)
	elif architecture == 'vgg13_bn':
		print('[Building vgg13_bn]')
		model = models.vgg13_bn(pretrained=pretrained)
	elif architecture == 'vgg16_bn':
		print('[Building vgg16_bn]')
		model = models.vgg16_bn(pretrained=pretrained)
	elif architecture == 'vgg19_bn':
		print('[Building vgg19_bn]')
		model = models.vgg19_bn(pretrained=pretrained)
	else:
		raise ValueError

	for param in model.parameters():
		param.requires_grad = False

	print("before", model)

	if 'resnet' in architecture:
		num_ftrs = model.fc.in_features
		model.fc = nn.Linear(num_ftrs, final_nodes)
	elif 'inception' in architecture:
		num_ftrs = model.fc.in_features
		model.fc = nn.Linear(num_ftrs, final_nodes)
	elif 'squeezenet' in architecture:
		in_ftrs = model.classifier[1].in_channels
		out_ftrs = model.classifier[1].out_channels
		features = list(model.classifier.children())
		features[1] = nn.Conv2d(in_ftrs, final_nodes, kernel_size=(2, 2), stride=(1, 1))
		features[3] = nn.AvgPool2d(12, stride=1)
		model.classifier = nn.Sequential(*features)
		model.num_classes = final_nodes
	elif 'densenet' in architecture:
		num_ftrs = model.classifier.in_features
		model.classifier = nn.Linear(num_ftrs, final_nodes)
	elif 'vgg' in architecture:
		print('original classifier ==>', model.classifier[0])
		print('in features==>', model.classifier[0].in_features)
		num_ftrs = model.classifier[0].in_features
		hidden_layers = build_layers(num_ftrs, hidden_units, final_nodes, dropout)
		model.classifier = nn.Sequential(OrderedDict(hidden_layers))
	elif 'alexnet' in architecture:
		num_ftrs = model.classifier[1].in_features
		hidden_layers = build_layers(num_ftrs, hidden_units, final_nodes, dropout)

		model.classifier = nn.Sequential(OrderedDict(hidden_layers))

	print('Successfully built classifer ...  ')
	print("after", model)

	return model
