#!/user/bin/env python3

# script to retrieve functions
import argparse


def get_input_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('data_directory', help="The Directory to pull images from")  # data_directory
	parser.add_argument('--save_dir', default="checkpoint_dir",
						help="The Directory to save Checkpoints")  # data_directory
	parser.add_argument('--arch', default="vgg19", help="The architecture to train the model")
	parser.add_argument('--learning_rate', default="0.001", help="Set the learning rate")
	parser.add_argument('--hidden_units', default="512",
						help="Set the number of layers/nodes of hidden units seperated by comma")
	parser.add_argument('--epochs', default="5", help="Set the Number of Epochs to use")
	parser.add_argument('--drop_out', default="0.5", help="Set the Number of Epochs to use")
	parser.add_argument('--gpu', default="cuda", help="Set the use CPU or GPU")

	return parser.parse_args()


def get_input_args_predict():
	parser = argparse.ArgumentParser()

	parser.add_argument('image_path', help="The Directory to pull images from")  # data_directory
	parser.add_argument('--checkpoint', help="Location of checkpoint file")  # data_directory
	parser.add_argument('--top_k', default="vgg19", help="Return top K most likely classes")
	parser.add_argument('--category_names', default="cat_to_name.json", help="Category Names of Images")
	parser.add_argument('--gpu ', default="cuda", help="Set the use CPU or GPU")

	return parser.parse_args()


def check_command_line_arguments(in_arg):
	pass
