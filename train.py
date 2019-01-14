from get_input_arg import *
from train_utils import *
from workspace_utils import active_session
from torch.optim import lr_scheduler


def main():
	# accept arguments from command line
	in_arg = get_input_args()

	# check the argument sent in
	check_command_line_arguments(in_arg)

	data_dir = in_arg.data_directory
	train_dir = data_dir + '/train'

	image_datasets, dataloaders, dataset_sizes = perform_transformations(data_dir)

	final_nodes = get_count_images_folder(train_dir)

	print('Total Images:', final_nodes)

	split_units = in_arg.hidden_units.split(',')
	hdu = [int(i) for i in split_units]
	model = classify_optimize_model(hdu, in_arg.arch, final_nodes, in_arg.drop_out, True)
	print(in_arg)
	if in_arg.gpu == 'cuda':
		model = model.cuda()
	criterion = nn.NLLLoss()
	optimizer = optim.Adam(model.classifier.parameters(), lr=float(in_arg.learning_rate))
	scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

	with active_session():
		do_deep_learning(
			model, dataloaders['train'], dataloaders['valid'], in_arg.epochs, 40, criterion, optimizer, scheduler, in_arg.gpu
		)
		# model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, sched, int(in_arg.epochs), in_arg.gpu)
	model.train()
	check_accuracy_on_test(model, dataloaders['test'], in_arg.gpu)

	model.class_to_idx = image_datasets['train'].class_to_idx
	checkpoint = {
		'arch': in_arg.arch,
		'state_dict': model.state_dict(),
		'class_to_idx': model.class_to_idx,
		'hidden_units': in_arg.hidden_units,
		'final_nodes': final_nodes
	}
	model_save_path = "model_classifier_path_" + in_arg.arch + "_.pth"

	save_check_point(checkpoint, model_save_path, in_arg.save_dir)

	print("Path of Classifier: ", model_save_path)


if __name__ == "__main__":
	main()
