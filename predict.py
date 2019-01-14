from get_input_arg import *
from train_utils import *
from predict_utils import *
import json



def main():
    # accept arguments from command line
    in_arg = get_input_args_predict()

    # check the argument sent in
    check_command_line_arguments(in_arg)

    model = load_checkpoint(in_arg.checkpoint)

    print("Image to be analyzed")
    print("=" * 40)


    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)

    probability, label_ids, flower_names = predict(in_arg.image_path, model, cat_to_name, topk=5)
    for p, l, f in zip(probability, label_ids, flower_names):
        print("Flower Name: ", f)
        print("Label ID: ", l)
        print("Probabilitity: ", p)

if __name__ == "__main__":
    main()