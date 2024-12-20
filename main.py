"""
Borrowed from Homework 5 - CNNs
CS1430 - Computer Vision
Brown University
"""
import os
import sys
import argparse
import re
from datetime import datetime
# import kagglehub
import tensorflow as tf

import hyperparameters as hp
from models import YourModel
from preprocess import Datasets
from skimage.transform import resize
from tensorboard_utils import \
        ImageLabelingLogger, ConfusionMatrixLogger, CustomModelSaver

from skimage.io import imread
from lime import lime_image
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        '--fourier',
        action='store_true',
        default=False,
        help='Enable Fourier Transform processing. Default is disabled.')
    
    parser.add_argument(
        '--random-fourier',
        action='store_true',
        default=False,
        help='Enable Concatenated Noise of same shape as fourier transform. Default is disabled.')
    
    parser.add_argument(
        '--fourier-only',
        action='store_true',
        default=False,
        help='Enable Fourier Transform processing with fully connected layers only. Default is disabled.')
    
    parser.add_argument(
        '--combined',
        action='store_true',
        default=False,
        help='Fourier Transform processing and CNN seperate, then combined into a final layer. Default is disabled.')
    
    parser.add_argument(
        '--combined-random',
        action='store_true',
        default=False,
        help='Random Fourier Transform processing and CNN seperate, then combined into a final layer. Default is disabled.')

    
    # parser.add_argument(
    #     '--augment',
    #     action='store_true',
    #     default=False,
    #     help='Enable data augmentation. Default is disabled.')
    # parser.add_argument(
    #     '--task',
    #     required=True,
    #     choices=['1', '3'],
    #     help='''Which task of the assignment to run -
    #     training from scratch (1), or fine tuning VGG-16 (3).''')
    # # parser.add_argument(
    # #     '--data',
    # #     default='..'+os.sep+'data'+os.sep,
    # #     help='Location where the dataset is stored.')
    # parser.add_argument(
    #     '--load-vgg',
    #     default='vgg16_imagenet.h5',
    #     help='''Path to pre-trained VGG-16 file (only applicable to
    #     task 3).''')
    # parser.add_argument(
    #     '--load-checkpoint',
    #     default=None,
    #     help='''Path to model checkpoint file (should end with the
    #     extension .h5). Checkpoints are automatically saved when you
    #     train your model. If you want to continue training from where
    #     you left off, this is how you would load your weights.''')
    # parser.add_argument(
    #     '--confusion',
    #     action='store_true',
    #     help='''Log a confusion matrix at the end of each
    #     epoch (viewable in Tensorboard). This is turned off
    #     by default as it takes a little bit of time to complete.''')
    # parser.add_argument(
    #     '--evaluate',
    #     action='store_true',
    #     help='''Skips training and evaluates on the test set once.
    #     You can use this to test an already trained model by loading
    #     its checkpoint.''')
    # parser.add_argument(
    #     '--lime-image',
    #     default='test/Bedroom/image_0003.jpg',
    #     help='''Name of an image in the dataset to use for LIME evaluation.''')

    return parser.parse_args()


def LIME_explainer(model, path, preprocess_fn, timestamp):
    """
    This function takes in a trained model and a path to an image and outputs 4
    visual explanations using the LIME model
    """

    save_directory = "lime_explainer_images" + os.sep + timestamp
    if not os.path.exists("lime_explainer_images"):
        os.mkdir("lime_explainer_images")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    image_index = 0

    def image_and_mask(title, positive_only=True, num_features=5,
                       hide_rest=True):
        nonlocal image_index

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=positive_only,
            num_features=num_features, hide_rest=hide_rest)
        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        plt.title(title)

        image_save_path = save_directory + os.sep + str(image_index) + ".png"
        plt.savefig(image_save_path, dpi=300, bbox_inches='tight')
        plt.show()

        image_index += 1

    image = imread(path)
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    image = resize(image, (hp.img_size, hp.img_size, 3), preserve_range=True)
    image = preprocess_fn(image)
    

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image.astype('double'), model.predict, top_labels=5, hide_color=0,
        num_samples=1000)

    image_and_mask("Top 5 superpixels", positive_only=True, num_features=5,
                   hide_rest=True)

    image_and_mask("Top 5 with the rest of the image present",
                   positive_only=True, num_features=5, hide_rest=False)

    image_and_mask("Pros(green) and Cons(red)",
                   positive_only=False, num_features=10, hide_rest=False)

    ind = explanation.top_labels[0]
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
    plt.colorbar()
    plt.title("Map each explanation weight to the corresponding superpixel")

    image_save_path = save_directory + os.sep + str(image_index) + ".png"
    plt.savefig(image_save_path, dpi=300, bbox_inches='tight')
    plt.show()


def train(model, datasets, checkpoint_path, logs_path, init_epoch):
    """ Training routine. """

    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='batch',
            profile_batch=0),
        ImageLabelingLogger(logs_path, datasets),
        CustomModelSaver(checkpoint_path, "1", hp.max_num_weights)
    ]

    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=None,           
        callbacks=callback_list,
        initial_epoch=init_epoch,
    )


def test(model, test_data):
    """ Testing routine. """

    model.evaluate(
        x=test_data,
        verbose=1,
    )


def main():
    """ Main function. """

    # path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
    path = "../data"

    print("Path: ", path)
    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    # Run script from location of main.py
    os.chdir(sys.path[0])

    datasets = Datasets(path, "1")
    
    model = YourModel(fourier=ARGS.fourier, 
                      fourier_only=ARGS.fourier_only, 
                      random_fourier=ARGS.random_fourier, 
                      combined=ARGS.combined,
                      combined_random=ARGS.combined_random)

    model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
    checkpoint_path = "checkpoints" + os.sep + \
        "your_model" + os.sep + timestamp + os.sep
    logs_path = "logs" + os.sep + "your_model" + \
        os.sep + timestamp + os.sep

    model.summary()

  
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # compile model graph
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["accuracy"])
    
    train(model, datasets, checkpoint_path, logs_path, init_epoch)


ARGS = parse_args()

main()
#interact -q gpu -g 1 -f ampere -m 96g -n 8 -t 24:00:00
#apptainer shell --nv /oscar/runtime/software/external/ngc-containers/tensorflow.d/x86_64.d/tensorflow-24.03-tf2-py3.simg
#python code/main.py --fourier