# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


""" Define the functions to load data. """
import os
import json
import argparse
from os import makedirs
from os.path import join
import numpy as np
import pickle

def load_data(config):
    # Set the current working directory to the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Current working directory: {os.getcwd()}")
    with open(config.dataset_path, "rb") as input_file:
        data = pickle.load(input_file)
            # for i in range(0, music_seq_len, interval):
            #     music_sub_seq = np_music[i: i + interval]
            #     lighting_sub_seq = np_light_seq[i: i + interval]
            #     if len(music_sub_seq) == interval:
            #         music_data.append(music_sub_seq)
            #         light_seq_data.append(lighting_sub_seq)

    return data['train_data'], data['test_data'], data['val_data'], data['preprocess_config']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
