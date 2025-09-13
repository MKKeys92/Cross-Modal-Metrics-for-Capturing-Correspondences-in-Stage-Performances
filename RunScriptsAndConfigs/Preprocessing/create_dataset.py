# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


import os
import json
import random
import argparse
import numpy as np
import pyhocon
from datetime import datetime
import pickle
from Sources.Evaluation.data_generator import DataGenerator
import Sources.utils.logger as logger


def load_light_data(light_dir):
    print('---------- Loading light data ----------')
    light_sequences = []
    fnames = sorted(os.listdir(light_dir))
    # dir_names = dir_names[:20]  # for debug
    print(f'light seq file names: {fnames}')
    # fnames = fnames[:60]  # For debug
    for fname in fnames:
        path = os.path.join(light_dir, fname)
        with open(path) as f:
            sample_dict = json.loads(f.read())
            np_light_seq = np.array(sample_dict['lighting_array'])
            light_sequences.append(np_light_seq)

    return light_sequences


def align(music, light_sequence):
    print('---------- Align the frames of music and dance ----------')

    min_seq_len = min(len(music), len(light_sequence))

    #assert len(music)>= len(light_sequence), "music should never be shorter than light seq"

    music = np.array(music[:min_seq_len])
    #waveform = np.array(waveform[:(min_seq_len*config.hop_length-1)])
    light_sequence = light_sequence[:min_seq_len, :]

    return music, light_sequence


def split_data(data, args):
    print('---------- Calculating split indices----------')

    assert args.test_split + args.train_split + args.val_split == 1, "Split config not supported. sum of split values not equal to 1!"

    indices = list(range(len(data)))
    random.shuffle(indices)
    n = len(indices)
    train_idx = indices[:int(n * args.train_split)]
    val_idx = indices[int(n * args.train_split) : int(n * args.train_split) + int(n * args.val_split)]
    test_idx = indices[int(n * args.train_split) + int(n * args.val_split):]

    return train_idx, test_idx, val_idx


def search_for_music(musics, names, light_name):
    for i in range(len(names)):
        if light_name in names[i]:
            return i

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='base')
    parser.add_argument('--light_input_dir', type=str, default="../../Data/Input/Light/AbstractedData")
    parser.add_argument('--audio_input_dir', type=str, default="../../Data/Input/Audio/ExtractedData")
    parser.add_argument('--dataset_output_dir', type=str, default="../../Data/Input/CombinedDataSets")

    base_args = parser.parse_args()

    config = pyhocon.ConfigFactory.parse_file("dataset.conf")[base_args.config]
    config['name'] = base_args.config
    config['audio_input_dir'] = base_args.audio_input_dir
    config['light_input_dir'] = base_args.light_input_dir
    config['dataset_output_dir'] = base_args.dataset_output_dir

    if not os.path.exists("datasets"):
        os.makedirs("datasets", exist_ok=True)

    logger.init(None)

    use_data_gen = False
    if config.dim_gen or config.hue_gen or config.sat_gen or config.tilt_gen or config.pan_gen or config.base_gen :
        use_data_gen = True

    dataset = {}
    data = []

    audio_input_path = os.path.join(config.audio_input_dir, config.audio_extraction_config)
    light_input_path = os.path.join(config.light_input_dir, config.lighting_abstraction_layer)
    output_path = config.dataset_output_dir

    subsets = os.listdir(audio_input_path)

    for subset_name in subsets:
        #skip datasets if needed
        if config.datasets:
            if not subset_name in config.datasets:
                continue

        ss_path = os.path.join(audio_input_path, subset_name)
        if not os.path.isdir(ss_path):
            continue
        files = os.listdir(ss_path)
        for file in files:
            fp = os.path.join(audio_input_path, subset_name, file)
            if not os.path.isfile(fp):
                continue
            if not file.endswith(".pkl"):
                continue

            with open(fp, 'rb') as f:
                audio_data = pickle.load(f)

            #concatenating audio features:
            audio_features = np.empty((audio_data[config.audio_features[0]].shape[0],0))

            for feature in config.audio_features:
                audio_features = np.concatenate([audio_features, audio_data[feature]], axis=1)

            name = os.path.splitext(file)[0]

            #loading light seq
            lp = os.path.join(light_input_path, subset_name, file)
            with open(lp, 'rb') as fl:
                light_seq = pickle.load(fl)

            final_music, final_light_seq = align(audio_features, light_seq)

            if use_data_gen:
                DataGen = DataGenerator(config, config.lighting_abstraction_layer)

                final_light_seq = DataGen.generate_or_modify_data(final_light_seq, audio_data)

            sample_dict = {
                'id': name,
                'subset': subset_name,
                'music_array': final_music,
                'lighting_array': final_light_seq
            }
            data.append(sample_dict)



    train_idx, test_idx, val_idx = split_data(data, config)
    train_idx = sorted(train_idx)
    logger.log(f'train ids: {[idx for idx in train_idx]}')
    test_idx = sorted(test_idx)
    logger.log(f'test ids: {[idx for idx in test_idx]}')
    val_idx = sorted(val_idx)
    logger.log(f'val ids: {[idx for idx in val_idx]}')

    config['music_dim'] = len(data[0]['music_array'][0])
    config['lighting_dim'] = len(data[0]['lighting_array'][0])

    dataset['preprocess_config'] = config
    dataset['train_data'] = []
    dataset['test_data'] = []
    dataset['val_data'] = []

    for idx in range(len(data)):
        if idx in train_idx:
            dataset['train_data'].append(data[idx])
        if idx in test_idx:
            dataset['test_data'].append(data[idx])
        if idx in val_idx:
            dataset['val_data'].append(data[idx])

    date_suffix = datetime.now().strftime('%b%d_%H-%M-%S')
    dataset_name = config.name + '_' + date_suffix

    isExist = os.path.exists(output_path)
    if not isExist:
        os.makedirs(output_path)

    sp = os.path.join(output_path, dataset_name + '.pkl')

    with open(sp,"wb") as fp:
        pickle.dump(dataset, fp)

if __name__ == '__main__':
    main()

