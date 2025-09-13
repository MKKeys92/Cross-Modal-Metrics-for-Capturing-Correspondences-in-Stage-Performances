from os import makedirs
import os.path
import numpy as np
import pyhocon
import Sources.utils.logger as logger
import json
import pickle


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def initialize_config(config_name, name_suffix):
    config = pyhocon.ConfigFactory.parse_file(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'RunScriptsAndConfigs', 'Training', 'models.conf'))[config_name]
    config.name = config_name
    config.name_suffix = name_suffix

    logger.log.setup(config)
    logger.log("Running experiment: {}".format(config_name))
    #config['log_dir'] = join(config["log_root"], config_name)
    #makedirs(config['log_dir'], exist_ok=True)

    #config['tb_dir'] = join(config['log_root'], 'tensorboard')
    #makedirs(config['tb_dir'], exist_ok=True)

    logger.log(pyhocon.HOCONConverter.convert(config, "hocon"))
    return config

def load_startup_light_data(dir, audio_fnames, predict_args, config):
    logger.log('loading startup light data')
    l_data = []
    for name in audio_fnames:
        if name.endswith('m4a') or name.endswith('wav'):
            name = name[:-3]
            name = name + "json"
            with open(os.path.join(dir, name)) as f:
                sample_dict = json.loads(f.read())
                np_light_seq = np.array(sample_dict['lighting_array'])
                if np_light_seq.shape[0] < predict_args.used_startup_frames:
                    raise ValueError("light seq to short" + name)
                if np_light_seq.shape[1] != config.prepro_config['lighting_dim']:
                    raise ValueError("light seq has wrong dimension" + name)
                l_data.append(np_light_seq)
        else:
            raise TypeError("unable to handle audio file " + name)
    return l_data

def split_dataset_in_chunks(config, dataset, max_seq_len, dataset_config, load_audio_features = False):
    split_data = []
    for v in dataset:
        l = v['lighting_array']
        a = v['music_array']

        if config.partitioning_algo == "WindowOrLoop":
            n = l.shape[0]

            if load_audio_features:
                fp = os.path.join(dataset_config.audio_input_dir, dataset_config.audio_extraction_config, v['subset'],
                                  v['id'] + ".pkl")
                with open(fp, 'rb') as f:
                    audio_data = pickle.load(f)

            #case loop
            if n < max_seq_len:
                l_loop = np.concatenate((l,l[:(max_seq_len-n),:]))
                a_loop = np.concatenate((a,a[:(max_seq_len-n),:]))

                dp = v.copy()
                dp['lighting_array'] = l_loop
                dp['music_array'] = a_loop
                if load_audio_features:
                    feature_dic = {}
                    for k in audio_data.keys():
                        feature = audio_data[k]
                        feature = feature[:l.shape[0], :]
                        feature_loop =  np.concatenate((feature,feature[:(max_seq_len-n),:]))
                        feature_dic[k] = feature_loop
                    dp['all_audio_features'] = feature_dic

                split_data.append(dp)

            else:
                split_points = []
                c = 1
                while c * max_seq_len < n:
                    split_points.append(c * max_seq_len)
                    c += 1

                s_l = np.split(l, split_points, axis=0)
                s_l.pop()
                s_l.append(l[-max_seq_len:, :])

                a_l = np.split(a, split_points, axis=0)
                a_l.pop()
                a_l.append(a[-max_seq_len:, :])

                audio_feature_l ={}

                if load_audio_features:
                    for k in audio_data.keys():
                        feature = audio_data[k]
                        feature = feature[:l.shape[0], :]
                        audio_feature_l[k] = np.split(feature, split_points, axis=0)
                        audio_feature_l[k].pop()
                        audio_feature_l[k].append(feature[-max_seq_len:, :])

                for i in range(len(s_l)):
                    dp = v.copy()
                    dp['lighting_array'] = s_l[i]
                    dp['music_array'] = a_l[i]
                    if load_audio_features:
                        feature_dic = {}
                        for k in audio_data.keys():
                            feature_l = audio_feature_l[k]
                            feature_dic[k] = feature_l[i]
                        dp['all_audio_features'] = feature_dic

                    split_data.append(dp)
        else:
            raise NotImplementedError


    return split_data

def add_all_audio_features(dataset, dataset_config):
    for v in dataset:
        l = v['lighting_array']
        n=l.shape[0]
        fp = os.path.join(dataset_config.audio_input_dir, dataset_config.audio_extraction_config, v['subset'],
                          v['id'] + ".pkl")
        with open(fp, 'rb') as f:
            audio_data = pickle.load(f)

        for k in audio_data.keys():
            if k != 'tempo':
                audio_data[k] = audio_data[k][:n,:]

        v['all_audio_features'] = audio_data
    return