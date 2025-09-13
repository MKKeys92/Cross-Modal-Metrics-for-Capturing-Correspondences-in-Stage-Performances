import json
import numpy as np



path = "/home/michaelk/PycharmProjects/alternator_v1.1/preprocessing/raw_data/Deichkind_Raw/light/DK2020-22-OMAA01.json"
path_old = "/home/michaelk/Development/Alternator_v1_2022-04-26/data_prepare/train_1min/etc_features_ID02MR01_high01_0.json"


if __name__ == '__main__':
    with open(path) as f:
        sample_dict = json.loads(f.read())
        np_light_seq = np.array(sample_dict['lighting_array'])
        # np_audio_seq = np.array(sample_dict['music_array'])
        # test_data = np_light_seq[:,0]
        print("done")

    # with open(path_old) as f_old:
    #     sample_dict = json.loads(f_old.read())
    #     old_data = np.array(sample_dict['dance_array'])
    #     old_audio_data = np.array(sample_dict['music_array'])
    #     test_data_old = old_data[:,0]
    #     print("done")

