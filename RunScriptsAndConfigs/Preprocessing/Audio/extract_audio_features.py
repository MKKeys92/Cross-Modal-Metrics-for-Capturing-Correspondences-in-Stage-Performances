# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


import argparse
import os
import pyhocon
import librosa
import pickle
import Sources.utils.logger as logger
from Sources.Preprocessing.Audio.audio_extractor import FeatureExtractor

def get_audio_extraction_conf(config_name):
    path = os.path.join(os.path.dirname(__file__), "audio_extraction.conf")
    return pyhocon.ConfigFactory.parse_file(path)[config_name]


def main():
    """ Main function """

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='standard',
                        help='the used audio extraction config')
    parser.add_argument('--input_path', type=str, default="../../../Data/Input/Audio/Raw",
                        help='used input path')
    parser.add_argument('--output_path', type=str, default="../../../Data/Input/Audio/ExtractedData",
                        help='used output path')
    parser.add_argument('--subset_whitelist', nargs='+', default=[], help='Whitelist of subsets to Convert')

    base_args = parser.parse_args()

    config = get_audio_extraction_conf(base_args.config)

    logger.init(None)
    logger.log("Extracting audio features")

    input_path = base_args.input_path
    output_path = base_args.output_path

    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Change the current working directory to the script directory
    os.chdir(script_dir)

    # Print the current working directory to verify
    print("Current working directory set to script directory:", os.getcwd())

    subsets = os.listdir(input_path)

    extractor = FeatureExtractor()

    for subset_name in subsets:
        if base_args.subset_whitelist and subset_name not in base_args.subset_whitelist:
            continue

        ss_path = os.path.join(input_path, subset_name)
        if not os.path.isdir(ss_path):
            continue
        print('subset for audio extraction: ' + str(subset_name))
        files = os.listdir(ss_path)
        for file in files:
            audio_file = os.path.join(input_path, subset_name, file)
            if not os.path.isfile(audio_file):
                continue

            if not audio_file.endswith((".wav", ".m4a", ".mp3", ".wave")):
                continue

            print('file for audio extraction: ' + str(file))
            logger.log(f'Process -> {audio_file}')
            ### load audio ###

            audio = librosa.load(audio_file, sr=config['sampling_rate'])[0]

            d ={}

            d['melspe_db'] = extractor.get_melspectrogram(audio, config)
            d['mfcc'] = extractor.get_mfcc(d['melspe_db'], config)
            d['mfcc_delta'] = extractor.get_mfcc_delta(d['mfcc'], config)
            d['mfcc_delta2'] = extractor.get_mfcc_delta2(d['mfcc'], config)

            audio_harmonic, audio_percussive = extractor.get_hpss(audio, config)
            d['harmonic_melspe_db'] = extractor.get_harmonic_melspe_db(audio_harmonic, config)
            d['percussive_melspe_db'] = extractor.get_percussive_melspe_db(audio_percussive, config)
            d['chroma_cqt'] = extractor.get_chroma_cqt(audio_harmonic, config)
            d['chroma_stft'] = extractor.get_chroma_stft(audio_harmonic, config)

            onset_env = extractor.get_onset_strength(audio_percussive, config)
            d['onset_env'] = onset_env.reshape(1,-1)
            d['tempogram'] = extractor.get_tempogram(onset_env, config)
            d['onset_beat'] = extractor.get_onset_beat(onset_env, config)
            d['rms'] = extractor.get_rms(audio, config)

            for k,v in d.items():
                t = v.transpose(1, 0)
                d[k] = t

            d['tempo'] = extractor.get_tempo(audio, config)

            abs_path = os.path.join(output_path, base_args.config)
            isExist = os.path.exists(abs_path)
            if not isExist:
                os.makedirs(abs_path)

            ss_out_path = os.path.join(abs_path,subset_name)
            isExist = os.path.exists(ss_out_path)
            if not isExist:
                os.makedirs(ss_out_path)

            file_short = os.path.splitext(file)[0]

            sp = os.path.join(abs_path, subset_name,  file_short + '.pkl')

            with open(sp, 'wb+') as fs:
                pickle.dump(d, fs)

if __name__ == '__main__':
    main()

