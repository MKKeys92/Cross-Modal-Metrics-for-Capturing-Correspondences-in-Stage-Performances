# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


import librosa
import numpy as np
import os

class FeatureExtractor:
    @staticmethod
    def get_melspectrogram(audio, config):
        melspe = librosa.feature.melspectrogram(y=audio, sr=config['sampling_rate'], hop_length = config['hop_length'], n_fft= config['window_size'])
        melspe_db = librosa.power_to_db(melspe, ref=np.max)
        # print(f'{melspe_db.shape} -> melspe_db')
        return melspe_db

    @staticmethod
    def get_hpss(audio, config):
        audio_harmonic, audio_percussive = librosa.effects.hpss(audio)
        return audio_harmonic, audio_percussive

    @staticmethod
    def get_mfcc(melspe_db, config):
        mfcc = librosa.feature.mfcc(S=melspe_db)
        print(f'{mfcc.shape} -> mfcc')
        return mfcc

    @staticmethod
    def get_mfcc_delta(mfcc, config):
        mfcc_delta = librosa.feature.delta(mfcc, width=3)
        print(f'{mfcc_delta.shape} -> mfcc_delta')
        return mfcc_delta

    @staticmethod
    def get_mfcc_delta2(mfcc, config):
        mfcc_delta_delta = librosa.feature.delta(mfcc, width=3, order=2)
        print(f'{mfcc_delta_delta.shape} -> mfcc_delta_delta')
        return mfcc_delta_delta

    @staticmethod
    def get_harmonic_melspe_db(audio_harmonic, config):
        harmonic_melspe = librosa.feature.melspectrogram(y=audio_harmonic, sr=config['sampling_rate'], hop_length = config['hop_length'], n_fft= config['window_size'])
        harmonic_melspe_db = librosa.power_to_db(harmonic_melspe, ref=np.max)
        print(f'{harmonic_melspe_db.shape} -> harmonic_melspe_db')
        return harmonic_melspe_db

    @staticmethod
    def get_percussive_melspe_db(audio_percussive, config):
        percussive_melspe = librosa.feature.melspectrogram(y=audio_percussive, sr=config['sampling_rate'], hop_length = config['hop_length'], n_fft= config['window_size'])
        percussive_melspe_db = librosa.power_to_db(percussive_melspe, ref=np.max)
        print(f'{percussive_melspe_db.shape} -> percussive_melspe_db')
        return percussive_melspe_db

    @staticmethod
    def get_chroma_cqt(audio_harmonic, config):
        chroma_cqt_harmonic = librosa.feature.chroma_cqt(y=audio_harmonic, sr=config['sampling_rate'], hop_length = config['hop_length'])
        print(f'{chroma_cqt_harmonic.shape} -> chroma_cqt_harmonic')
        return chroma_cqt_harmonic

    @staticmethod
    def get_chroma_stft(audio_harmonic, config):
        chroma_stft_harmonic = librosa.feature.chroma_stft(y=audio_harmonic, sr=config['sampling_rate'], hop_length = config['hop_length'], n_fft= config['window_size'])
        print(f'{chroma_stft_harmonic.shape} -> chroma_stft_harmonic')
        return chroma_stft_harmonic

    @staticmethod
    def get_tonnetz(audio_harmonic, config):
        tonnetz = librosa.feature.tonnetz(y=audio_harmonic, sr=config['sampling_rate'], hop_length = config['hop_length'], n_fft= config['window_size'])
        print(f'{tonnetz.shape} -> tonnetz')
        return tonnetz

    @staticmethod
    def get_onset_strength(audio_percussive, config):
        onset_env = librosa.onset.onset_strength(y=audio_percussive, aggregate=np.median, sr=config['sampling_rate'], hop_length = config['hop_length'], n_fft= config['window_size'])
        print(f'{onset_env.reshape(1, -1).shape} -> onset_env')
        return onset_env

    @staticmethod
    def get_tempogram(onset_env, config):
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=config['sampling_rate'])
        print(f'{tempogram.shape} -> tempogram')
        return tempogram

    @staticmethod
    def get_onset_beat(onset_env, config):
        onset_tempo, onset_beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=config['sampling_rate'])
        beats_one_hot = np.zeros(len(onset_env))
        for idx in onset_beats:
            beats_one_hot[idx] = 1
        beats_one_hot = beats_one_hot.reshape(1, -1)
        print(f'{beats_one_hot.shape} -> beats_feature')
        return beats_one_hot

    @staticmethod
    def get_rms(audio, config):
        S, phase = librosa.magphase(librosa.stft(audio, hop_length = config['hop_length'], n_fft= config['window_size']))
        rms = librosa.feature.rms(S=S, hop_length=config['hop_length'],frame_length=config['window_size'])
        return rms

    @staticmethod
    def get_tempo(audio, config):
        onset_env = librosa.onset.onset_strength(y=audio, sr=config['sampling_rate'])
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=config['sampling_rate'])
        return tempo[0]


# def extract_acoustic_feature(input_audio_dir, config):
#     extractor = FeatureExtractor()
#
#     print('---------- Extract features from raw audio ----------')
#     musics = []
#     waveforms = []
#     fnames = sorted(os.listdir(input_audio_dir))
#
#     audio_fnames = []
#     for audio_fname in fnames:
#         if not audio_fname.endswith((".wav",".m4a",".mp3",".wave")):
#             continue
#         audio_fnames.append(audio_fname)
#         audio_file = os.path.join(input_audio_dir, audio_fname)
#         print(f'Process -> {audio_file}')
#         ### load audio ###
#
#         audio = librosa.load(audio_file, sr=config['sampling_rate'])[0]
#
#         melspe_db = extractor.get_melspectrogram(audio, config)
#         mfcc = extractor.get_mfcc(melspe_db, config)
#         mfcc_delta = extractor.get_mfcc_delta(mfcc, config)
#         mfcc_delta2 = extractor.get_mfcc_delta2(mfcc, config)
#
#         audio_harmonic, audio_percussive = extractor.get_hpss(audio, config)
#         harmonic_melspe_db = extractor.get_harmonic_melspe_db(audio_harmonic, config)
#         percussive_melspe_db = extractor.get_percussive_melspe_db(audio_percussive, config)
#         chroma_cqt = extractor.get_chroma_cqt(audio_harmonic, config)
#         #  # orignal comment out // chroma_stft = extractor.get_chroma_stft(audio_harmonic, sr)
#         chroma_stft = extractor.get_chroma_stft(audio_harmonic, config)
#
#         onset_env = extractor.get_onset_strength(audio_percussive, config)
#         tempogram = extractor.get_tempogram(onset_env, config)
#         onset_beat = extractor.get_onset_beat(onset_env, config)
#         #  # orignal comment out // onset_tempo, onset_beat = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
#         #  # orignal comment out // onset_beats.append(onset_beat)
#
#         onset_env = onset_env.reshape(1, -1)
#
#         feature = np.empty((0, onset_beat.shape[1]))
#
#         if 'melspe_db' in config['audio_features']:
#             feature = np.concatenate([feature, melspe_db], axis=0)
#         if 'mfcc' in config['audio_features']:
#             feature = np.concatenate([feature, mfcc], axis=0)
#         if 'mfcc_delta' in config['audio_features']:
#             feature = np.concatenate([feature, mfcc_delta], axis=0)
#         if 'mfcc_delta2' in config['audio_features']:
#             feature = np.concatenate([feature, mfcc_delta2], axis=0)
#         if 'harmonic_melspe_db' in config['audio_features']:
#             feature = np.concatenate([feature, harmonic_melspe_db], axis=0)
#         if 'percussive_melspe_db' in config['audio_features']:
#             feature = np.concatenate([feature, percussive_melspe_db], axis=0)
#         if 'chroma_stft' in config['audio_features']:
#             feature = np.concatenate([feature, chroma_stft], axis=0)
#         if 'chroma_cqt' in config['audio_features']:
#             feature = np.concatenate([feature, chroma_cqt], axis=0)
#         if 'onset_env' in config['audio_features']:
#             feature = np.concatenate([feature, onset_env], axis=0)
#         if 'tempogram' in config['audio_features']:
#             feature = np.concatenate([feature, tempogram], axis=0)
#         if 'onset_beat' in config['audio_features']:
#             feature = np.concatenate([feature, onset_beat], axis=0)
#
#
#
#
#
#         feature = feature.transpose(1, 0)
#         print(f'acoustic feature -> {feature.shape}')
#         musics.append(feature.tolist())
#         waveforms.append(audio)
#
#     return musics, audio_fnames, waveforms