import os
import glob
import librosa
import numpy as np
import pandas as pd

def normalize_feature(feature):
    if feature.min() < -1.0 or feature.max() > 1.0:
        # Scale to [-1, 1]
        feature = 2.*(feature - feature.min()) / (feature.max() - feature.min()) - 1
    return feature

def process_wav_file(wav_file_path, output_file_path):
    # Load the audio file
    y, sr = librosa.load(wav_file_path, sr=None)

    # Parameters
    hop_length = int(sr / 30)  # 30 frames per second

    # Extract MFCCs and normalize
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length)
    mfccs = np.apply_along_axis(normalize_feature, 1, mfccs)

    # Extract Chroma features and normalize
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=6, hop_length=hop_length)
    chroma = np.apply_along_axis(normalize_feature, 1, chroma)

    # Compute Spectral Flux and normalize
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    spectral_flux = np.diff(onset_env)
    spectral_flux = normalize_feature(spectral_flux)

    # Perform beat tracking and normalize
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    beat_activation = onset_env
    beat_activation = normalize_feature(beat_activation)

    # Initialize Beat_0 array
    beats = np.zeros_like(beat_activation)
    beats[beat_frames] = 1  # Set 1 at frames where beats occur

    # Truncate all features to the shortest length
    min_length = min(mfccs.shape[1], chroma.shape[1], len(spectral_flux), len(beat_activation), len(beats))
    mfccs = mfccs[:, :min_length]
    chroma = chroma[:, :min_length]
    spectral_flux = spectral_flux[:min_length]
    beat_activation = beat_activation[:min_length]
    beats = beats[:min_length]

    # Create DataFrame
    column_names = [f'MFCC_{i}' for i in range(mfccs.shape[0])] + \
                   [f'Chroma_{i}' for i in range(chroma.shape[0])] + \
                   ['Spectralflux_0', 'Beatactivation_0', 'Beat_0']
    df = pd.DataFrame(np.vstack((mfccs, chroma, spectral_flux, beat_activation, beats)).T, columns=column_names)

    # Save the DataFrame as a .pkl file
    df.to_pickle(output_file_path)

def process_directory(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process all WAV files in the input directory and its subdirectories
    for wav_file in glob.glob(os.path.join(input_dir, '**/*.wav'), recursive=True):
        # Construct the output file path
        relative_path = os.path.relpath(wav_file, input_dir)
        base_name = os.path.splitext(relative_path)[0]
        output_file_subdir = os.path.join(output_dir, os.path.dirname(relative_path))
        os.makedirs(output_file_subdir, exist_ok=True)  # Create subdirectory if it doesn't exist
        output_file_path = os.path.join(output_file_subdir, f"{os.path.basename(base_name)}_00.audio29_30fps.pkl")

        # Process the WAV file
        process_wav_file(wav_file, output_file_path)
        print(f"Processed {wav_file} and saved features to {output_file_path}")


# Replace these with your actual input and output directories
input_directory = "../../../Data/Input/Audio/Raw/EDGE_2024-05-29"
output_directory = "../../../Data/Input/LearningDataSets/LDA/tester_audio_pkl"
process_directory(input_directory, output_directory)
