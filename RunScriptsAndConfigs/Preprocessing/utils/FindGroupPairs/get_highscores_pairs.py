import numpy as np
import os
import random
import pandas as pd
from scipy.linalg import sqrtm
import pickle
from scipy.stats import skew, kurtosis

def calculate_fid(act1, act2, epsilon=1e-6):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    sigma1 += np.eye(sigma1.shape[0]) * epsilon
    sigma2 += np.eye(sigma2.shape[0]) * epsilon
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)


def interpolate_to_common_length(series, target_length):
    # Check if the series is indeed 2D: [time, features]
    if series.ndim != 2:
        raise ValueError("Data must be 2D for interpolation.")

    # Interpolate each feature across the time dimension
    num_features = series.shape[1]
    interpolated_series = np.zeros((target_length, num_features))

    for i in range(num_features):
        # Interpolate each feature separately
        feature_series = series[:, i]
        interpolated_series[:, i] = np.interp(
            np.linspace(0, len(feature_series) - 1, num=target_length),
            np.arange(len(feature_series)),
            feature_series
        )

    return interpolated_series


def calculate_interpolated_mean_distance(group_A, group_B):
    # Determine the target length as the maximum of both groups' lengths
    target_length = max(group_A.shape[0], group_B.shape[0])

    group_A_resampled = interpolate_to_common_length(group_A, target_length)
    group_B_resampled = interpolate_to_common_length(group_B, target_length)

    # Calculate the mean of the absolute differences
    return np.mean(np.abs(group_A_resampled - group_B_resampled))


def extract_features(series):
    # Ensure the input is a numpy array and is flat
    series = np.asarray(series).flatten()

    features = {
        'mean': np.mean(series),
        'std_dev': np.std(series),
        'min': np.min(series),
        'max': np.max(series),
        'skewness': skew(series),
        'kurtosis': kurtosis(series),
        'quantile_25': np.quantile(series, 0.25),
        'quantile_50': np.quantile(series, 0.50),
        'quantile_75': np.quantile(series, 0.75)
    }
    # Convert features to a list of values, ensuring all are scalars
    feature_values = [features[key] for key in features]
    return np.array(feature_values)  # This should now work without error


def calculate_feature_distance(group_A, group_B):
    # Extract features for each group
    features_A = extract_features(group_A)
    features_B = extract_features(group_B)

    # Calculate Euclidean distance between feature sets
    return np.linalg.norm(features_A - features_B)


def load_random_arrays(folder, n=1):
    files = os.listdir(folder)
    selected_files = random.sample(files, n)
    arrays = [pickle.load(open(os.path.join(folder, f), 'rb')) for f in selected_files]
    return arrays, selected_files

def extract_and_concatenate_groups(arrays, group_definitions):
    groups = {}
    for group_name, indices in group_definitions.items():
        groups[group_name] = np.concatenate([arr[:, indices[0]:indices[1]] for arr in arrays], axis=0)
    return groups

def main():
    folder_A = "../../../Data/Input/Light/AbstractedData/PASv02/RF2023"
    folder_B_list = ["../../../Data/Input/Light/AbstractedData/PASv02/DK2020"]
    output_folder = "../../../Data/Input/Light/AbstractedData/PASv02/Results_Pair_Search/RF_to_DK"
    os.makedirs(output_folder, exist_ok=True)
    group_definitions = {f'Section_{chr(65 + i)}': [i * 6, (i + 1) * 6] for i in range(10)}
    num_runs = 96

    for folder_B in folder_B_list:
        for _ in range(num_runs):
            arrays_A, files_A = load_random_arrays(folder_A)
            arrays_B, files_B = load_random_arrays(folder_B)
            groups_A = extract_and_concatenate_groups(arrays_A, group_definitions)
            groups_B = extract_and_concatenate_groups(arrays_B, group_definitions)

            results = []
            for group_name_A, group_A in groups_A.items():
                for group_name_B, group_B in groups_B.items():
                    # print(group_B)
                    # print(files_B)
                    fid_score = calculate_fid(group_A, group_B)
                    mean_distance = calculate_interpolated_mean_distance(group_A, group_B)
                    feature_distance = calculate_feature_distance(group_A, group_B)
                    results.append({
                        'Group A': f'{group_name_A} ({", ".join(files_A)})',
                        'Group B': f'{group_name_B} ({", ".join(files_B)})',
                        'FID': fid_score,
                        'Mean Distance': mean_distance,
                        'Feature Distance': feature_distance
                    })

            files_A_names = '_'.join([os.path.splitext(f)[0] for f in files_A])
            files_B_names = '_'.join([os.path.splitext(f)[0] for f in files_B])
            csv_filename = f'fid_leaderboard_{files_A_names}_vs_{files_B_names}.csv'
            csv_file_path = os.path.join(output_folder, csv_filename)
            print(csv_filename)
            results_df = pd.DataFrame(results)
            results_df.sort_values(by='FID', inplace=True)
            results_df.to_csv(csv_file_path, index=False)


    folder_B_list = ["../../../Data/Input/Light/AbstractedData/PASv02/HAW2021"]
    output_folder = "../../../Data/Input/Light/AbstractedData/PASv02/Results_Pair_Search/RF_to_HAW"
    os.makedirs(output_folder, exist_ok=True)
    group_definitions = {f'Section_{chr(65 + i)}': [i * 6, (i + 1) * 6] for i in range(10)}
    num_runs = 96

    for folder_B in folder_B_list:
        for _ in range(num_runs):
            arrays_A, files_A = load_random_arrays(folder_A)
            arrays_B, files_B = load_random_arrays(folder_B)
            groups_A = extract_and_concatenate_groups(arrays_A, group_definitions)
            groups_B = extract_and_concatenate_groups(arrays_B, group_definitions)

            results = []
            for group_name_A, group_A in groups_A.items():
                for group_name_B, group_B in groups_B.items():
                    fid_score = calculate_fid(group_A, group_B)
                    mean_distance = calculate_interpolated_mean_distance(group_A, group_B)
                    feature_distance = calculate_feature_distance(group_A, group_B)
                    results.append({
                        'Group A': f'{group_name_A} ({", ".join(files_A)})',
                        'Group B': f'{group_name_B} ({", ".join(files_B)})',
                        'FID': fid_score,
                        'Mean Distance': mean_distance,
                        'Feature Distance': feature_distance
                    })

            files_A_names = '_'.join([os.path.splitext(f)[0] for f in files_A])
            files_B_names = '_'.join([os.path.splitext(f)[0] for f in files_B])
            csv_filename = f'fid_leaderboard_{files_A_names}_vs_{files_B_names}.csv'
            csv_file_path = os.path.join(output_folder, csv_filename)
            print(csv_filename)
            results_df = pd.DataFrame(results)
            results_df.sort_values(by='FID', inplace=True)
            results_df.to_csv(csv_file_path, index=False)


    folder_B_list = ["../../../Data/Input/Light/AbstractedData/PASv02/NC2023"]
    output_folder = "../../../Data/Input/Light/AbstractedData/PASv02/Results_Pair_Search/RF_to_NC"
    os.makedirs(output_folder, exist_ok=True)
    group_definitions = {f'Section_{chr(65 + i)}': [i * 6, (i + 1) * 6] for i in range(10)}
    num_runs = 96

    for folder_B in folder_B_list:
        for _ in range(num_runs):
            arrays_A, files_A = load_random_arrays(folder_A)
            arrays_B, files_B = load_random_arrays(folder_B)
            groups_A = extract_and_concatenate_groups(arrays_A, group_definitions)
            groups_B = extract_and_concatenate_groups(arrays_B, group_definitions)

            results = []
            for group_name_A, group_A in groups_A.items():
                for group_name_B, group_B in groups_B.items():
                    fid_score = calculate_fid(group_A, group_B)
                    mean_distance = calculate_interpolated_mean_distance(group_A, group_B)
                    feature_distance = calculate_feature_distance(group_A, group_B)
                    results.append({
                        'Group A': f'{group_name_A} ({", ".join(files_A)})',
                        'Group B': f'{group_name_B} ({", ".join(files_B)})',
                        'FID': fid_score,
                        'Mean Distance': mean_distance,
                        'Feature Distance': feature_distance
                    })

            files_A_names = '_'.join([os.path.splitext(f)[0] for f in files_A])
            files_B_names = '_'.join([os.path.splitext(f)[0] for f in files_B])
            csv_filename = f'fid_leaderboard_{files_A_names}_vs_{files_B_names}.csv'
            csv_file_path = os.path.join(output_folder, csv_filename)
            print(csv_filename)
            results_df = pd.DataFrame(results)
            results_df.sort_values(by='FID', inplace=True)
            results_df.to_csv(csv_file_path, index=False)

if __name__ == '__main__':
    main()
