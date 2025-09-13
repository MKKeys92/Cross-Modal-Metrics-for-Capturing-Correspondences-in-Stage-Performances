import numpy as np
import pickle
import os
import json
from scipy.signal import butter, filtfilt
import scipy.io as sio
import tkinter as tk
from tkinter import filedialog


def lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    while order > 0:
        try:
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            y = filtfilt(b, a, data, axis=0)
            return y
        except ValueError as e:
            if str(e) == "The length of the input vector x must be greater than padlen, which is {padlen}":
                order -= 1
            else:
                raise
    return data


def process_timeseries_with_window(timeseries, fs, cutoff, window_size=1800):
    adjusted_timeseries = np.zeros_like(timeseries)
    step_size = window_size

    for i in range(timeseries.shape[1]):
        for start in range(0, timeseries.shape[0], step_size):
            end = start + window_size
            if end > timeseries.shape[0]:
                end = timeseries.shape[0]
            ts = timeseries[start:end, i]

            delta = np.max(ts) - np.min(ts)

            if delta < 0.1:
                # New processing mode for small delta
                mean = np.mean(ts)
                adjusted_ts = ts - mean + 0.11  # Position the series to have a mean of 0.11
                adjusted_ts = np.clip(adjusted_ts, 0.0, 1.0)
            elif np.min(ts) >= -0.2 and np.max(ts) <= 1.2:
                adjusted_ts = np.clip(ts, 0.0, 1.0)
            elif np.min(ts) < -0.2 or np.max(ts) > 1.2:
                # Scale and offset the complete time series
                adjusted_ts = (ts - np.min(ts)) / delta
                adjusted_ts = np.clip(adjusted_ts, 0.0, 1.0)
            else:
                trend = lowpass_filter(ts, cutoff, fs)
                adjusted_ts = ts - trend + 0.5
                adjusted_ts = np.clip(adjusted_ts, 0.0, 1.0)

            adjusted_timeseries[start:end, i] = adjusted_ts

    return adjusted_timeseries


def load_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.npy':
        data = np.load(file_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.shape == ():
            data = data.item()
    elif file_extension == '.pkl':
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    elif file_extension == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    if isinstance(data, dict):
        if 'pred_position' in data:
            return data['pred_position']
        else:
            return np.array(list(data.values())[0])
    elif isinstance(data, list):
        return np.array(data)
    else:
        return data


def extract_and_save_npy_to_pkl_and_mat(input_folder, fs=30, cutoff=0.5):
    for filename in os.listdir(input_folder):
        if filename.endswith(('.pkl', '.npy', '.json')):
            file_path = os.path.join(input_folder, filename)
            try:
                data = load_file(file_path)

                if not isinstance(data, np.ndarray):
                    raise ValueError("Loaded data is not a numpy array")

                processed_data = process_timeseries_with_window(data, fs, cutoff)

                output_filename_base = os.path.splitext(filename)[0]
                output_pkl_path = os.path.join(input_folder, output_filename_base + '_processed.pkl')
                output_mat_path = os.path.join(input_folder, output_filename_base + '_processed.mat')
                raw_output_mat_path = os.path.join(input_folder, output_filename_base + '_raw.mat')

                with open(output_pkl_path, 'wb') as output_file:
                    pickle.dump(processed_data, output_file)

                sio.savemat(output_mat_path, {'data': processed_data})
                # sio.savemat(raw_output_mat_path, {'raw_data': data})

                print(f"Saved {output_pkl_path} and {output_mat_path}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")


# GUI for folder selection
root = tk.Tk()
root.withdraw()  # Hide the main window

input_folder_path = filedialog.askdirectory(title="Select Input Folder")

if input_folder_path:
    extract_and_save_npy_to_pkl_and_mat(input_folder_path)
else:
    print("No folder selected. Exiting.")