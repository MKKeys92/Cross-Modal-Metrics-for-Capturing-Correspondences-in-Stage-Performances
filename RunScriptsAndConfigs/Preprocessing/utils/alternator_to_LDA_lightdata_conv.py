import numpy as np
import os
import glob
import pickle
import pandas as pd

# Source directory containing .pkl files to process
source_dir = "../../../Data/Input/Light/AbstractedData/PASv02/RF2023"

# Output directory to save the processed .npy files
output_dir = "../../../Data/Input/LearningDataSets/LDA/RF2023"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)


def process_pkl_file(input_file_path, output_dir):
    # Load the data from the source .pkl file
    with open(input_file_path, 'rb') as file:
        data = pickle.load(file)

    # Convert to numpy array if it's not already
    if isinstance(data, np.ndarray):
        data_np = data
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        data_np = data.values
    else:
        raise ValueError(f"Unsupported data type in {input_file_path}")

    # Ensure that the numpy array has at least 60 features
    if data_np.shape[1] < 60:
        raise ValueError(f"The data in {input_file_path} does not have enough features.")

    # Slice the numpy array to retain the first 60 features
    data_np_sliced = data_np[:, :60]

    # Add a 61st parameter of zeros
    num_frames = data_np_sliced.shape[0]
    data_np_with_dummy = np.hstack((data_np_sliced, np.zeros((num_frames, 1))))

    # Ensure all values are between 0.0 and 1.0
    data_np_normalized = np.clip(data_np_with_dummy, 0.0, 1.0)

    # Construct the output file path
    output_file_name = os.path.splitext(os.path.basename(input_file_path))[0] + '_for_LDA_preprocessing.npy'
    output_file_path = os.path.join(output_dir, output_file_name)

    # Save the processed numpy array as a .npy file
    np.save(output_file_path, data_np_normalized)
    print(f"Saved processed array to {output_file_path}")
    print(f"Shape of the saved array: {data_np_normalized.shape}")


# Process all .pkl files in the source directory
for input_file_path in glob.glob(os.path.join(source_dir, '*.pkl')):
    process_pkl_file(input_file_path, output_dir)

print("Processing complete.")