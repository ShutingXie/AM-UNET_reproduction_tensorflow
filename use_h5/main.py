from data_handler import DataHandler
from model import AMUNet
# from evaluator import Evaluator
from volume_restorer_using_resize_using_padding import VolumeRestorer
# from volume_restorer_using_resize import VolumeRestorer
import numpy as np
import os
from pathlib import Path

# Configuration, change as needed
#################################################################
TEST_DATA_DIR = "../input_data/input_mri"
TEST_MASK_DIR = "../input_data/input_labels"
H5_FILE_PATH = "./axial_model_82.hdf5"
OUTPUT_DIR = "./predictions_Albishri_05mm"
#################################################################

# Constants
ROI_SIZE = (96, 128) 
# THRESHOLD = 0.5

# Create the folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # Initialize components
    data_handler = DataHandler(TEST_DATA_DIR, TEST_MASK_DIR, slice_size=(256, 256), roi_size=ROI_SIZE)
    model = AMUNet(input_shape=(ROI_SIZE[0], ROI_SIZE[1], 1), weights_path=H5_FILE_PATH)
    volume_restorer = VolumeRestorer(data_handler,  resized_size=(256,256), roi_size=ROI_SIZE)

    # Load data
    test_data, test_masks = data_handler.load_data()
    print(f"Loaded test data shape: {test_data.shape}")
    print(f"Loaded test masks shape: {test_masks.shape}")

    # Check if test_data is valid
    if test_data is None or len(test_data) == 0:
        raise ValueError("No test data loaded. Please check the TEST_DATA_DIR path and data format.")

    # Ensure the input shape is correct
    if len(test_data.shape) != 4:  # Expecting (n_samples, height, width, channels)
        raise ValueError(f"Invalid test_data shape: {test_data.shape}. Expected 4 dimensions.")

    # Predict
    print(f"Test data shape: {test_data.shape}")
    test_predictions = model.predict(test_data)
    #################################################################
    # test_predictions[test_predictions >= THRESHOLD] = 1.
    # test_predictions[test_predictions < THRESHOLD] = 0.
    #################################################################
    print(f"Test predictions shape: {test_predictions.shape}")

    # Reconstruct and save 3D volumes
    for subject_id, mri_file in enumerate(data_handler.mri_files):  # Iterate through each subject
        subject_predictions = [test_predictions[idx] for idx, (file_idx, _) in enumerate(data_handler.valid_indices) if file_idx == subject_id] # list of numpu array
        # print(f"subject_predictions1: {type(subject_predictions1[0])}")
        # print(f"subject_predictions: {type(subject_predictions[0])}")
        subject_predictions = np.array(subject_predictions)
        # print(f"\nTwo ways to compute subject predictions: {(subject_predictions1==subject_predictions).all()}")
        
        reconstructed_volume = volume_restorer.reconstruct_3d_volume(subject_predictions, subject_id)
        
        filename = os.path.basename(mri_file)
        filename = Path(filename).stem
        basename = os.path.splitext(filename)[0]
        save_path = os.path.join(OUTPUT_DIR, f"{basename}_pred.nii.gz")
        volume_restorer.save_as_nifti(subject_id, reconstructed_volume, save_path)
        print(f"Saved reconstructed predictions for subject {basename} to {save_path}")
        
        # Doule-check the treshold valye
        # print(f"Threshold: {THRESHOLD}")

if __name__ == "__main__":
    main()
