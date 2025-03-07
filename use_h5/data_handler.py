import nibabel as nib
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt

class DataHandler:
    # For MRI image, the dimension is (x,y,z), which means (width, height, slcies)
    # For ROI, 96 is height, 128 is width
    def __init__(self, data_dir, mask_dir, slice_size=(256, 256), roi_size=(96, 128)):
        """
        Initialize the data handler.
        """
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.slice_size = slice_size 
        self.roi_size = roi_size

        # List and sort data and mask files
        self.mri_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".nii.gz") or f.endswith(".nii")])
        self.mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".nii.gz") or f.endswith(".nii")])
        
        # Assert the number of MRI and mask files match
        assert len(self.mri_files) == len(self.mask_files), "Number of MRI and mask files must match."

        # Get slices per image
        self.slices_per_image = self._get_slices_per_image()
        print(f"The number of slices per image: {self.slices_per_image}")

        # Get the original size of each MRI (the slice size after transposition is (height, width))
        self.original_sizes = self._get_original_sizes()  # (H, W, #slices)
        print(f"Original slice size of each subject (height, width, number of slices): {self.original_sizes}")

        # Filter valid slices containing the claustrum
        self.valid_indices = self._filter_valid_slices()
        # print(f"Valid slices: {self.valid_indices}")
        print(f"The total number of valid slices: {len(self.valid_indices)}")

    def _get_slices_per_image(self):
        """Return a list with the number of slices per MRI file."""
        slices_per_image = []
        for data_file in self.mri_files:
            data = nib.load(data_file).get_fdata()
            slices_per_image.append(data.shape[2])
        return slices_per_image

    def _get_original_sizes(self):
        """
        Get the original slice size of each MRI.
        Note that the original data shape is (width, height, number of slices), and the slice size after transposition is (height, width).
        """
        sizes = []
        for data_file in self.mri_files:
            data = nib.load(data_file).get_fdata()  # shape: (W, H, #slices)
            # The reason why reordering the size:
            # 
            sizes.append((data.shape[1], data.shape[0], data.shape[2]))  # (H, W, #slices)
        return sizes

    def _filter_valid_slices(self):
        """
        Filter out slices that do not contain the claustrum in the masks.
        """
        valid_indices = []
        for mask_idx, mask_file in enumerate(self.mask_files):
            mask_data = nib.load(mask_file).get_fdata()
            for slice_idx in range(self.slices_per_image[mask_idx]):
                mask_slice = mask_data[:, :, slice_idx]
                # Check if the slice contains the claustrum
                if np.any(mask_slice):  # Non-empty mask
                    valid_indices.append((mask_idx, slice_idx))
        return valid_indices

    def __len__(self):
        """Return the total number of valid slices."""
        return len(self.valid_indices)

    def min_max_normalize(self, image):
        """
        Min-max normalize an image.
        """
        return (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

    def crop_to_roi(self, image, roi_size):
        """
        Crop an image to the ROI size.

        Note: if change the return order from (width, height) to (height, width), 
                it will generate a perfect image with the height of 96 and the width of 128,
                but the direction is not correct
        """
        # h, w = image.shape[:2] # 320, 488
        # target_h, target_w = roi_size # 96, 128
        # start_h = (h - target_h) // 2 # (320-96) // 2 = 112
        # start_w = (w - target_w) // 2 # (488 - 128) // 2 = 180
        # return image[start_h:start_h + target_h, start_w:start_w + target_w] # (H, W)

        # The following line came from original AM-UNET GitHub.
        # This is might not suitable for all images.
        return image[100: 196, 62:190] 

    def load_data(self):
        """
        Load and preprocess data.
        """
        data, masks = [], []

        for file_idx, slice_idx in self.valid_indices:
            # Load MRI and mask slices
            mri_data = nib.load(self.mri_files[file_idx]).get_fdata() # (width, height, slices)
            mask_data = nib.load(self.mask_files[file_idx]).get_fdata() # (width, height, slices)
            # if file_idx == 2:
            #     print(f"MRI data shape: {mri_data.shape}")
            #     print(f"Mask data shape: {mask_data.shape}")
            #     print("MRI Orientation:", nib.aff2axcodes(nib.load(self.mri_files[file_idx]).affine))
            #     print("Mask Orientation:", nib.aff2axcodes(nib.load(self.mask_files[file_idx]).affine))

            # Extract specific slices
            mri_slice = mri_data[:, :, slice_idx]
            mri_slice = np.transpose(mri_slice, (1, 0)) # Transpose (x,y,z)/(w,d,#slices) to (y,z,x)/(h,w,#slices)
            mask_slice = mask_data[:, :, slice_idx]
            mask_slice = np.transpose(mask_slice, (1, 0)) # Transpose (x,y,z)/(w,d,#slices) to (y,z,x)/(h,w,#slices)
            # if file_idx == 2:
            #     print(f"MRI per subject slices: {mri_slice.shape}")
            #     print(f"Mask per subject slices: {mask_slice.shape}")

            ############################################################################################
            # Resize to 256x256
            # cv2.resize(image, (width, height)), here is 256 x 256, does not matter
            resized_mri = cv2.resize(mri_slice, self.slice_size, interpolation=cv2.INTER_LINEAR)
            resized_mask = cv2.resize(mask_slice, self.slice_size, interpolation=cv2.INTER_NEAREST)
            # if file_idx == 2:
            #     print(f"Resized MRI per subject slices: {resized_mri.shape}")
            #     print(f"Resized Mask per subject slices: {resized_mask.shape}")
    
            # Crop to height 96, width 128
            cropped_mri = self.crop_to_roi(resized_mri, self.roi_size) # (height, width)
            cropped_mask = self.crop_to_roi(resized_mask, self.roi_size) # (height, width)
            # if file_idx == 2:
            #     print("Crop...")
            #     print(f"Cropped MRI per subject slices: {cropped_mri.shape}")
            #     print(f"Cropped Mask per subject slices: {cropped_mask.shape}")
            ############################################################################################

            # Normalize and add channel dimension
            normalized_mri = self.min_max_normalize(cropped_mri)
            data.append(normalized_mri[..., np.newaxis])
            masks.append(cropped_mask[..., np.newaxis])

        data = np.array(data)
        masks = np.array(masks)
        print(f"Finally, data shape: {data.shape}, mask shape: {masks.shape}")
        return data, masks

    def get_subject_wise_data(self, data, masks):
        """
        Load and preprocess data.
        """
        subject_wise_data = {}
        subject_wise_masks = {}

        for idx, (file_idx, _) in enumerate(self.valid_indices):
            if file_idx not in subject_wise_data:
                subject_wise_data[file_idx] = []
                subject_wise_masks[file_idx] = []
            
            # "subject_wise_data[file_idx]"" is a list of valid sclices in the "file_idx" MRI image
            subject_wise_data[file_idx].append(data[idx]) 
            subject_wise_masks[file_idx].append(masks[idx])

        # Convert lists to arrays
        for file_idx in subject_wise_data:
            subject_wise_data[file_idx] = np.array(subject_wise_data[file_idx])
            subject_wise_masks[file_idx] = np.array(subject_wise_masks[file_idx])

        return subject_wise_data, subject_wise_masks
    
    # This function is used for testing the dataloader
    def save_samples_to_folder(self, data, masks, folder_path="output_samples", num_samples=5):
        """
        Save a few samples of preprocessed data and their corresponding masks to a folder.

        Parameters:
            data (numpy.ndarray): Preprocessed MRI slices.
            masks (numpy.ndarray): Preprocessed masks.
            folder_path (str): Folder to save the images.
            num_samples (int): Number of samples to save.
        """
        if data is None or masks is None:
            print("No data or masks to save.")
            return
        
        # Ensure num_samples doesn't exceed available data
        num_samples = min(num_samples, len(data))
        
        # Randomly select sample indices
        indices = random.sample(range(len(data)), num_samples)
        
        # Create the folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)
        
        # Save each sample as an image pair
        for i, idx in enumerate(indices):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle(f"Sample {i + 1} - Slice {idx}", fontsize=16)

            # Data slice
            # In the 4th dimensionality, using index 0 to extract this unique channel and obtain a two-dimensional grayscale image.
            # If not setting origin to "lower", the images look upside down, but does not affect the strcture.
            axes[0].imshow(data[idx, :, :, 0], cmap='gray', origin='lower')
            axes[0].set_title(f"Data Slice {idx}")
            axes[0].axis('off')

            # Mask slice
            axes[1].imshow(masks[idx, :, :, 0], cmap='gray', origin='lower') 
            axes[1].set_title(f"Mask Slice {idx}")
            axes[1].axis('off')

            # Save the figure
            file_path = os.path.join(folder_path, f"sample_{i + 1}_slice_{idx}.png")
            plt.savefig(file_path)
            plt.close(fig)
        
        print(f"Saved {num_samples} samples to {folder_path}")

if __name__ == "__main__":
    TEST_DATA_DIR = "../input_data/input_mri"
    TEST_MASK_DIR = "../input_data/input_labels"

    data_handler = DataHandler(TEST_DATA_DIR, TEST_MASK_DIR, slice_size=(256, 256))

    test_data, test_masks = data_handler.load_data()
    print(f"Loaded {len(test_data)} MRI slices and {len(test_masks)} masks.")
    print(f"The shape of test data: {test_data.shape}")
    print(f"The shape of mask data: {test_masks.shape}")

    subject_wise_data, subject_wise_masks = data_handler.get_subject_wise_data(test_data, test_masks)
    print("Subject wise information...")
    print(f"Keys of subject wise data and masks: {subject_wise_data.keys()}, {subject_wise_masks.keys()}")
    [print(subject_wise_masks[i].shape) for i in subject_wise_masks]

    # Save some random samples to folder
    data_handler.save_samples_to_folder(test_data, test_masks, folder_path="output_samples", num_samples=10)
