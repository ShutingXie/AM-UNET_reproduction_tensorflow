import nibabel as nib
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
    
    def visualize_all_valid_slices_roi(self, output_folder="roi_validation"):
        """
        Visualize all valid slices to verify if the ROI crop [100:196, 62:190] contains the complete claustrum.
        For each valid slice, display:
        1. The original slice (transposed)
        2. The resized slice (256x256)
        3. The ROI (cropped region)
        4. The overlay of mask on the original, resized, and cropped images
        """
        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"Starting visualization of all {len(self.valid_indices)} valid slices...")
        
        for i, (file_idx, slice_idx) in enumerate(self.valid_indices):
            print(f"Processing slice {i+1}/{len(self.valid_indices)}: File {file_idx}, Slice {slice_idx}")
            
            # Load MRI and mask slices
            mri_data = nib.load(self.mri_files[file_idx]).get_fdata()
            mask_data = nib.load(self.mask_files[file_idx]).get_fdata()
            
            # Extract specific slices
            mri_slice = mri_data[:, :, slice_idx]
            mri_slice = np.transpose(mri_slice, (1, 0))  # Transpose to (height, width)
            mask_slice = mask_data[:, :, slice_idx]
            mask_slice = np.transpose(mask_slice, (1, 0))  # Transpose to (height, width)
            
            # Normalize original slice for better visualization
            norm_mri_slice = self.min_max_normalize(mri_slice)
            
            # Resize to 256x256
            resized_mri = cv2.resize(mri_slice, self.slice_size, interpolation=cv2.INTER_LINEAR)
            resized_mask = cv2.resize(mask_slice, self.slice_size, interpolation=cv2.INTER_NEAREST)
            norm_resized_mri = self.min_max_normalize(resized_mri)
            
            # Create visualization of the ROI box on the resized image
            resized_with_roi = norm_resized_mri.copy()
            # Draw rectangle around ROI area [100:196, 62:190]
            cv2.rectangle(resized_with_roi, (62, 100), (190, 196), (1, 1, 1), 2)
            
            # Crop to ROI
            cropped_mri = self.crop_to_roi(resized_mri, self.roi_size)
            cropped_mask = self.crop_to_roi(resized_mask, self.roi_size)
            norm_cropped_mri = self.min_max_normalize(cropped_mri)

            # Create overlay images
            # For original
            original_overlay = np.zeros_like(norm_mri_slice)
            original_overlay = np.stack((norm_mri_slice, norm_mri_slice, norm_mri_slice), axis=-1)
            original_mask_vis = np.zeros_like(mask_slice)
            if np.any(mask_slice):
                original_mask_vis = mask_slice / np.max(mask_slice)  # Normalize mask to [0,1]
            original_overlay[mask_slice > 0, 0] = 0  # Zero out red channel where mask is
            original_overlay[mask_slice > 0, 1] = original_mask_vis[mask_slice > 0]  # Green for mask
            
            # For resized
            resized_overlay = np.zeros_like(norm_resized_mri)
            resized_overlay = np.stack((norm_resized_mri, norm_resized_mri, norm_resized_mri), axis=-1)
            resized_mask_vis = np.zeros_like(resized_mask)
            if np.any(resized_mask):
                resized_mask_vis = resized_mask / np.max(resized_mask)  # Normalize mask to [0,1]
            resized_overlay[resized_mask > 0, 0] = 0  # Zero out red channel where mask is
            resized_overlay[resized_mask > 0, 1] = resized_mask_vis[resized_mask > 0]  # Green for mask
            
            # For cropped
            cropped_overlay = np.zeros_like(norm_cropped_mri)
            cropped_overlay = np.stack((norm_cropped_mri, norm_cropped_mri, norm_cropped_mri), axis=-1)
            cropped_mask_vis = np.zeros_like(cropped_mask)
            if np.any(cropped_mask):
                cropped_mask_vis = cropped_mask / np.max(cropped_mask)  # Normalize mask to [0,1]
            cropped_overlay[cropped_mask > 0, 0] = 0  # Zero out red channel where mask is
            cropped_overlay[cropped_mask > 0, 1] = cropped_mask_vis[cropped_mask > 0]  # Green for mask
            
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 10))
            gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1])
            
            # Row 1: MRI Images
            ax1 = plt.subplot(gs[0, 0])
            ax1.imshow(norm_mri_slice, cmap='gray', origin='lower')
            ax1.set_title(f"Original MRI\nFile {file_idx}, Slice {slice_idx}")
            ax1.axis('off')
            
            ax2 = plt.subplot(gs[0, 1])
            ax2.imshow(norm_resized_mri, cmap='gray', origin='lower')
            ax2.set_title(f"Resized MRI (256x256)")
            ax2.axis('off')
            
            ax3 = plt.subplot(gs[0, 2])
            ax3.imshow(resized_with_roi, cmap='gray', origin='lower')
            ax3.set_title(f"Resized MRI with ROI box")
            ax3.axis('off')
            
            ax4 = plt.subplot(gs[0, 3])
            ax4.imshow(norm_cropped_mri, cmap='gray', origin='lower')
            ax4.set_title(f"Cropped ROI [100:196, 62:190]")
            ax4.axis('off')
            
            # Row 2: Overlays
            ax5 = plt.subplot(gs[1, 0])
            ax5.imshow(original_overlay, origin='lower')
            ax5.set_title(f"Original with Mask Overlay")
            ax5.axis('off')
            
            ax6 = plt.subplot(gs[1, 1])
            ax6.imshow(resized_overlay, origin='lower')
            ax6.set_title(f"Resized with Mask Overlay")
            ax6.axis('off')
            
            # Display the mask separately for better visualization
            ax7 = plt.subplot(gs[1, 2])
            ax7.imshow(resized_mask, cmap='Greens', origin='lower')
            ax7.set_title(f"Resized Mask Only")
            ax7.axis('off')
            
            ax8 = plt.subplot(gs[1, 3])
            ax8.imshow(cropped_overlay, origin='lower')
            ax8.set_title(f"Cropped ROI with Mask Overlay")
            ax8.axis('off')
            
            plt.tight_layout()
            
            # Save the figure
            file_path = os.path.join(output_folder, f"slice_{i+1}_file{file_idx}_slice{slice_idx}.png")
            plt.savefig(file_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
        print(f"Visualization complete. All images saved to {output_folder}")
        
        # Create an additional summary image with counts of claustrum pixels in and out of ROI
        self._create_roi_coverage_summary(output_folder)
    
    def _create_roi_coverage_summary(self, output_folder):
        """
        Create a summary of how well the ROI covers the claustrum across all valid slices.
        Calculates what percentage of claustrum pixels are inside vs outside the ROI.
        """
        total_mask_pixels = 0
        mask_pixels_in_roi = 0
        
        results = []
        
        for file_idx, slice_idx in self.valid_indices:
            # Load mask slice
            mask_data = nib.load(self.mask_files[file_idx]).get_fdata()
            mask_slice = mask_data[:, :, slice_idx]
            mask_slice = np.transpose(mask_slice, (1, 0))  # Transpose to (height, width)
            
            # Resize to 256x256
            resized_mask = cv2.resize(mask_slice, self.slice_size, interpolation=cv2.INTER_NEAREST)
            
            # Count total mask pixels
            mask_pixel_count = np.sum(resized_mask > 0)
            total_mask_pixels += mask_pixel_count
            
            # Create an empty image of the same size as resized mask
            roi_mask = np.zeros_like(resized_mask)
            # Set the ROI region to 1
            roi_mask[100:196, 62:190] = 1
            
            # Count mask pixels inside ROI
            mask_in_roi = np.sum((resized_mask > 0) & (roi_mask > 0))
            mask_pixels_in_roi += mask_in_roi
            
            # Calculate percentage for this slice
            percentage_in_roi = 100.0 * mask_in_roi / mask_pixel_count if mask_pixel_count > 0 else 0
            
            results.append({
                'file_idx': file_idx,
                'slice_idx': slice_idx,
                'total_mask_pixels': mask_pixel_count,
                'mask_pixels_in_roi': mask_in_roi,
                'percentage_in_roi': percentage_in_roi
            })
        
        # Calculate overall percentage
        overall_percentage = 100.0 * mask_pixels_in_roi / total_mask_pixels if total_mask_pixels > 0 else 0
        
        # Create a summary file
        summary_path = os.path.join(output_folder, "roi_coverage_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"ROI Coverage Summary\n")
            f.write(f"===================\n\n")
            f.write(f"Total claustrum pixels across all valid slices: {total_mask_pixels}\n")
            f.write(f"Claustrum pixels inside ROI [100:196, 62:190]: {mask_pixels_in_roi}\n")
            f.write(f"Overall percentage of claustrum covered by ROI: {overall_percentage:.2f}%\n\n")
            
            f.write(f"Per-slice breakdown:\n")
            f.write(f"-------------------\n")
            
            for result in results:
                f.write(f"File {result['file_idx']}, Slice {result['slice_idx']}: ")
                f.write(f"{result['mask_pixels_in_roi']} of {result['total_mask_pixels']} pixels ")
                f.write(f"({result['percentage_in_roi']:.2f}%) inside ROI\n")
        
        print(f"ROI coverage summary saved to {summary_path}")
        
        # Create a visual summary
        plt.figure(figsize=(10, 6))
        
        slice_labels = [f"F{r['file_idx']}S{r['slice_idx']}" for r in results]
        percentages = [r['percentage_in_roi'] for r in results]
        
        plt.bar(range(len(percentages)), percentages, color='steelblue')
        plt.axhline(y=100, color='r', linestyle='-', alpha=0.3)
        plt.axhline(y=overall_percentage, color='g', linestyle='--', label=f'Overall: {overall_percentage:.2f}%')
        
        plt.xlabel('Slice')
        plt.ylabel('Percentage of Claustrum in ROI')
        plt.title('Percentage of Claustrum Pixels inside ROI per Slice')
        plt.xticks(range(len(slice_labels)), slice_labels, rotation=90, fontsize=8)
        plt.yticks(range(0, 110, 10))
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        plot_path = os.path.join(output_folder, "roi_coverage_plot.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"ROI coverage plot saved to {plot_path}")

if __name__ == "__main__":
    TEST_DATA_DIR = "../input_data/input_mri"
    TEST_MASK_DIR = "../input_data/input_labels"

    data_handler = DataHandler(TEST_DATA_DIR, TEST_MASK_DIR, slice_size=(256, 256))

    # Method1: visualize ROI
    # test_data, test_masks = data_handler.load_data()
    # print(f"Loaded {len(test_data)} MRI slices and {len(test_masks)} masks.")
    # print(f"The shape of test data: {test_data.shape}")
    # print(f"The shape of mask data: {test_masks.shape}")

    # subject_wise_data, subject_wise_masks = data_handler.get_subject_wise_data(test_data, test_masks)
    # print("Subject wise information...")
    # print(f"Keys of subject wise data and masks: {subject_wise_data.keys()}, {subject_wise_masks.keys()}")
    # [print(subject_wise_masks[i].shape) for i in subject_wise_masks]

    # # Save some random samples to folder
    # data_handler.save_samples_to_folder(test_data, test_masks, folder_path="output_samples", num_samples=10)

    # Method2: visualize ROI
    data_handler.visualize_all_valid_slices_roi(output_folder="roi_validation")
