import nibabel as nib
import numpy as np
import os
import cv2

class VolumeRestorer:
    def __init__(self, data_handler, resized_size=(256,256), roi_size=(96, 128)): # (H, W)
        """
        Initialize VolumeRestorer with access to DataHandler.
        """
        self.data_handler = data_handler
        self.roi_size = roi_size
        self.resized_size = resized_size

    def pad_with_256(self, c_x):
        """
        "Hard-code" a ROI of shape (96,128) into a 256x256 image, with a position range of [100:196, 62:190]
        """
        n, r = c_x.shape  # n=96, r=128
        c_y = np.zeros((n+160, r+128), dtype=c_x.dtype)  # => (256,256)
        c_y[100:196, 62:190] = c_x
        return c_y
    
    def filter_false_positives(self, binary_roi):
        """
        Further filter false positive pixels in the binary ROI image based on prior knowledge.
        In this version, we assume that the CLAUSTRUM should not appear in columns 45 to 90 of the ROI.
        Thus, all pixels in these columns are set to 0.
        """
        # Remove channel dimension if present
        binary_roi = np.squeeze(binary_roi)  # now shape is (96,128)
        # Set the pixels in columns 45 to 90 to zero
        binary_roi[:, 45:90] = 0
        return binary_roi
    
    def restore_roi_to_256(self, cropped_image):
        cropped_image = np.squeeze(cropped_image)  # (96,128)
        full_256 = self.pad_with_256(cropped_image)  # (256,256)
        return full_256
    
    def upsample_to_original_size(self, slice_256, org_size):
        """
        Return an upsampling result from (256,256) back to (height=488, width=320).
        slice_256: shape (256,256)
        org_size : shape (H, W, #slices)
        return   : shape (488,320)
        """
        # Note that the parameter order of cv2.resize is (width, height)
        # Choose what interpolation method？？？
        slice_big = cv2.resize(slice_256, (org_size[1], org_size[0]), interpolation=cv2.INTER_NEAREST)
        # shape = (488,320)
        return slice_big

    def restore_roi_to_full_size(self, cropped_image, org_size):
        """
        First recover to 256x256, then recover to original resolution org_size.
        """
        slice_256 = self.restore_roi_to_256(cropped_image)
        slice_org = self.upsample_to_original_size(slice_256, org_size)
        return slice_org

    # I first thought this function is useless since it will conver the non-CL areas when overlapping 3D reconstruction over the original images.
    # Why necessary: This function can make sure the third dimension is also the same as original images.
    def restore_invalid_slices(self, subject_predictions, total_slices, valid_indices):
        """
        Restore invalid slices by inserting zeros for missing slices.
        """
        restored_volume = np.zeros((subject_predictions.shape[1], subject_predictions.shape[2], total_slices), dtype=subject_predictions.dtype) # (H, W)

        for idx, (_, slice_idx) in enumerate(valid_indices):
            restored_volume[:, :, slice_idx] = subject_predictions[idx]

        return restored_volume

    def reconstruct_3d_volume(self, subject_predictions, subject_id):
        """
        Reconstruct the full 3D volume with restored ROI and invalid slices for a given subject.
        """
        # Step 1: Get valid/predicted indices for the subject
        # valid_indices1 = [(file_idx, slice_idx) for file_idx, slice_idx in self.data_handler._filter_valid_slices() if file_idx == subject_id] # No need to invoke the function _filter_valid_slices and compute it again
        valid_indices = [(file_idx, slice_idx) for file_idx, slice_idx in self.data_handler.valid_indices if file_idx == subject_id]
        # print(f"\nvalid indeces1: {valid_indices1}") # list
        # print(f"\nvalid indeces: {valid_indices}") # list
        # print(f"\nTwo ways to compute valid indices for each subject: {valid_indices1==valid_indices}") # no need to use a.all() or a.any()

        # Step 2: Get total slices for the subject
        total_slices = self.data_handler.slices_per_image[subject_id]

        # Step 3: Restore ROI to full size for each slice
        org_h, org_w, _ = self.data_handler.original_sizes[subject_id] # (H, W, #slices)
        org_size = (org_h, org_w)

        restored_slices = []
        for slice_2d in subject_predictions:
            # First, perform false positive filtering: only keep the pixels in the left and right areas
            filtered_roi = self.filter_false_positives(slice_2d)
            big_slice = self.restore_roi_to_full_size(filtered_roi, org_size)  # => shape (488,320)/(H,W)
            restored_slices.append(big_slice)
        restored_slices = np.array(restored_slices)

        # Step 4: Restore invalid slices to full 3D volume
        print("Transpose during restore processing...")
        restored_volume = self.restore_invalid_slices(restored_slices, total_slices, valid_indices)
        print(f"Before tranpose, the shape of restored slices is: {restored_volume.shape}")
        restored_volume = np.transpose(restored_volume, (1, 0, 2)) # (H, W) --> (W, H)
        print(f"After tranpose, the shape of restored slices is: {restored_volume.shape}")
            
        return restored_volume

    def save_as_nifti(self, subject_id, volume, save_path):
        """
        Save the reconstructed volume as a NIfTI file.
        """

        # Using the information from original data to create an affine matrix
        original_nii = nib.load(self.data_handler.mri_files[subject_id])
        """
        affine: --> float
        Original affine: [[   0.5           0.           -0.          -80.03103638]
        [   0.            0.5          -0.         -109.258255  ]
        [   0.            0.            0.5         -92.30477905]
        [   0.            0.            0.            1.        ]]

        nib.aff2axcodes(nib.load(self.mri_files[file_idx]).affine): --> int
        ['R', 'S', 'A']
        """
        original_affine = original_nii.affine
        if subject_id == 0:
            print(f"Original affine: {original_affine}")
        original_header = original_nii.header
        if subject_id == 0:
            print(f"Original header: {original_header}")
        new_nii = nib.Nifti1Image(volume, original_affine, original_header)
        nib.save(new_nii, save_path)
