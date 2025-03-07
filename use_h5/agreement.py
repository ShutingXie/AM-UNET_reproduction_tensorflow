import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.spatial.distance import directed_hausdorff

def calculate_dsc(seg1, seg2):
    """
    Compute Dice Similarity Coefficient (DSC).
    """
    intersection = np.logical_and(seg1, seg2).sum()
    return (2.0 * intersection) / (seg1.sum() + seg2.sum()) if seg1.sum() + seg2.sum() > 0 else np.nan

def calculate_hd(seg1, seg2):
    """
    Compute Hausdorff Distance (HD).
    """
    seg1_points = np.argwhere(seg1 > 0)
    seg2_points = np.argwhere(seg2 > 0)
    
    if len(seg1_points) == 0 or len(seg2_points) == 0:
        return np.nan  # Undefined if one segmentation is empty
    
    return max(
        directed_hausdorff(seg1_points, seg2_points)[0],
        directed_hausdorff(seg2_points, seg1_points)[0]
    )

def calculate_dilated_dsc(seg1, seg2, dilation_iters=1):
    """
    Compute Dilated Dice Similarity Coefficient (dDSC).
    """
    seg1_dilated = binary_dilation(seg1, iterations=dilation_iters)
    seg1_eroded = binary_erosion(seg1, iterations=dilation_iters)
    seg2_dilated = binary_dilation(seg2, iterations=dilation_iters)
    seg2_eroded = binary_erosion(seg2, iterations=dilation_iters)
    
    seg1_modified = np.logical_or(seg1_dilated, seg1_eroded)
    seg2_modified = np.logical_or(seg2_dilated, seg2_eroded)
    
    intersection = np.logical_and(seg1_modified, seg2_modified).sum()
    return (2.0 * intersection) / (seg1_modified.sum() + seg2_modified.sum()) if seg1_modified.sum() + seg2_modified.sum() > 0 else np.nan

def calculate_balanced_average_hd(seg1, seg2):
    """
    Compute Balanced Average Hausdorff Distance (baHD).
    """
    seg1_points = np.argwhere(seg1 > 0)
    seg2_points = np.argwhere(seg2 > 0)
    
    if len(seg1_points) == 0 or len(seg2_points) == 0:
        return np.nan  # Undefined if one segmentation is empty
    
    hd_1 = directed_hausdorff(seg1_points, seg2_points)[0]
    hd_2 = directed_hausdorff(seg2_points, seg1_points)[0]
    
    weight1 = len(seg1_points) / (len(seg1_points) + len(seg2_points))
    weight2 = len(seg2_points) / (len(seg1_points) + len(seg2_points))
    
    return (weight1 * hd_1) + (weight2 * hd_2)

