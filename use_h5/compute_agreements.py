import os 
import nibabel as nib
import numpy as np
from agreement import (calculate_dsc, calculate_hd, calculate_dilated_dsc, calculate_balanced_average_hd)

# Configuration, you should change the directory name
#################################################################
MASK_DIR = "../input_data/input_labels"
PRED_DIR = "predictions_Albishri_05mm"
OUTPUT_FILENAME = "agreement_computation_results_Albishri_05mm_threshold05.txt"
#################################################################

# Constant you can change
THRESHOLD = 0.5

def compute_agreements_subjectwise(mask_array, pred_array):
    dsc = calculate_dsc(mask_array, pred_array)
    hd = calculate_hd(mask_array, pred_array)
    dilated_dsc = calculate_dilated_dsc(mask_array, pred_array)
    balanced_average_hd = calculate_balanced_average_hd(mask_array, pred_array)
    return {
        "DSC": dsc,
        "HD": hd,
        "dDSC": dilated_dsc,
        "baHD": balanced_average_hd
    }

if __name__ == "__main__":
    mask_files = sorted([os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR) if f.endswith(".nii.gz") or f.endswith(".nii")])
    pred_files = sorted([os.path.join(PRED_DIR, f) for f in os.listdir(PRED_DIR) if f.endswith(".nii.gz") or f.endswith(".nii")])
    
    assert len(mask_files) == len(pred_files), "Number of MRI and mask files must match."
    
    all_subject_metrics = [] 
    subject_ids = []
    for i, (mask_file, pred_file) in enumerate(zip(mask_files, pred_files)):
        mask_array = nib.load(mask_file).get_fdata()
        pred_array = nib.load(pred_file).get_fdata()

        #################################################################
        pred_array[pred_array >= THRESHOLD] = 1.
        pred_array[pred_array < THRESHOLD] = 0.
        #################################################################
        print(f"\nComputing for the predict file: {pred_file}")
        
        # Extract subject ID from pred_file basename, e.g. "sub-PNC002_acq-uni_final_template0.nii_pred.nii.gz"
        base_name = os.path.basename(pred_file)
        subject_id = base_name.split("_")[0]
        subject_ids.append(subject_id)

        agreements_results_subjectwise = compute_agreements_subjectwise(mask_array, pred_array)
        print(agreements_results_subjectwise)
        all_subject_metrics.append(agreements_results_subjectwise)
        
    # Compute the average results
    num_subjects = len(all_subject_metrics)
    # Compute the average results, ignoring nan values for each metric
    avg_metrics = {}
    for key in ["DSC", "HD", "dDSC", "baHD"]:
        valid_values = [subject[key] for subject in all_subject_metrics if not np.isnan(subject[key])]
        if valid_values:
            avg_metrics[key] = sum(valid_values) / len(valid_values)
        else:
            avg_metrics[key] = np.nan

    print("\nAverage metrics:")
    print(avg_metrics)

    # Save agreement results in s txt file
    with open(OUTPUT_FILENAME, "w") as f:
        f.write("Subject-wise Evaluation Metrics:\n")
        for subject_id, metrics in zip(subject_ids, all_subject_metrics):
            f.write(f"Subject {subject_id}: {metrics}\n")
        f.write("\nAverage Evaluation Metrics:\n")
        f.write(str(avg_metrics))
    
    print(f"\nEvaluation metrics saved to {OUTPUT_FILENAME}")

    # Doule-check the threshold value
    print(f"Threshold: {THRESHOLD}")


