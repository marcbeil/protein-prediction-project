import math
import os
from collections import Counter
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d

from src.protenn2.dataset import CathPredPerResidueDataset


def get_train_val_test_paths(dataset_folder, train_filename="train_split.csv", val_filename="val_split.csv",
                             test_filename="test_split.csv"):
    train_path = os.path.join(dataset_folder, train_filename)
    val_path = os.path.join(dataset_folder, val_filename)
    test_path = os.path.join(dataset_folder, test_filename)
    return train_path, val_path, test_path


def calculate_max_protein_length(dataset_folder):
    paths = get_train_val_test_paths(dataset_folder)
    max_len = -1
    for path in paths:
        curr_max = pd.read_csv(path)["protein_length"].max()
        if curr_max > max_len:
            max_len = curr_max
    print("Max protein length:", max_len)
    return max_len


def calculate_baseline_accuracy_and_se(dataset: CathPredPerResidueDataset):
    """
    Calculates the per-residue baseline accuracy, its standard error,
    and a 95% confidence interval using the most frequent class strategy.
    """
    print("\n--- Calculating Baseline Accuracy and Standard Error ---")

    # --- Step 1: Find the most frequent label in the entire dataset ---
    print("Step 1: Finding the most frequent label across all residues...")
    label_counts = Counter()
    total_residues = 0

    for i in range(len(dataset)):
        _, y_full_protein, _ = dataset[i]
        label_counts.update(y_full_protein.tolist())
        total_residues += len(y_full_protein)

    most_frequent_label_id, most_frequent_label_count = label_counts.most_common(1)[0]
    most_frequent_label_name = dataset.label_encoder.inverse_transform([most_frequent_label_id])[0]

    print(f"Total residues (n) in dataset: {total_residues}")
    print(f"Most frequent label: '{most_frequent_label_name}' (ID: {most_frequent_label_id})")
    print(f"It appears {most_frequent_label_count} times.")

    # --- Step 2: Calculate accuracy (p) ---
    print("\nStep 2: Calculating accuracy...")
    correct_predictions = most_frequent_label_count
    baseline_accuracy = correct_predictions / total_residues

    print(f"Baseline Accuracy (p): {correct_predictions} / {total_residues} = {baseline_accuracy:.4f}")

    # --- Step 3: Calculate Standard Error and Confidence Interval ---
    print("\nStep 3: Calculating Standard Error (SE) and Confidence Interval...")
    if total_residues == 0:
        standard_error = float('nan')
    else:
        # Standard Error formula for a proportion: sqrt(p*(1-p)/n)
        standard_error = math.sqrt((baseline_accuracy * (1 - baseline_accuracy)) / total_residues)

    # Calculate 95% confidence interval: p ± 1.96 * SE
    z_score = 1.96  # For 95% confidence
    conf_interval_lower = baseline_accuracy - z_score * standard_error
    conf_interval_upper = baseline_accuracy + z_score * standard_error

    print(f"Standard Error (SE): {standard_error:.4f}")
    print(f"95% Confidence Interval: [{conf_interval_lower:.4f}, {conf_interval_upper:.4f}]")
    print(
        f"\nYou can report the baseline accuracy as: {baseline_accuracy:.2%} ± {z_score * standard_error:.2%} (95% CI)")

    return baseline_accuracy, standard_error


def calculate_se_for_model_accuracy(accuracy: float, n_samples: int, model_name: str = "Your Model"):
    """
    Calculates the standard error and 95% confidence interval for a given
    model accuracy and sample size.

    Args:
        accuracy (float): The accuracy of the model (e.g., 0.69 for 69%).
        n_samples (int): The total number of samples the accuracy was calculated on.
        model_name (str): The name of the model for printing.
    """
    print(f"\n--- Calculating SE and CI for {model_name} ---")
    print(f"Provided Accuracy (p): {accuracy:.4f}")
    print(f"Number of Samples (n): {n_samples}")

    if n_samples == 0:
        standard_error = float('nan')
    else:
        # Standard Error formula for a proportion: sqrt(p*(1-p)/n)
        standard_error = math.sqrt((accuracy * (1 - accuracy)) / n_samples)

    # Calculate 95% confidence interval: p ± 1.96 * SE
    z_score = 1.96  # For 95% confidence
    margin_of_error = z_score * standard_error
    conf_interval_lower = accuracy - margin_of_error
    conf_interval_upper = accuracy + margin_of_error

    print(f"Standard Error (SE): {standard_error:.4f}")
    print(f"Margin of Error: {margin_of_error:.4f}")
    print(f"95% Confidence Interval: [{conf_interval_lower:.4f}, {conf_interval_upper:.4f}]")
    print(f"\nYou can report the {model_name} accuracy as: {accuracy:.2%} ± {margin_of_error:.2%} (95% CI)")


def _coalesce_contiguous_regions(thresholded_confidences: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Helper function to find contiguous regions for each class.
    Returns a list of (class_idx, start_idx, end_idx) tuples.
    Start and end indices are 0-based, end is exclusive (Python slice style).
    """
    sequence_length, num_classes = thresholded_confidences.shape
    contiguous_regions = []

    for class_idx in range(num_classes):
        # Pad with False at start and end to easily catch regions at boundaries
        padded_column = np.pad(thresholded_confidences[:, class_idx], (1, 1), 'constant', constant_values=False)
        # Find where the boolean value changes
        diff = np.diff(padded_column.astype(int))

        # 'starts' are indices immediately after a False-to-True transition
        starts = np.where(diff == 1)[0]
        # 'ends' are indices immediately after a True-to-False transition
        ends = np.where(diff == -1)[0]

        for start_padded, end_padded in zip(starts, ends):
            # Adjust back to 0-indexed, original array coordinates
            # A 'start' at index `i` in `diff` means the region starts at `i` in `padded_column`
            # which is `i-1` in the original column.
            # An 'end' at index `j` in `diff` means the region ends *before* `j` in `padded_column`
            # so the last residue is at `j-1-1 = j-2` in original column.
            # Python slicing typically uses exclusive end, so end index is `j-1`.
            contiguous_regions.append((class_idx, start_padded - 1, end_padded - 1))

    return contiguous_regions


def _filter_region_length(regions: List[Tuple[int, int, int]], min_length: int) -> List[Tuple[int, int, int]]:
    """
    Helper function to filter regions based on their length.
    Length is calculated as (end_index - start_index) for 0-indexed, exclusive end.
    """
    filtered_regions = []
    for class_idx, start, end in regions:
        if (end - start) >= min_length:
            filtered_regions.append((class_idx, start, end))
    return filtered_regions


def call_domains(
        confidences: np.ndarray,
        no_domain_label_id: int,
        reporting_threshold: float = 0.5,
        region_min_length: int = 20,
        gaussian_sigma: float = 0.0
):
    """
    Converts a per-residue confidence array into a list of domain calls,
    following the ProtENN2 paper's domain calling procedure, with optional Gaussian smoothing.

    Args:
        confidences: A 2D numpy array of values between 0 and 1 (inclusive).
                     Shape: (sequence_length, number_of_output_classes).
                     Each cell [r, c] represents the confidence that residue 'r'
                     belongs to class 'c'.
        no_domain_label_id: The integer ID used to represent residues that do not belong to any domain.
        reporting_threshold: All confidences above this value are considered for
                             inclusion in a potential domain. Default is 0.5.
        region_min_length: Only return regions (domains) that are at least this
                           long. Default is 20.
        gaussian_sigma: The standard deviation for the Gaussian filter.
                        If 0.0, no smoothing is applied. Larger values mean more smoothing.

    Returns:
        A 1D numpy array of predicted class indices for each residue.
        The predicted_class_index corresponds to the column index in the
        'confidences' array (e.g., an index representing a Pfam family ID).
        Residues not assigned to a domain will have 'no_domain_label_id'.
    """
    sequence_length = confidences.shape[0]
    num_classes = confidences.shape[1]

    # --- Optional: Apply Gaussian Smoothing ---
    if gaussian_sigma > 0:
        smoothed_confidences = np.copy(confidences)  # Work on a copy to avoid modifying original input
        for i in range(num_classes):
            # Apply Gaussian filter to each column (class) independently
            smoothed_confidences[:, i] = gaussian_filter1d(smoothed_confidences[:, i], sigma=gaussian_sigma)
        # Ensure values remain within [0, 1] after smoothing
        smoothed_confidences = np.clip(smoothed_confidences, 0, 1)
        confidences_to_threshold = smoothed_confidences
    else:
        confidences_to_threshold = confidences

    # 1. Threshold the (optionally smoothed) confidence predictions
    thresholded_confidences = confidences_to_threshold > reporting_threshold

    # 2. Coalesce contiguous regions for each class
    contiguous_regions = _coalesce_contiguous_regions(thresholded_confidences)

    # 3. Filter out short regions
    long_contiguous_regions = _filter_region_length(contiguous_regions, region_min_length)

    # 4. Initialize the output 1D array with the 'no_domain_label'
    predicted_labels_1d = np.full(sequence_length, no_domain_label_id, dtype=int)

    # 5. Populate the 1D array with predicted class IDs for the identified domains
    for class_idx, start_idx, end_idx in long_contiguous_regions:
        # Assign the predicted class_idx to the residues within this domain
        predicted_labels_1d[start_idx:end_idx] = class_idx

    return predicted_labels_1d


def call_domains_list(
        confidences_list: List[np.ndarray],
        no_domain_label_id: int,
        reporting_threshold: float = 0.5,
        region_min_length: int = 20,
        gaussian_sigma: float = 0.0  # Pass new parameter through
):
    return [call_domains(confidences, no_domain_label_id=no_domain_label_id,
                         reporting_threshold=reporting_threshold,
                         region_min_length=region_min_length,
                         gaussian_sigma=gaussian_sigma)
            for confidences in confidences_list]


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU).")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device
