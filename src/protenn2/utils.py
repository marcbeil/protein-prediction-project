import math
import os
from collections import Counter

import pandas as pd

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
