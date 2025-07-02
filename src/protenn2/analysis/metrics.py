import random  # For bootstrapping random.seed
from typing import List, Tuple, Union, Callable, Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, recall_score, precision_score
from tqdm import tqdm

from src.protenn2.analysis.cath_hierarchy_mapper import CATHHierarchyMapper


# --- Bootstrapping Confidence Interval Function (no changes needed here) ---
def bootstrap_confidence_interval(
        protein_data: List[Tuple[np.ndarray, np.ndarray]],  # Simplified to only per-residue arrays
        metric_func: Callable,  # A function that takes specific aggregated inputs and returns a dict of metrics.
        num_bootstrap_samples: int = 1000,
        alpha: float = 0.05,  # For a 95% CI, alpha=0.05
        random_seed: int = 42,  # For reproducibility of bootstrap samples
        **kwargs  # Catch all other args to pass to metric_func
) -> Dict[str, Union[float, Tuple[float, float]]]:
    """
    Computes a metric and its confidence interval using bootstrapping.

    Args:
        protein_data: A list, where each element corresponds to a single protein's
                      (ground_truth, prediction) pair: (y_true_1D_array, y_pred_1D_array).
                      The unit of resampling is *one protein*.
        metric_func: A function (like `get_per_residue_classification_metrics` or `calculate_single_protein_segment_f1`)
                     that calculates metrics. It should accept the aggregated
                     `y_true`/`y_pred` arrays (for per-residue) or individual protein arrays (for per-protein).
        num_bootstrap_samples: Number of bootstrap samples to draw.
        alpha: Significance level for the confidence interval (e.g., 0.05 for 95% CI).
        random_seed: Seed for random number generation for reproducibility.
        **kwargs: Additional keyword arguments to pass to `metric_func`.

    Returns:
        A dictionary containing the mean metric value and its confidence interval.
        Example: {'accuracy_mean': 0.85, 'accuracy_95_ci_lower': 0.84, 'accuracy_95_ci_upper': 0.86}
    """
    n_proteins = len(protein_data)
    assert n_proteins > 0

    random.seed(random_seed)  # Set seed for Python's random module
    np.random.seed(random_seed)  # Set seed for NumPy's random module

    bootstrap_scores = []

    y_true_all_flat = np.concatenate([d[0] for d in protein_data])
    y_pred_all_flat = np.concatenate([d[1] for d in protein_data])

    original_score = metric_func(y_true_all_flat, y_pred_all_flat, **kwargs)

    if num_bootstrap_samples <= 0:
        return {
            f'mean': original_score,
            f'ci_lower': original_score,
            f'ci_upper': original_score,
            f"alpha": alpha,
        }

    for _ in tqdm(range(num_bootstrap_samples), desc="Bootstrapping Progress"):
        # Resample proteins with replacement (the core of bootstrapping)
        resampled_indices = [random.randrange(n_proteins) for _ in range(n_proteins)]
        resampled_data = [protein_data[i] for i in resampled_indices]

        y_true_resampled_flat = np.concatenate([d[0] for d in resampled_data])
        y_pred_resampled_flat = np.concatenate([d[1] for d in resampled_data])
        score = metric_func(y_true=y_true_resampled_flat, y_pred=y_pred_resampled_flat, **kwargs)
        bootstrap_scores.append(score)

    # Calculate confidence interval using the percentile method
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)
    ci_lower = np.percentile(bootstrap_scores, lower_percentile)
    ci_upper = np.percentile(bootstrap_scores, upper_percentile)

    return {
        f'mean': original_score,
        f'ci_lower': ci_lower,
        f'ci_upper': ci_upper,
        f"alpha": alpha,
    }


def calculate_metrics(protein_data, num_classes, bootstrap_samples: int = 1000):
    all_results = {}
    print("----- Accuracy Metrics -----")
    acc_results = bootstrap_confidence_interval(
        protein_data=protein_data,
        metric_func=accuracy_score,
        num_bootstrap_samples=bootstrap_samples,
    )
    all_results["accuracy"] = acc_results
    print(acc_results)

    print("----- F1 Score Metrics -----")
    f1_results_post = bootstrap_confidence_interval(
        protein_data=protein_data,
        metric_func=f1_score,
        num_bootstrap_samples=bootstrap_samples,
        # additional params for metric func
        average='weighted'
    )
    all_results["f1_score"] = f1_results_post
    print(f1_results_post)

    print("----- Jaccard Score Metrics -----")
    jaccard_results = bootstrap_confidence_interval(
        protein_data=protein_data,
        metric_func=jaccard_score,
        num_bootstrap_samples=bootstrap_samples,
        # additional params for metric func
        average="weighted",
        zero_division=0,
        # Skip no_domain_encoded and padding
        labels=range(num_classes - 2)
    )
    all_results["jaccard_score"] = jaccard_results
    print(jaccard_results)

    print("----- Recall Score Metrics -----")
    recall_results = bootstrap_confidence_interval(
        protein_data=protein_data,
        metric_func=recall_score,
        num_bootstrap_samples=bootstrap_samples,
        average="weighted",
        zero_division=np.nan,
        labels=range(num_classes - 2)
    )
    all_results["recall_score"] = recall_results
    print(recall_results)

    print("----- Precision Score Metrics -----")
    precision_results = bootstrap_confidence_interval(
        protein_data=protein_data,
        metric_func=precision_score,
        num_bootstrap_samples=bootstrap_samples,
        average="weighted",
        zero_division=np.nan,
        labels=range(num_classes - 2)
    )
    all_results["precision_score"] = precision_results
    print(precision_results)

    return all_results


def calculate_metrics_for_cath_levels(y_true_labels_list, y_pred_confidences_list, mapper: CATHHierarchyMapper,
                                      bootstrap_samples: int = 1000, levels=("C", "A", "T", "H"),
                                      post_process_func=None, post_process_kwargs=None):
    all_results = {}
    for level in levels:
        print(f"---- Computing Metrics for hierarchy: {level}")
        y_true_labels_list_level, y_pred_confidences_list_level = mapper.map_y_true_labels_y_pred_confidences_lists(
            y_true_labels_list,
            y_pred_confidences_list,
            target_hierarchy=level)
        y_pred_labels_list_level = [np.argmax(protein, axis=-1) for protein in y_pred_confidences_list_level]
        per_protein_data_level = list(zip(y_true_labels_list_level, y_pred_labels_list_level))
        num_classes = mapper.get_class_count(level)
        all_results[f"raw_{level}"] = calculate_metrics(per_protein_data_level, num_classes=num_classes,
                                                        bootstrap_samples=bootstrap_samples)
        if post_process_func is not None:
            if post_process_kwargs is None:
                post_process_kwargs = {"no_domain_label_id": num_classes - 2}
            else:
                post_process_kwargs = {**post_process_kwargs, "no_domain_label_id": num_classes - 2}
            y_pred_labels_list_post_level = post_process_func(y_pred_confidences_list_level, **post_process_kwargs)
            per_protein_data_post_level = list(zip(y_true_labels_list_level, y_pred_labels_list_post_level))
            all_results[f"post_{level}"] = calculate_metrics(per_protein_data_post_level, num_classes=num_classes,
                                                             bootstrap_samples=bootstrap_samples)
    return all_results
