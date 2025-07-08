from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm


def run_inference(
        model,
        dataloader,
        padding_encoded_id: int,
        device: torch.device = torch.device("cpu"),
        return_protein_chain_id=False
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Run model inference and return trimmed ground truth and probabilities.

    Args:
        model: The trained model
        dataloader: DataLoader for inference
        padding_encoded_id: ID used for padding tokens
        device: Device to run inference on

    Returns:
        Tuple of (y_true_list, probabilities_list) where each list contains
        numpy arrays for each protein with padding removed.
        - y_true_list: List of np.ndarray, where each array has shape (sequence_len,)
        - probabilities_list: List of np.ndarray, where each array has shape (sequence_len, num_classes)
    """
    model.eval()

    y_true_list = []
    probabilities_list = []
    protein_chain_id_list = []
    print(f"Running inference on {len(dataloader)} proteins...")

    with torch.no_grad():
        for i, (x, y_true_batch, protein_chain_id) in enumerate(tqdm(dataloader, desc="Inference Progress")):
            # Get ground truth
            # y_true_batch[0] assumes batch size of 1 for correct trimming per protein
            y_true_protein_full = y_true_batch[0].cpu().numpy()

            # Determine actual protein length (excluding padding)
            padding_start_indices = np.where(y_true_protein_full == padding_encoded_id)[0]
            actual_protein_length = padding_start_indices[0] if padding_start_indices.size > 0 else len(
                y_true_protein_full)

            # Move input to device
            for k, v in x.items():
                x[k] = v.to(device, non_blocking=True)

            # Get model outputs
            raw_outputs_logits = model(x)

            # Convert to probabilities and move to CPU
            # raw_outputs_logits[0] assumes batch size of 1
            probabilities_full = torch.softmax(raw_outputs_logits[0], dim=-1).cpu().numpy()

            # Trim padding from both ground truth and probabilities
            y_true_trimmed = y_true_protein_full[:actual_protein_length]
            probabilities_trimmed = probabilities_full[:actual_protein_length]

            y_true_list.append(y_true_trimmed)
            probabilities_list.append(probabilities_trimmed)
            protein_chain_id_list.append(protein_chain_id[0])

    print(f"Inference complete. Processed {len(y_true_list)} proteins")
    if return_protein_chain_id:
        return y_true_list, probabilities_list, protein_chain_id_list
    return y_true_list, probabilities_list


def run_inference_dummy(
        dummy_classifier,
        dataloader,
        padding_encoded_id: int,
        num_classes: int,
        dummy_classifier_kwargs: dict = {},
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Run dummy inference and return trimmed ground truth and dummy probabilities.

    Args:
        dummy_classifier: Callable that returns predicted labels for a given true label array.
        dataloader: DataLoader for inference (expects batch size 1)
        padding_encoded_id: ID used for padding tokens

    Returns:
        Tuple of (y_true_list, probabilities_list), where:
            - y_true_list: List of np.ndarray, shape (seq_len,)
            - probabilities_list: List of np.ndarray, shape (seq_len, num_classes)
    """
    y_true_list = []
    probabilities_list = []
    protein_chain_id_list = []

    print(f"Running dummy inference on {len(dataloader)} proteins...")

    for i, (_, y_true_batch, protein_chain_id) in enumerate(tqdm(dataloader, desc="Dummy Inference Progress")):
        # Get full true labels
        y_true_full = y_true_batch[0].cpu().numpy()

        # Trim padding
        padding_start = np.where(y_true_full == padding_encoded_id)[0]
        seq_len = padding_start[0] if padding_start.size > 0 else len(y_true_full)
        y_true_trimmed = y_true_full[:seq_len]

        # Get dummy predictions from the classifier
        y_pred_dummy = dummy_classifier(y_true_trimmed, **dummy_classifier_kwargs)

        # Convert to one-hot encoded probability-like vectors

        probs_dummy = np.eye(num_classes)[y_pred_dummy]

        y_true_list.append(y_true_trimmed)
        probabilities_list.append(probs_dummy)
        protein_chain_id_list.append(protein_chain_id)

    print(f"Dummy inference complete. Processed {len(y_true_list)} proteins.")

    return y_true_list, probabilities_list, protein_chain_id_list


def dummy_majority_classifier_per_protein(y_true_trimmed: np.ndarray, majority_label) -> np.ndarray:
    return np.full_like(y_true_trimmed, majority_label)


def dummy_stratified_classifier_per_protein(
        y_true_trimmed: np.ndarray,
        distribution: np.ndarray
) -> np.ndarray:
    # Predicts classes according to the provided global distribution
    class_ids = np.arange(distribution.shape[0])
    return np.random.choice(class_ids, size=y_true_trimmed.shape[0], p=distribution)


def dummy_uniform_classifier_per_protein(
        y_true_trimmed: np.ndarray,
        num_classes: int
) -> np.ndarray:
    return np.random.randint(0, num_classes, size=y_true_trimmed.shape[0])


def dummy_constant_classifier_per_protein(
        y_true_trimmed: np.ndarray,
        constant_label: int
) -> np.ndarray:
    # always predicts the same constant_label
    return np.full_like(y_true_trimmed, fill_value=constant_label)
