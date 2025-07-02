from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm


def run_inference(
        model,
        dataloader,
        padding_encoded_id: int,
        device: torch.device = torch.device("cpu")
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

    print(f"Running inference on {len(dataloader)} proteins...")

    with torch.no_grad():
        for i, (x, y_true_batch, domain_id_batch) in enumerate(tqdm(dataloader, desc="Inference Progress")):
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

    print(f"Inference complete. Processed {len(y_true_list)} proteins")

    return y_true_list, probabilities_list