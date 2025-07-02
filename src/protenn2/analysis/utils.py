# --- Helper for Segment Overlap (SOV-like) Score ---
from typing import List, Tuple

import numpy as np


def _get_segments(labels: np.ndarray, ignore_id: int) -> List[Tuple[int, int, int]]:
    """
    Converts a 1D array of labels into a list of (start, end, class_id) tuples
    for contiguous regions, ignoring the specified ID.
    """
    segments = []
    if len(labels) == 0:
        return segments

    current_segment_start = None
    current_segment_label = None

    for i, label in enumerate(labels):
        if label == ignore_id:
            if current_segment_start is not None:
                segments.append((current_segment_start, i - 1, current_segment_label))
                current_segment_start = None
                current_segment_label = None
        else:  # Not an ignore_id
            if current_segment_start is None:  # Start of a new segment
                current_segment_start = i
                current_segment_label = label
            elif label != current_segment_label:  # Segment changed class
                segments.append((current_segment_start, i - 1, current_segment_label))
                current_segment_start = i
                current_segment_label = label

    # Handle the last segment if it extends to the end of the array
    if current_segment_start is not None:
        segments.append((current_segment_start, len(labels) - 1, current_segment_label))

    return segments