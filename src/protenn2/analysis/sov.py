from typing import List

import numpy as np


def get_segments(label_array, no_label):
    segments = []
    start = None
    current_label = None

    for i, label in enumerate(label_array):
        if label == no_label:
            if current_label is not None:
                segments.append((current_label, start, i - 1))
                current_label = None
        else:
            if label != current_label:
                if current_label is not None:
                    segments.append((current_label, start, i - 1))
                current_label = label
                start = i

    if current_label is not None:
        segments.append((current_label, start, len(label_array) - 1))

    return segments


def compute_delta(minov, maxov, len_s1, len_s2):
    return min(
        (maxov - minov),
        minov,
        int(0.5 * len_s1),
        int(0.5 * len_s2)
    )


def compute_sov_score(true_segs, pred_segs, label):
    overlaps = 0
    total_len = 0

    matched_pred = set()

    for true_label, start_t, end_t in true_segs:
        if true_label != label:
            continue
        len_s1 = end_t - start_t + 1
        total_len += len_s1

        best_sov = 0
        for i, (pred_label, start_p, end_p) in enumerate(pred_segs):
            if pred_label != label:
                continue
            if end_t < start_p or end_p < start_t:
                continue  # no overlap
            matched_pred.add(i)

            ov_start = max(start_t, start_p)
            ov_end = min(end_t, end_p)
            minov = ov_end - ov_start + 1
            maxov = max(end_t, end_p) - min(start_t, start_p) + 1
            delta = compute_delta(minov, maxov, len_s1, end_p - start_p + 1)
            sov = ((minov + delta) / maxov) * len_s1
            best_sov = max(best_sov, sov)

        overlaps += best_sov

    return overlaps, total_len


def segment_overlap_score(y_true_labels: List[np.ndarray], y_pred_labels: List[np.ndarray], no_domain_label_id: int):
    assert len(y_true_labels) == len(y_pred_labels)

    total_score = 0
    total_length = 0

    for y_true, y_pred in zip(y_true_labels, y_pred_labels):
        true_segs = get_segments(y_true, no_domain_label_id)
        pred_segs = get_segments(y_pred, no_domain_label_id)

        labels = set(y_true) | set(y_pred)
        labels.discard(no_domain_label_id)

        for label in labels:
            score, length = compute_sov_score(true_segs, pred_segs, label)
            total_score += score
            total_length += length

    return (total_score / total_length) if total_length > 0 else 0.0
