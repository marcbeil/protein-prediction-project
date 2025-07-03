def parse_cath(cath_string):
    """
    Parses a CATH string (e.g., "3.30.559.10") into its components.
    Returns (Class, Architecture, Topology, Homologous Superfamily) as strings.
    If a component is missing, it returns an empty string for that part.
    Handles 'NO_DOMAIN_REGION' specifically.
    """
    if cath_string == "NO_DOMAIN_REGION":
        return None, None, None, None
    parts = cath_string.split('.')
    # Pad with empty strings if parts are missing, to always return 4 components
    return (parts + [''] * 4)[:4]


# Pre-defined base colors for CATH domains (e.g., based on their full ID)
# This will be a fixed set of distinct colors, one for each unique CATH domain seen in the dataset.
# Initialize this once, or lazily populate it.
_CATH_BASE_COLORS = {}
_CATH_COLOR_COUNTER = 0
_N_COLORS = 20  # Number of distinct base colors to generate cyclically

import colorsys

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches  # Needed for legend patches
import matplotlib.pyplot as plt  # Needed for plotting functions
import numpy as np  # Needed for numpy operations

# Global variables for consistent hierarchical color assignment
# _CATH_BASE_COLORS and _CATH_COLOR_COUNTER from your old code are no longer needed
# for the hierarchical scheme, but I'll keep them commented out for clarity.
# global _CATH_BASE_COLORS, _CATH_COLOR_COUNTER

_CATH_CLASS_HUES = {}  # Maps CATH Class (e.g., 'C', 'D') to a fixed hue
_CLASS_HUE_COUNTER = 0  # Counter to assign unique hues to new classes
_N_CLASS_HUES = 4  # Number of distinct base hues to cycle through for classes


def _generate_distinct_hue(index, max_hues):
    """Generates a distinct hue value (0-1) based on an index."""
    return index / max_hues


def get_base_cath_color(cath_label_string):
    """
    Retrieves a consistent RGB color for a given CATH label string,
    implementing a hierarchical coloring scheme.

    Format: X.Y.Z.W (Class.Architecture.Topology.HomologousSuperfamily)
    - Base color (hue) is determined by the CATH Class (X).
    - Accent (saturation/value) can be varied based on deeper hierarchies (Y, Z).
    """
    global _CATH_CLASS_HUES, _CLASS_HUE_COUNTER

    if cath_label_string == "NO_DOMAIN_REGION":
        return mcolors.to_rgb('white')  # Consistent gray for non-domain regions

    # Parse the CATH label string to get the class and other parts
    parts = cath_label_string.split('.')
    cath_class = parts[0]  # The 'C' from 'C.1.2.3'

    # Assign a consistent base hue for each unique CATH Class
    if cath_class not in _CATH_CLASS_HUES:
        _CATH_CLASS_HUES[cath_class] = _generate_distinct_hue(_CLASS_HUE_COUNTER, _N_CLASS_HUES)
        _CLASS_HUE_COUNTER = (_CLASS_HUE_COUNTER + 1) % _N_CLASS_HUES  # Cycle through available hues

    base_hue = _CATH_CLASS_HUES[cath_class]

    # Initialize saturation and value (lightness) for the accent
    saturation = 0.8
    value = 0.8

    # Apply accent based on Architecture (Y)
    if len(parts) > 1:
        try:
            # Use the Architecture index to subtly vary saturation
            arch_index = int(parts[1])
            # Small variation: e.g., 0.8, 0.85, 0.9, 0.95
            saturation = 0.8 + (arch_index % 4) * 0.05
        except ValueError:
            pass  # If Architecture part is not a number, ignore for accent

    # Apply accent based on Topology (Z)
    if len(parts) > 2:
        try:
            # Use the Topology index to subtly vary value
            topo_index = int(parts[2])
            # Small variation: e.g., 0.8, 0.85, 0.9, 0.95
            value = 0.8 + (topo_index % 4) * 0.05
        except ValueError:
            pass  # If Topology part is not a number, ignore for accent

    # Ensure saturation and value are within valid HSV ranges (0 to 1)
    saturation = np.clip(saturation, 0.4, 1.0)  # Prevent colors from being too pale
    value = np.clip(value, 0.4, 1.0)  # Prevent colors from being too dark or too bright

    # Convert HSV to RGB
    r, g, b = colorsys.hsv_to_rgb(base_hue, saturation, value)
    return (r, g, b)


def visualize_prediction_sample_only_true(
        y_true_labels: np.ndarray,
        y_pred_confidences: np.ndarray,
        label_encoder,
        post_process_func,
        post_process_func_args: dict,
        protein_chain_id: str
):
    """
    Visualizes the ground truth, raw prediction (confidences for true labels only),
    and post-processed output for one protein sample.
    """
    # --- Map IDs to string labels ---
    id_to_label = {i: label for i, label in enumerate(label_encoder.classes_)}
    true_labels_str = [id_to_label[int(x)] for x in y_true_labels]

    # --- Predicted graph: confidences for true labels only ---
    true_label_ids = y_true_labels.astype(int)
    true_label_confidences = y_pred_confidences[
        np.arange(len(y_true_labels)), true_label_ids
    ]
    # RGBA for predicted: use true-label color + confidence as alpha
    # (clipped to [0.1, 1.0] for visibility)

    # --- Post-processed prediction ---
    y_post_processed = post_process_func(y_pred_confidences, **post_process_func_args)
    post_labels_str = [id_to_label[int(x)] for x in y_post_processed]

    # --- Unique labels and colors ---
    all_labels = sorted(set(true_labels_str + post_labels_str + ["NO_DOMAIN_REGION"]))
    label_to_color = {label: get_base_cath_color(label) for label in all_labels}

    # --- RGBA arrays ---
    true_rgba = [list(label_to_color[label]) + [1.0] for label in true_labels_str]
    pred_rgba = [
        list(label_to_color[label]) + [np.clip(conf, 0.1, 1.0)]
        for label, conf in zip(true_labels_str, true_label_confidences)
    ]
    allowed_labels = set(true_labels_str)
    post_rgba = [
        # if the post label was seen in the true labels, draw it; otherwise make it fully transparent
        list(label_to_color[label]) + [1.0] if label in allowed_labels else [0, 0, 0, 0]
        for label in post_labels_str
    ]
    # --- Plotting ---
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(10, 4), sharex=True, constrained_layout=True
    )
    fig.suptitle(f"Protein Chain {protein_chain_id}", fontsize=14)

    ax1.imshow([true_rgba], aspect='auto')
    ax1.set_title("Ground Truth", loc='left')
    ax1.set_yticks([])

    ax2.imshow([pred_rgba], aspect='auto')
    ax2.set_title("Predicted (True-label confidences)", loc='left')
    ax2.set_yticks([])

    ax3.imshow([post_rgba], aspect='auto')
    ax3.set_title("Post-Processed", loc='left')
    ax3.set_yticks([])
    ax3.set_xlabel("Residue Index")



    # --- Legend: only labels both in true and post-processed outputs ---
    legend_handles = [
        mpatches.Patch(color=label_to_color[label], label=label)
        for label in sorted(allowed_labels & set(post_labels_str))
    ]
    print(legend_handles)
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=min(6, len(legend_handles)),
        bbox_to_anchor=(0.5, -0.1),
        frameon=False
    )

    return fig

