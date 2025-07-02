import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder


class CATHHierarchyMapper:
    """
    A class to map encoded CATH labels and their corresponding probability distributions
    from the H-level to any other CATH hierarchy level.
    """

    def __init__(self, label_encoder: LabelEncoder):
        """
        Initializes the mapper using a pre-fitted H-level LabelEncoder.
        This will pre-compute all necessary mapping dictionaries and matrices.
        """
        self.h_level_encoder = label_encoder
        h_level_classes = self.h_level_encoder.classes_
        num_h_classes = len(h_level_classes)

        # --- Configuration ---
        self.special_labels = {"PADDING_REGION", "NO_DOMAIN_REGION"}
        self.hierarchy_levels = {'C': 1, 'A': 2, 'T': 3, 'H': 4}

        # --- Build Mappings ---
        self.target_maps = {}
        self.target_class_count = {}
        self.mapping_matrices = {}

        for level_name, depth in self.hierarchy_levels.items():
            # For the H-level, the mapping is an identity transformation
            if level_name == 'H':
                self.target_maps['H'] = {i: i for i in range(num_h_classes)}
                self.target_class_count['H'] = num_h_classes
                self.mapping_matrices['H'] = np.identity(num_h_classes, dtype=np.float32)
                continue

            # 1. Create a unified vocabulary for the target level that includes
            #    both CATH classes and special labels, giving them all new, compact IDs.
            target_vocab = {}
            next_new_id = 0

            # Add the truncated CATH labels
            for label_str in h_level_classes:
                if label_str not in self.special_labels:
                    target_str = '.'.join(label_str.split('.')[:depth])
                    if target_str not in target_vocab:
                        target_vocab[target_str] = next_new_id
                        next_new_id += 1

                # Add special labels first for consistent IDs (e.g., 0 and 1)
            for special_label in sorted(list(self.special_labels)):
                if special_label not in target_vocab:
                    target_vocab[special_label] = next_new_id
                    next_new_id += 1

            num_target_classes = len(target_vocab)
            self.target_class_count[level_name] = num_target_classes

            # 2. Build the final map from each original H-level int to a new target-level int
            h_to_target_map = {}
            for h_level_int, h_level_str in enumerate(h_level_classes):
                if h_level_str in self.special_labels:
                    # Map to the special label's *new* ID in the target vocab
                    h_to_target_map[h_level_int] = target_vocab[h_level_str]
                else:
                    # Map to the CATH class's new ID in the target vocab
                    target_str = '.'.join(h_level_str.split('.')[:depth])
                    h_to_target_map[h_level_int] = target_vocab[target_str]
            self.target_maps[level_name] = h_to_target_map

            # 3. Build the mapping matrix using the new, compact target IDs
            mapping_matrix = np.zeros((num_h_classes, num_target_classes), dtype=np.float32)
            for h_level_int, target_level_int in h_to_target_map.items():
                # This will now work because target_level_int is always a valid index
                mapping_matrix[h_level_int, target_level_int] = 1.0
            self.mapping_matrices[level_name] = mapping_matrix

    def map_label(self, h_level_label_int: int, target_hierarchy: str = 'H') -> int:
        """Maps a single integer label from H-level to a target hierarchy."""
        if target_hierarchy not in self.target_maps:
            raise ValueError(f"Target hierarchy must be one of {list(self.hierarchy_levels.keys())}")
        return self.target_maps[target_hierarchy][h_level_label_int]

    def map_labels(self, labels, target_hierarchy: str = 'H') -> np.ndarray:
        if target_hierarchy not in self.target_maps:
            raise ValueError(f"Target hierarchy must be one of {list(self.hierarchy_levels.keys())}")
        mapped_labels = [self.map_label(label, target_hierarchy) for label in labels]
        return np.array(mapped_labels)

    def map_probabilities(self, h_level_probs, target_hierarchy: str = 'H'):
        """
        Maps a probability distribution from H-level classes to target-level classes
        by summing probabilities of related classes.

        Args:
            h_level_probs (np.ndarray or torch.Tensor): Probabilities array of shape
                (protein_length, num_h_level_classes).
            target_hierarchy (str): The target hierarchy level ('C', 'A', 'T', or 'H').

        Returns:
            np.ndarray or torch.Tensor: The mapped probabilities of shape
                (protein_length, num_target_classes).
        """
        if target_hierarchy not in self.mapping_matrices:
            raise ValueError(f"Target hierarchy must be one of {list(self.hierarchy_levels.keys())}")

        # Select the pre-computed matrix for the target hierarchy
        mapping_matrix = self.mapping_matrices[target_hierarchy]

        # Perform the matrix multiplication
        if isinstance(h_level_probs, torch.Tensor):
            device = h_level_probs.device
            # Move matrix to the same device as the tensor
            mapping_matrix_tensor = torch.from_numpy(mapping_matrix).to(device)
            target_probs = h_level_probs @ mapping_matrix_tensor
        else:  # Assume numpy array
            target_probs = h_level_probs @ mapping_matrix

        return target_probs

    def map_y_true_labels_y_pred_confidences_lists(self, y_true_labels_list, y_pred_confidences_list,
                                                   target_hierarchy: str = 'H'):
        if target_hierarchy not in self.target_maps:
            raise ValueError(f"Target hierarchy must be one of {list(self.hierarchy_levels.keys())}")
        y_true_labels_list_mapped = [self.map_labels(y_true_labels, target_hierarchy) for y_true_labels in
                                     y_true_labels_list]

        y_pred_confidences_list_mapped = [self.map_probabilities(y_pred_confidences, target_hierarchy) for
                                          y_pred_confidences in
                                          y_pred_confidences_list]
        return y_true_labels_list_mapped, y_pred_confidences_list_mapped

    def get_class_count(self, target_hierarchy: str = 'H') -> int:
        """Gets the total number of unique classes for a target hierarchy."""
        return self.target_class_count[target_hierarchy]

    def get_special_labels_map(self, target_hierarchy: str = 'T') -> dict:
        """
        Gets a map of special label names to their corresponding integer IDs
        at a specific target hierarchy.

        Args:
            target_hierarchy (str): The target hierarchy level ('C', 'A', 'T', or 'H').

        Returns:
            dict: A dictionary mapping special label strings to their new integer IDs.
                  Example: {'PADDING_REGION': 0, 'NO_DOMAIN_REGION': 1}
        """
        special_ids_map = {}
        # Get all classes known by the original encoder
        h_level_classes = list(self.h_level_encoder.classes_)

        for label_str in self.special_labels:
            try:
                # Find the original H-level ID of the special label
                h_level_id = h_level_classes.index(label_str)

                # Map this H-level ID to the target hierarchy's ID
                target_id = self.map_label(h_level_id, target_hierarchy=target_hierarchy)
                special_ids_map[label_str] = target_id

            except ValueError:
                # This handles cases where a defined special label (e.g., "NO_DOMAIN_REGION")
                # might not have been present in the specific dataset used to fit the encoder.
                continue

        return special_ids_map
