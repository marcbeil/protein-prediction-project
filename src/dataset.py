import os
from typing import Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder  # Keep LabelEncoder
# from sklearn.preprocessing import OneHotEncoder # REMOVE THIS IMPORT
from torch.utils.data import Dataset


class CathPredDomainDataset(Dataset):
    """
    A PyTorch Dataset for loading protein domain embeddings and their corresponding labels.
    It expects full protein embeddings and uses domain_start/end to slice the domain.
    """

    def __init__(self, data_df: Union[pd.DataFrame, str], label_encoder: LabelEncoder, target_label_cols: str,
                 embedding_dir: str = "data/embeddings/protein_embeddings", fit=False):
        """
        Initializes the CathPredDataset.

        Args:
            data_df (pd.DataFrame): A pandas DataFrame containing the data for this split
                                    (e.g., train_split.csv, val_split.csv, test_split.csv).
                                    Must contain 'domain_id', 'domain_start', 'domain_end',
                                    and the column specified by `target_label_col`.
            embedding_dir (str): Path to the directory where the full protein embeddings
                                 (.npy files) are stored.
            label_encoder (sklearn.preprocessing.LabelEncoder): A pre-fitted LabelEncoder
                                                                instance to transform string labels
                                                                into numerical IDs consistently.
            target_label_col (str): The name of the column in `data_df` that contains the
                                    target labels for classification (e.g., 'class.architecture').
        """
        if type(data_df) == str:
            data_df = pd.read_csv(data_df)
        self.data_df = data_df
        self.embedding_dir = embedding_dir
        self.label_encoder = label_encoder  # Rename parameter to label_encoder
        target_label_col_list = target_label_cols.split('.')

        data_df['__combined_label_temp_col__'] = data_df[target_label_col_list[0]].astype(str)
        for col in target_label_col_list[1:]:
            data_df['__combined_label_temp_col__'] = data_df['__combined_label_temp_col__'] + '.' + data_df[col].astype(
                str)
        self.labels = data_df['__combined_label_temp_col__'].tolist()
        # Extract relevant columns as lists for efficient indexing
        self.domain_ids = self.data_df['domain_id'].tolist()
        # Convert domain_start and domain_end to 0-indexed integers for slicing
        # Assuming domain_start and domain_end from CSV are 1-indexed as commonly seen in biological data
        self.domain_starts = (self.data_df['domain_start'] - 1).tolist()
        self.domain_ends = self.data_df['domain_end'].tolist()

        # Reshape labels to fit LabelEncoder's expected input (1D array or column vector)
        # LabelEncoder expects a 1D array-like, not a 2D array or list of lists for fit/transform
        if fit:
            self.label_encoder.fit(self.labels)  # Fit on the 1D list of labels
        # Encode the target labels using the provided pre-fitted LabelEncoder
        # The output of transform is a 1D numpy array of integers
        self.encoded_labels = self.label_encoder.transform(self.labels).tolist()

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data_df)

    def __getitem__(self, index: int):
        """
        Retrieves a single sample from the dataset at the given index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                   - dict: A dictionary with the 'embedding' tensor for the domain. (protein_length * 1024)
                   - torch.Tensor: The encoded numerical label for the domain.
        """
        domain_id = self.domain_ids[index]
        domain_start_idx = self.domain_starts[index]
        domain_end_idx = self.domain_ends[index]
        encoded_label = self.encoded_labels[index]  # This will now be a single integer

        path_to_embedding = os.path.join(self.embedding_dir, f"{domain_id}.npy")

        # Handle missing embedding files
        if not os.path.exists(path_to_embedding):
            # Fallback or raise error, depending on desired behavior
            # For now, let's return None or a placeholder for this sample,
            # which will need handling in the DataLoader or collation.
            # A better approach might be to filter out invalid samples during dataset creation.
            print(f"Error: Embedding file not found for {domain_id} at {path_to_embedding}. Skipping.")
            return None, None  # Returning None will cause issues in DataLoader if not handled by a custom collate_fn

        full_protein_embedding = np.load(path_to_embedding)

        # Updated: Log warning for invalid domain indices and skip if severe issues are found
        # (e.g., start >= end implies empty or invalid slice)
        if domain_start_idx < 0 or domain_end_idx > full_protein_embedding.shape[
            0] or domain_start_idx >= domain_end_idx:
            print(
                f"Warning: Invalid domain indices for {domain_id}: start={domain_start_idx}, end={domain_end_idx}, shape={full_protein_embedding.shape}. Skipping this sample.")
            # Return None or handle appropriately to prevent errors later.
            # For this example, we'll return a placeholder, ideally, filter these out beforehand.
            return None, None

        domain_embedding = full_protein_embedding[domain_start_idx:domain_end_idx, :]

        x = {
            "embedding": torch.tensor(domain_embedding, dtype=torch.float32)
        }

        # 'y' will now be a single integer, which is what CrossEntropyLoss expects.
        y = torch.tensor(encoded_label, dtype=torch.long)
        return x, y
