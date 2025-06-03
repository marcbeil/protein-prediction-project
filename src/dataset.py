import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class CathPredDomainDataset(Dataset):
    """
    A PyTorch Dataset for loading protein domain embeddings and their corresponding labels.
    It expects full protein embeddings and uses domain_start/end to slice the domain.
    """

    def __init__(self, data_df: pd.DataFrame, onehot_encoder: LabelEncoder, target_label_cols: str,
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
            onehot_encoder (sklearn.preprocessing.LabelEncoder): A pre-fitted LabelEncoder
                                                                instance to transform string labels
                                                                into numerical IDs consistently.
            target_label_col (str): The name of the column in `data_df` that contains the
                                    target labels for classification (e.g., 'class.architecture').
        """
        self.data_df = data_df
        self.embedding_dir = embedding_dir
        self.label_encoder = onehot_encoder
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
        reshaped_labels = np.array(self.labels).reshape(-1, 1)
        if fit:
            onehot_encoder.fit(reshaped_labels)
        # Encode the target labels using the provided pre-fitted LabelEncoder
        self.encoded_labels = self.label_encoder.transform(reshaped_labels).toarray().tolist()

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
                   - dict: A dictionary with the 'embedding' tensor for the domain.
                   - torch.Tensor: The encoded numerical label for the domain.
        """
        domain_id = self.domain_ids[index]
        domain_start_idx = self.domain_starts[index]
        domain_end_idx = self.domain_ends[index]
        encoded_label = self.encoded_labels[index]

        path_to_embedding = os.path.join(self.embedding_dir, f"{domain_id}.npy")

        full_protein_embedding = np.load(path_to_embedding)

        if domain_start_idx < 0 or domain_end_idx > full_protein_embedding.shape[
            0] or domain_start_idx >= domain_end_idx:
            print(
                f"Warning: Invalid domain indices for {domain_id}: start={domain_start_idx}, end={domain_end_idx}. Skipping this sample.")

        domain_embedding = full_protein_embedding[domain_start_idx:domain_end_idx, :]

        x = {
            "embedding": torch.tensor(domain_embedding, dtype=torch.float32)
        }

        y = torch.tensor(encoded_label, dtype=torch.long)
        return x, y
