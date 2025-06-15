import os
from typing import Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class CathPredPerResidueDataset(Dataset):
    NO_DOMAIN_LABEL = 'NO_DOMAIN_REGION'

    def __init__(self, data_df: Union[pd.DataFrame, str], label_encoder: LabelEncoder,
                 embedding_dir: str = "data/embeddings/protein_embeddings", fit: bool = False):
        if isinstance(data_df, str):
            data_df = pd.read_csv(data_df)
        self.data_df = data_df
        self.embedding_dir = embedding_dir
        self.label_encoder = label_encoder

        cath_labels = sorted(list(self.data_df['cath'].unique()))

        all_labels_to_encode = cath_labels + [self.NO_DOMAIN_LABEL]

        if fit:
            self.label_encoder.fit(all_labels_to_encode)
        self.num_classes = len(self.label_encoder.classes_)

        # Store the encoded ID for the 'no-domain' class for easy access
        self.no_domain_encoded_id = self.label_encoder.transform([self.NO_DOMAIN_LABEL])[0]

        self.encoded_labels = self.label_encoder.transform(self.data_df['cath'].tolist())

        self.domain_ids = self.data_df['domain_id'].tolist()
        # Convert domain_start and domain_end to 0-indexed integers for slicing
        # Assuming domain_start and domain_end from CSV are 1-indexed as commonly seen in biological data
        self.domain_starts = (self.data_df['domain_start'] - 1).tolist()
        self.domain_ends = self.data_df['domain_end'].tolist()

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index: int):
        domain_id = self.domain_ids[index]
        # This is the CATH class ID for the specific domain
        encoded_cath_label = self.encoded_labels[index]

        path_to_embedding = os.path.join(self.embedding_dir, f"{domain_id}.npy")
        if not os.path.exists(path_to_embedding):
            raise FileNotFoundError(f"Embedding file not found: {path_to_embedding}")

        full_protein_embedding = np.load(path_to_embedding)
        full_seq_len = full_protein_embedding.shape[0]

        domain_start_idx = self.domain_starts[index]
        domain_end_idx = self.domain_ends[index]

        # Validate domain indices
        if (domain_start_idx < 0 or
                domain_end_idx > full_seq_len or
                domain_start_idx >= domain_end_idx):
            raise ValueError(
                f"Invalid domain indices for {domain_id}: start={domain_start_idx}, end={domain_end_idx}, "
                f"full_protein_embedding shape={full_protein_embedding.shape}. "
                f"Check domain_start/end values or embedding integrity."
            )

        x = {
            "embedding": torch.tensor(full_protein_embedding, dtype=torch.float32)
        }

        y_full_protein = torch.full((full_seq_len,), fill_value=self.no_domain_encoded_id, dtype=torch.long)

        y_full_protein[domain_start_idx:domain_end_idx] = encoded_cath_label

        return x, y_full_protein


def create_protein_collate_fn(global_max_len: int, no_domain_encoded_id: int):
    def protein_collate_fn(batch):
        embeddings = []
        labels = []

        # Iterate through each item in the batch to collect embeddings and labels
        for embedding_dict, label_tensor in batch:
            embeddings.append(embedding_dict["embedding"])
            labels.append(label_tensor)

        # Determine the embedding dimension from the first embedding in the batch
        if not embeddings:
            raise ValueError("Batch is empty, cannot determine embedding dimension.")
        embedding_dim = embeddings[0].shape[1]

        # Manually pad each embedding tensor to the global_max_len
        padded_embeddings = []
        for emb_tensor in embeddings:
            current_length = emb_tensor.shape[0]

            if current_length < global_max_len:
                # Calculate the amount of padding needed
                padding_needed = global_max_len - current_length
                # Create a tensor of zeros for padding, matching the embedding's dtype and device
                padding = torch.zeros((padding_needed, embedding_dim),
                                      dtype=emb_tensor.dtype,
                                      device=emb_tensor.device)
                # Concatenate the original embedding with the padding
                padded_emb = torch.cat([emb_tensor, padding], dim=0)
            else:
                # If current_length is already global_max_len or larger, truncate it
                padded_emb = emb_tensor[:global_max_len]
            padded_embeddings.append(padded_emb)

        # Stack all the padded embedding tensors into a single batch tensor
        batch_embeddings = torch.stack(padded_embeddings, dim=0)

        # Manually pad each label tensor to the global_max_len
        padded_labels = []
        for label_tensor in labels:
            current_label_length = label_tensor.shape[0]
            if current_label_length < global_max_len:
                padding_needed = global_max_len - current_label_length
                # Create a tensor filled with the no_domain_encoded_id for padding labels
                padding = torch.full((padding_needed,), fill_value=no_domain_encoded_id,
                                     dtype=label_tensor.dtype,
                                     device=label_tensor.device)
                # Concatenate the original label tensor with the padding
                padded_label = torch.cat([label_tensor, padding], dim=0)
            else:
                # If current_label_length is already global_max_len or larger, truncate it
                padded_label = label_tensor[:global_max_len]
            padded_labels.append(padded_label)

        # Stack all the padded label tensors into a single batch tensor
        batch_labels = torch.stack(padded_labels, dim=0)

        return {"embedding": batch_embeddings}, batch_labels

    return protein_collate_fn
