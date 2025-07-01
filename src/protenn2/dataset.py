import os
from typing import Union, List, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class CathPredPerResidueDataset(Dataset):
    NO_DOMAIN_LABEL = 'NO_DOMAIN_REGION'
    PADDING_LABEL = 'PADDING_REGION'

    def __init__(self, data_df: Union[pd.DataFrame, str], label_encoder: LabelEncoder,
                 embedding_dir: str = "data/embeddings/protein_embeddings_new", fit: bool = False):
        if isinstance(data_df, str):
            data_df = pd.read_csv(data_df)
        self.data_df = data_df
        self.embedding_dir = embedding_dir
        self.label_encoder = label_encoder

        cath_labels = sorted(list(self.data_df['cath'].unique()))

        all_labels_to_encode = cath_labels + [self.NO_DOMAIN_LABEL, self.PADDING_LABEL]

        if fit:
            self.label_encoder.fit(all_labels_to_encode)
        self.num_classes = len(self.label_encoder.classes_)

        self.no_domain_encoded_id = self.label_encoder.transform([self.NO_DOMAIN_LABEL])[0]
        self.padding_encoded_id = self.label_encoder.transform([self.PADDING_LABEL])[0]

        self.protein_chain_data: Dict[str, List[Dict]] = {}
        for protein_chain_id, group in self.data_df.groupby('protein_chain_id'):
            self.protein_chain_data[protein_chain_id] = []
            for _, row in group.iterrows():
                domain_start_0_indexed = int(row['domain_start'] - 1)
                domain_end_0_indexed = int(row['domain_end'])
                encoded_cath_label = self.label_encoder.transform([row['cath']])[0]

                self.protein_chain_data[protein_chain_id].append({
                    'domain_id': row['domain_id'],
                    'domain_start': domain_start_0_indexed,
                    'domain_end': domain_end_0_indexed,
                    'encoded_cath_label': encoded_cath_label
                })

        # This list defines the order of proteins that will be yielded by the dataset
        self.protein_chain_ids_list = list(self.protein_chain_data.keys())
        print(f"Dataset initialized with {len(self.protein_chain_ids_list)} unique proteins.")

    def __len__(self):
        """
        Returns the number of unique proteins in the dataset.
        """
        return len(self.protein_chain_ids_list)

    def __getitem__(self, index: int):
        """
        Retrieves the embedding and per-residue labels for a full protein.
        """
        protein_chain_id = self.protein_chain_ids_list[index]

        # Get all domain information associated with this protein
        domains_info = self.protein_chain_data[protein_chain_id]

        # Construct the path to the full protein embedding file
        path_to_embedding = os.path.join(self.embedding_dir, f"{protein_chain_id}.npy")
        if not os.path.exists(path_to_embedding):
            raise FileNotFoundError(f"Embedding file not found for protein {protein_chain_id}: {path_to_embedding}")

        # Load the full protein embedding
        full_protein_embedding = np.load(path_to_embedding)
        full_seq_len = full_protein_embedding.shape[0]  # Length of the full protein sequence

        x = {
            "embedding": torch.tensor(full_protein_embedding, dtype=torch.float32)
        }

        # Initialize the full protein label tensor with 'NO_DOMAIN_LABEL'
        y_full_protein = torch.full((full_seq_len,), fill_value=self.no_domain_encoded_id, dtype=torch.long)

        # Iterate through all domains of the current protein and "paint" their labels
        for domain_info in domains_info:
            domain_start_idx = domain_info['domain_start']
            domain_end_idx = domain_info['domain_end']
            encoded_cath_label = domain_info['encoded_cath_label']

            # Validate domain indices against the full protein length
            # This handles cases where domain annotations might be out of bounds for the loaded embedding
            if (domain_start_idx < 0 or
                    domain_end_idx > full_seq_len or
                    domain_start_idx >= domain_end_idx):
                print(
                    f"Warning: Invalid domain indices for domain {domain_info['domain_id']} (protein {protein_chain_id}): "
                    f"start={domain_start_idx}, end={domain_end_idx}, "
                    f"full_protein_embedding shape={full_protein_embedding.shape}. "
                    f"This domain will be skipped for labeling.")
                continue  # Skip this specific invalid domain and proceed with others

            # Assign the domain's CATH label to its specific region
            # If domains overlap, the last one in the list (as they appear in the grouped data)
            # will overwrite previous ones for shared residues.
            y_full_protein[domain_start_idx:domain_end_idx] = encoded_cath_label

        # Return the embedding, the per-residue labels for the whole protein, and the protein_id itself
        return x, y_full_protein, protein_chain_id


def create_protein_collate_fn(global_max_len: int, padding_encoded_id: int):
    """
    Creates a collate function for DataLoader that pads protein embeddings and labels
    to a common length.
    """

    def protein_collate_fn(batch):
        embeddings = []
        labels = []
        protein_ids = []  # To store protein_ids from the batch

        # Iterate through each item in the batch (which now represents a full protein)
        for embedding_dict, label_tensor, protein_id_item in batch:
            embeddings.append(embedding_dict["embedding"])
            labels.append(label_tensor)
            protein_ids.append(protein_id_item)  # Collect protein_ids for tracking

        # Determine the embedding dimension from the first embedding in the batch
        if not embeddings:
            raise ValueError("Batch is empty, cannot determine embedding dimension.")
        embedding_dim = embeddings[0].shape[1]

        # Manually pad each embedding tensor to the global_max_len
        padded_embeddings = []
        for emb_tensor in embeddings:
            current_length = emb_tensor.shape[0]

            if current_length < global_max_len:
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
                padding = torch.full((padding_needed,), fill_value=padding_encoded_id,
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

        # Return the padded embeddings, padded labels, and the list of protein_ids
        return {"embedding": batch_embeddings}, batch_labels, protein_ids

    return protein_collate_fn
