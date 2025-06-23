import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LABELS = {
    'NO_DOMAIN_REGION': 0,
    'DOMAIN_START': 1,
    'DOMAIN_MIDDLE': 2,
    'DOMAIN_END': 3,
}


class DomainBoundaryDataset(Dataset):
    def __init__(self, csv_or_df, embedding_dir: str):
        if isinstance(csv_or_df, str):
            df = pd.read_csv(csv_or_df)
        else:
            df = csv_or_df

        self.embedding_dir = embedding_dir
        self.protein_to_domains = {}

        for _, row in df.iterrows():
            pid = row["domain_id"]
            start = int(row["domain_start"]) - 1  # convert to 0-based indexing
            end = int(row["domain_end"])

            if pid not in self.protein_to_domains:
                self.protein_to_domains[pid] = []
            self.protein_to_domains[pid].append((start, end))

        self.protein_ids = list(self.protein_to_domains.keys())

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self, idx):
        protein_id = self.protein_ids[idx]
        embedding_path = os.path.join(self.embedding_dir, f"{protein_id}.npy")

        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Missing embedding: {embedding_path}")

        embedding = np.load(embedding_path)
        L = embedding.shape[0]

        y = np.full(L, LABELS['NO_DOMAIN_REGION'], dtype=np.int64)

        for start, end in self.protein_to_domains[protein_id]:
            if end - start == 1:
                y[start] = LABELS['DOMAIN_START']
            elif end - start == 2:
                y[start] = LABELS['DOMAIN_START']
                y[start + 1] = LABELS['DOMAIN_END']
            else:
                y[start] = LABELS['DOMAIN_START']
                y[start + 1:end - 1] = LABELS['DOMAIN_MIDDLE']
                y[end - 1] = LABELS['DOMAIN_END']
    
        x = {"domain_id": protein_id ,"embedding": torch.tensor(embedding, dtype=torch.float32)}
        y = torch.tensor(y, dtype=torch.long)
        return x, y
