import argparse
import os
import re
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer


class ProtT5Embeddings():
    def __init__(self, model_name="Rostlab/prot_t5_xl_half_uniref50-enc"):
        self.model_name = model_name

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (Apple Silicon GPU) for training.")
        else:
            self.device = torch.device("cpu")

        # Load the pre-trained model and tokenizer
        print(f"Loading model: {model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("Model loaded successfully")

    def embed_sequences(self, sequences):
        # this will replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
        sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]

        # tokenize sequences and pad up to the longest sequence in the batch
        ids = self.tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device)

        # generate embeddings
        with torch.no_grad():
            embedding_repr = self.model(input_ids=input_ids, attention_mask=attention_mask)

        return embedding_repr.last_hidden_state


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="Generate ProtT5 embeddings for protein sequences from a CSV file.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    input_group = parser.add_argument_group(title='Input',
                                            description='Input data parameters')
    input_group.add_argument('-i', '--input-file', type=str, required=True,
                             help='Path to the input CSV file containing protein sequences and IDs.')

    output_group = parser.add_argument_group(title='Output',
                                             description='Output parameters')
    output_group.add_argument('-o', '--output-folder', type=str, required=True,
                              help='Path to the output folder where embeddings will be saved as .npy files.')

    model_group = parser.add_argument_group(title='Model',
                                            description='ProtT5 model parameters')
    model_group.add_argument('-m', '--model-name', default="Rostlab/prot_t5_xl_half_uniref50-enc", type=str,
                             help='Name or path of the ProtT5 encoder model to use.')

    processing_group = parser.add_argument_group(title='Processing',
                                                 description='Processing parameters')
    processing_group.add_argument('-b', '--batch-size', default=16, type=int,
                                  help='Batch size for processing sequences.')
    processing_group.add_argument('--overwrite', action='store_true',
                                  help='Overwrite existing embedding files. If set, existing files will be overwritten.')

    args = parser.parse_args()

    cwd = os.getcwd()

    dataset = pd.read_csv(os.path.join(cwd, args.input_file))
    dataset['protein_sequence_length'] = dataset['protein_sequence'].str.len()
    output_dir = os.path.join(cwd, args.output_folder)
    os.makedirs(output_dir, exist_ok=True)

    # Store the count of sequences that will actually be processed
    sequences_to_embed_count = len(dataset)

    if not args.overwrite:
        already_embedded_set = {filename[:-4] for filename in os.listdir(output_dir)}
        original_len = len(dataset)
        dataset = dataset[~dataset['domain_id'].isin(already_embedded_set)]
        print(
            f"{original_len - len(dataset)} sequences were already embedded. {len(dataset)} sequences will be embedded now.")
        sequences_to_embed_count = len(dataset)
        if len(dataset) == 0:
            print("All sequences already embedded")
            return

    dataset.sort_values(by='protein_sequence_length', inplace=True)

    protT5Embeddings = ProtT5Embeddings(model_name=args.model_name)

    for i in tqdm(range(0, len(dataset), args.batch_size), desc="Processing batches"):
        batch_df = dataset[i:i + args.batch_size]
        prot_sequences = batch_df['protein_sequence'].tolist()
        domain_ids = batch_df['domain_id'].tolist()

        if prot_sequences:
            embeddings = protT5Embeddings.embed_sequences(prot_sequences)
            for j, domain_id in enumerate(domain_ids):
                prot_sequence = prot_sequences[j]
                protein_embedding = embeddings[j, :len(prot_sequence)].cpu()
                protein_embedding_path = os.path.join(output_dir, f"{domain_id}")
                np.save(protein_embedding_path, protein_embedding)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

    if sequences_to_embed_count > 0:
        avg_time_per_embedding = total_time / sequences_to_embed_count
        print(f"Average time per embedding: {avg_time_per_embedding:.4f} seconds")
    else:
        print("No new sequences were embedded.")


if __name__ == '__main__':
    main()
