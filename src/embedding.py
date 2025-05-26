import re

import torch
from transformers import T5EncoderModel, T5Tokenizer


class ProtT5Embeddings():
    def __init__(self, model_name="Rostlab/prot_t5_xl_half_uniref50-enc"):
        self.model_name = model_name

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        # embedding = embedding_repr.last_hidden_state[0, :len(sequence)]
        return embedding_repr.last_hidden_state
