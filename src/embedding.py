import torch
from transformers import T5Tokenizer, T5EncoderModel
from Bio import SeqIO

# Load ProtT5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
model = model.eval()

# Check if CUDA is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Read sequences from the FASTA file
sequences = list(SeqIO.parse("cath-50-samples.fa", "fasta"))

# Dictionary to store embeddings
all_embeddings = {}

# Process each sequence
for seq_record in sequences[:2]:
    sequence = str(seq_record.seq)
    sequence = sequence.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
    ids = tokenizer.batch_encode_plus([sequence], add_special_tokens=True, padding=True, return_tensors="pt")
    input_ids = ids['input_ids'].to(device)
    attention_mask = ids['attention_mask'].to(device)

    with torch.no_grad():
        embedding = model(input_ids=input_ids, attention_mask=attention_mask)

    # Average over tokens to get a single vector per sequence
    sequence_embedding = embedding.last_hidden_state.mean(dim=1).squeeze().cpu()

    # Store in dictionary
    all_embeddings[seq_record.id] = sequence_embedding

    print(f"Processed: {seq_record.id}")

# Save all embeddings to one file
torch.save(all_embeddings, "all_embeddings.pt")
print("All embeddings saved to all_embeddings.pt")
