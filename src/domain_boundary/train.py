import torch, os
import argparse
from datetime import datetime
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, random_split
from dataset import DomainBoundaryDataset
from model import DomainBoundaryCNN
from torch.utils.tensorboard import SummaryWriter

def create_protein_collate_fn(global_max_len: int, no_domain_encoded_id: int):
    def protein_collate_fn(batch):
        embeddings = []
        labels = []
        domain_ids = []

        # Iterate through each item in the batch to collect embeddings and labels
        for embedding_dict, label_tensor in batch:
            embeddings.append(embedding_dict["embedding"])
            labels.append(label_tensor)
            domain_ids.append(embedding_dict["domain_id"])

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

        return {"embedding": batch_embeddings, "domain_id": domain_ids }, batch_labels

    return protein_collate_fn

def train(model, dataloader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    for x, y in dataloader:
        x = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in dataloader:
            x = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}

            y = y.to(device)

            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)


def predict_and_save(model, dataloader, device, output_path):
    model.eval()
    predictions = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
            y = y.to(device)  # shape: (B, L)

            logits = model(x)  # shape: (B, L, C)
            preds = torch.argmax(logits, dim=-1)  # shape: (B, L)

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

            domain_ids = x["domain_id"]
            preds_np = preds.cpu().numpy()

            for domain_id, pred in zip(domain_ids, preds_np):
                predictions.append({
                    "domain_id": domain_id,
                    "prediction": pred.tolist()
                })

    # Save predictions
    pd.DataFrame(predictions).to_json(output_path, orient="records", lines=True)
    print(f"Test predictions saved to: {output_path}")

    # Compute domain detection accuracy
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    mask = (all_labels != 0)              # Only positions that are domain in ground truth
    detected = (all_preds != 0)           # Only positions predicted as domain (any label ≠ 0)
    correct_domain_detection = detected & mask

    accuracy = correct_domain_detection.sum().item() / mask.sum().item()

    print(f"Domain detection accuracy (pred ≠ 0 | true ≠ 0): {accuracy:.4f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter
        )

    test = parser.add_argument_group(title = 'Test parameters',
                                     description = 'Parameters for testing model')
    test.add_argument('-e', '--epoch', default = 50, type = int, 
                      help = 'Epoch number for model training')
    test.add_argument('-b', '--batch', default = 16, type = int,
                      help = 'Batch size for model training')
    test.add_argument('-s', '--split', default = 0.8, type = float,
                      help = 'Proportion for the training dataset')
    test.add_argument('-l', '--learning', default = 0.005, type = float,
                      help = 'Learning rate for model training')
    test.add_argument('-d', '--weight_decay', default = 0.01, type = float, 
                      help = 'Weight decay (L2 penalty)') # TODO: not used currently
    test.add_argument('-w', '--warmup', default = 5, type = int, 
                      help = 'Warm-up epochs for model training') # TODO: not used currently
    
    args = parser.parse_args()

    df_path = "/Users/bene/Developer/protein-prediction-project/data/subset_protein_mapped_enhanced_limited_len_600.csv"
    embedding_dir = "/Users/bene/Developer/protein-prediction-project/data/embeddings/protein_embeddings"
    patience = 3  # Early stopping patience
    checkpoint_path = "best_model.pt"
    run_name = f"domain_boundary_lr{args.learning}_bs{args.batch}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    log_dir = os.path.join("runs", run_name)

    device = "mps"#torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(df_path)
    dataset = DomainBoundaryDataset(df, embedding_dir=embedding_dir)

    num_train = int(args.split * len(dataset))
    num_val = (len(dataset) - num_train) // 2
    num_test = len(dataset) - num_train - num_val

    train_set, val_set, test_set = random_split(dataset, [num_train, num_val, num_test],
                                                generator=torch.Generator().manual_seed(42))
    
    max_protein_length = df["protein_length"].max()
    collate_fn = create_protein_collate_fn(max_protein_length, 0)

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch, collate_fn=collate_fn)

    model = DomainBoundaryCNN(max_length=max_protein_length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    writer = SummaryWriter(log_dir=log_dir)

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(1, args.epoch + 1):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)

        # Log to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")

        # Scheduler step
        scheduler.step(val_loss)

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ New best model saved at epoch {epoch} with val loss {val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"✗ No improvement for {epochs_without_improvement} epoch(s)")

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break
        
    writer.close()

    # Load the best model before testing
    model.load_state_dict(torch.load(checkpoint_path))

    output_json = "/Users/bene/Developer/protein-prediction-project/data/domain_boundary_predictions_2.json"
    predict_and_save(model, test_loader, device, output_json)
