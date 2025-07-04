import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, random_split
from dataset import DomainBoundaryDataset
from model import DomainBoundaryCNN
import numpy as np

def train(model, dataloader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    for x, y in dataloader:
        x = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
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
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)


def predict_and_save(model, dataloader, device, output_path):
    model.eval()
    predictions = []

    with torch.no_grad():
        for x, y in dataloader:
            x = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}

            logits = model(x)  # shape: (1, L, C)
            preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()  # shape: (L,)
            domain_id = x["domain_id"]  # assuming it's passed in x

            predictions.append({
                "domain_id": domain_id,
                "prediction": preds.tolist()
            })

    # Save as a CSV
    pd.DataFrame(predictions).to_json(output_path, orient="records", lines=True)
    print(f"Test predictions saved to: {output_path}")


if __name__ == "__main__":
    df_path = "/Users/b.madran/master/protein-prediction-project/data/subset50.cvs"
    embedding_dir = "/Users/b.madran/master/protein-prediction-project/data/prot_embeddings"
    batch_size = 2
    learning_rate = 1e-3
    num_epochs = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(df_path)
    dataset = DomainBoundaryDataset(df, embedding_dir=embedding_dir)

    # Split into train/val/test (80/10/10)
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len

    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len],
                                                generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = DomainBoundaryCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")

    # Save test predictions
    output_json = "/Users/b.madran/master/protein-prediction-project/data/test_predictions.json"
    predict_and_save(model, test_loader, device, output_json)
