import torch, os
import argparse
from datetime import datetime
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, random_split
from dataset import DomainBoundaryDataset
from model import DomainBoundaryCNN
from torch.utils.tensorboard import SummaryWriter

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
            domain_id = x["domain_id"][0]  # assuming it's passed in x

            predictions.append({
                "domain_id": domain_id,
                "prediction": preds.tolist()
            })

    # Save as a CSV
    pd.DataFrame(predictions).to_json(output_path, orient="records", lines=True)
    print(f"Test predictions saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter
        )

    test = parser.add_argument_group(title = 'Test parameters',
                                     description = 'Parameters for testing model')
    test.add_argument('-e', '--epoch', default = 50, type = int, 
                      help = 'Epoch number for model training')
    test.add_argument('-b', '--batch', default = 1, type = int, 
                      help = 'Batch size for model training')
    test.add_argument('-s', '--split', default = 0.8, type = float,
                      help = 'Proportion for the training dataset')
    test.add_argument('-l', '--learning', default = 0.0005, type = float, 
                      help = 'Learning rate for model training')
    test.add_argument('-d', '--weight_decay', default = 0.01, type = float, 
                      help = 'Weight decay (L2 penalty)') # TODO: not used currently
    test.add_argument('-w', '--warmup', default = 5, type = int, 
                      help = 'Warm-up epochs for model training') # TODO: not used currently
    
    args = parser.parse_args()

    df_path = "/Users/b.madran/master/protein-prediction-project/data/subset50.cvs"
    embedding_dir = "/Users/b.madran/master/protein-prediction-project/data/prot_embeddings"
    patience = 50  # Early stopping patience
    checkpoint_path = "best_model.pt"
    run_name = f"domain_boundary_lr{args.learning}_bs{args.batch}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    log_dir = os.path.join("runs", run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(df_path)
    dataset = DomainBoundaryDataset(df, embedding_dir=embedding_dir)

    num_train = int(args.split * len(dataset))
    num_val = (len(dataset) - num_train) // 2
    num_test = len(dataset) - num_train - num_val

    train_set, val_set, test_set = random_split(dataset, [num_train, num_val, num_test],
                                                generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch)
    test_loader = DataLoader(test_set, batch_size=args.batch)

    model = DomainBoundaryCNN().to(device)
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

    output_json = "/Users/b.madran/master/protein-prediction-project/data/test_predictions.json"
    predict_and_save(model, test_loader, device, output_json)
