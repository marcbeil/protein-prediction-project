import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from config import Config
from model import CathPred


def train(model, train_dataloader, optimizer, loss_fn, device):
    # Set model to training mode
    model.train()
    # Initialize running loss function value
    train_loss = 0.0

    # Loop over the batches of data
    for x, y in train_dataloader:
        for k, v in x.items():
            x[k] = v.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass: get outputs by passing inputs to the model
        with torch.autocast("cuda"):
            outputs = model(x)
            # Compute loss: compare outputs with labels
            loss = loss_fn(outputs, y)

        # Backward pass: compute gradients of the loss with respect to model parameters
        loss.backward()
        # Update parameters: perform a single optimization step (parameter update)
        optimizer.step()

        # Record statistics
        train_loss += loss.item()  # * x.size(0) # total loss of the batch (not averaged) #TODO: batch = 1 irrelevant

    train_loss = train_loss / len(
        train_dataloader.dataset)  # loss averaged across all training examples for the current epoch

    return train_loss


def evaluate(model, val_dataloader, loss_fn, device):
    # Set model to evaluation mode
    model.eval()
    # Initialize validation loss
    val_loss = 0.0

    # Disable gradient computation and turn it back on after the validation loop is finished
    with torch.no_grad():
        for x, y in val_dataloader:
            for k, v in x.items():
                x[k] = v.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast("cuda"):
                outputs = model(x)
                loss = loss_fn(outputs, y)

            val_loss += loss.item()  # * x.size(0) #TODO: batch = 1 irrelevant

    val_loss = val_loss / len(val_dataloader.dataset)

    return val_loss


def make_prediction(model, test_dataloader, device):
    # Initialize lists or tensors to store outputs and labels
    pred_list = []
    target_list = []

    # Set model to evaluation mode
    model.eval()

    # Disable gradient computation
    with torch.no_grad():
        # Loop over the batches of test data
        for x_test, y_test in test_dataloader:
            for k, v in x_test.items():
                x_test[k] = v.to(device, non_blocking=True)
            y_test = y_test.to(device, non_blocking=True)

            # Forward pass: get outputs by passing inputs to the model
            with torch.autocast("cuda"):
                # Get raw logits from the model
                logits = model(x_test)
                # For classification, get the predicted class index
                pred = torch.argmax(logits, dim=1)

            # Append outputs and labels to lists or tensors
            pred_list.append(pred.cpu())
            target_list.append(y_test.cpu())

    pred_tensor = torch.cat(pred_list, dim=0)
    target_tensor = torch.cat(target_list, dim=0)

    return pred_tensor, target_tensor


class CathPredDataset(Dataset):
    def __init__(self, domain_id, domain_start, domain_end, y, embedding_path="data/embeddings/domain_embeddings"):
        self.domain_id = list(domain_id)  # Convert to list for easier indexing if needed
        self.domain_start = torch.tensor(domain_start, dtype=torch.int64)
        self.domain_end = torch.tensor(domain_end, dtype=torch.int64)
        self.y = torch.tensor(y, dtype=torch.int64)
        self.embedding_path = embedding_path

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        path_to_embedding = os.path.join(self.embedding_path, f"{self.domain_id[index]}.npy")
        embedding = torch.load(path_to_embedding)
        x = {
            "embedding": embedding
        }
        y = self.y[index]
        return x, y


def multiLevelCATHLoss(class_pred, class_true, weights=[1, 1, 1, 1]):
    # TODO: currently only class level
    loss = torch.nn.CrossEntropyLoss()
    total_loss = 0
    for pred, true, weight in zip(class_pred, class_true, weights):
        total_loss += weight * loss(pred, true)
    return total_loss


def create_prediction_df(dataset, predictions, targets):
    """Creates a Pandas DataFrame from the dataset, predictions, and targets."""
    df = pd.DataFrame({
        "domain_id": dataset.domain_id,
        "start": dataset.domain_start.numpy(),
        "end": dataset.domain_end.numpy(),
        "target": dataset.y.numpy(),
        "prediction": predictions.numpy()
    })
    return df


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    test = parser.add_argument_group(title='Test parameters',
                                     description='Parameters for testing model')
    test.add_argument('-e', '--epoch', default=50, type=int,
                      help='Epoch number for model training')
    test.add_argument('-b', '--batch', default=1, type=int,
                      help='Batch size for model training')
    test.add_argument('-s', '--split', default=0.7, type=float,
                      help='Proportion for the training dataset')
    test.add_argument('-l', '--learning', default=0.0005, type=float,
                      help='Learning rate for model training')
    test.add_argument('-d', '--weight_decay', default=0.01, type=float,
                      help='Weight decay (L2 penalty)')
    test.add_argument('-w', '--warmup', default=5, type=int,
                      help='Warm-up epochs for model training')
    test.add_argument('-save', '--save', action='store_true',
                      help='Save the model and training results')

    run = parser.add_argument_group(title='Prediction parameters',
                                    description='Parameters for Prediction using model')
    run.add_argument('-o', '--output', help='Output pytorch model')
    run.add_argument('-i', '--input_folder', help='Input data folder')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("--------------------------------------------------\nData loading")
    path = ("data/datasets")
    print(path)
    data_set_path = os.path.join(path, "subset_with_protein_embedding.csv")
    print(f"Loading dataset: {data_set_path}")

    dataset = pd.read_csv(data_set_path)
    x_c = dataset[["domain_id", "domain_start", "domain_end"]]
    y_c = dataset[["class"]]
    assert x_c.shape[0] == y_c.shape[0]
    data_size = x_c.shape[0]
    num_train = int(args.split * data_size)
    num_val = (data_size - num_train) // 2

    # model configs
    model_config = Config()
    with open(os.path.join(path, "config.txt"), "w") as f:
        f.write(model_config.__repr__())
    print(y_c.head())
    # Reformat y to numpy matrix
    ymat = y_c["class"].to_numpy()  # TODO: only single class

    # randomly shuffle the data
    indices = np.random.default_rng(seed=42).permutation(data_size)
    training_idx, test_idx, val_idx = np.split(indices, [num_train, num_train + num_val])

    x_train, y_train = x_c.iloc[training_idx], ymat[training_idx]
    x_val, y_val = x_c.iloc[val_idx], ymat[val_idx]
    x_test, y_test = x_c.iloc[test_idx], ymat[test_idx]

    print(len(x_train), "Training sequences")
    print(len(x_val), "Validation sequences")
    print(len(x_test), "Testing sequences")

    # initiate the model
    model = CathPred(model_config)
    model.to(device)

    # Create DataLoaders
    train_dataset = CathPredDataset(
        domain_id=x_train["domain_id"].values,
        domain_start=x_train["domain_start"].values,
        domain_end=x_train["domain_end"].values,
        y=y_train
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, pin_memory=True, num_workers=11)

    val_dataset = CathPredDataset(
        domain_id=x_val["domain_id"].values,
        domain_start=x_val["domain_start"].values,
        domain_end=x_val["domain_end"].values,
        y=y_val
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch, pin_memory=True, num_workers=11)

    test_dataset = CathPredDataset(
        domain_id=x_test["domain_id"].values,
        domain_start=x_test["domain_start"].values,
        domain_end=x_test["domain_end"].values,
        y=y_test
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch, pin_memory=True, num_workers=11)

    # Define optimizer with a scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning, weight_decay=args.weight_decay)
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, total_iters=args.warmup)
    train_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01,
                                                  total_iters=(args.epoch - args.warmup))
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, train_scheduler], [args.warmup])

    # Define loss criterion
    loss_fn = multiLevelCATHLoss

    print("--------------------------------------------------\nTraining")

    output_path = os.path.join(path, "output")
    # Training loop. 1 epoch = 1 Loop over the dataset:
    best_loss = np.inf
    best_epoch = 0
    for epoch in range(args.epoch):
        train_loss = train(model, train_dataloader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_dataloader, loss_fn, device)
        scheduler.step()

        print(f"Epoch {epoch + 1}/{args.epoch}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print()

        # save best performance
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            if args.save:
                torch.save(model.state_dict(), os.path.join(output_path, "best_model.pt"))

        # early stopping if loss does not imporve for a patience of 10 epochs
        if (epoch - best_epoch) == 10:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # load best model
    model.load_state_dict(torch.load(os.path.join(output_path, "best_model.pt")))

    # output all the correlations
    out_tensors_test, target_tensors_test = make_prediction(model, test_dataloader, device)
    testdf = create_prediction_df(test_dataset, out_tensors_test, target_tensors_test)
    testdf.to_parquet(os.path.join(output_path, 'model_prediction.parquet'))
    print(f"Test set predictions saved to: {os.path.join(output_path, 'model_prediction.parquet')}")

    train_unshuffled_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=False, pin_memory=True, num_workers=14)
    out_tensors_train, target_tensors_train = make_prediction(model, train_unshuffled_dataloader, device)
    traindf = create_prediction_df(train_dataset, out_tensors_train, target_tensors_train)
    traindf.to_parquet(os.path.join(output_path, 'model_trainingset.parquet'))
    print(f"Training set predictions saved to: {os.path.join(output_path, 'model_trainingset.parquet')}")

    print("--------------------------------------------------\nFinished!")


if __name__ == '__main__':
    main()