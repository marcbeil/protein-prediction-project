import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader

from dataset import CathPredDomainDataset
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
    run.add_argument('-t', '--target-label-cols', help='Output pytorch model',
                     choices=['class', 'class.architecture', 'class.architecture.topology',
                              'class.architecture.topology.homology'], default='class.architecture')
    run.add_argument('-o', '--output', help='Output pytorch model', default='output')
    run.add_argument('-i', '--input_folder', help='Input data folder', default="datasets/v1")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_folder = args.input_folder
    train_path, val_path = os.path.join(input_folder, 'train_split.csv'), os.path.join(input_folder, 'val_split.csv')
    print("--------------------------------------------------\nData loading")
    onehot_encoder = OneHotEncoder()
    train_dataset = pd.read_csv(train_path)
    train_dataset = CathPredDomainDataset(train_dataset, onehot_encoder, args.target_label_cols, fit=True)

    val_dataset = pd.read_csv(val_path)
    val_dataset = CathPredDomainDataset(val_dataset, onehot_encoder, args.target_label_cols)

    num_classes = onehot_encoder.categories_[0].shape[0]
    # initiate the model
    model = CathPred(num_classes=num_classes)
    model.to(device)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, pin_memory=True, num_workers=11)

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch, pin_memory=True, num_workers=11)

    # Define optimizer with a scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning, weight_decay=args.weight_decay)
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, total_iters=args.warmup)
    train_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01,
                                                  total_iters=(args.epoch - args.warmup))
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, train_scheduler], [args.warmup])

    # Define loss criterion
    loss_fn = torch.nn.CrossEntropyLoss()

    print("--------------------------------------------------\nTraining")

    output_path = args.output
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

    train_unshuffled_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=False, pin_memory=True,
                                             num_workers=14)
    out_tensors_train, target_tensors_train = make_prediction(model, train_unshuffled_dataloader, device)
    traindf = create_prediction_df(train_dataset, out_tensors_train, target_tensors_train)
    traindf.to_parquet(os.path.join(output_path, 'model_trainingset.parquet'))
    print(f"Training set predictions saved to: {os.path.join(output_path, 'model_trainingset.parquet')}")

    print("--------------------------------------------------\nFinished!")


if __name__ == '__main__':
    main()
