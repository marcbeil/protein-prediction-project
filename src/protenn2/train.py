import argparse
import json
import os
import pickle

import pandas as pd
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CathPredPerResidueDataset, create_protein_collate_fn
from model import CathPredEnn2
from utils import get_train_val_test_paths, calculate_max_protein_length


def train(model, train_dataloader, optimizer, loss_fn, device):
    model.train()

    total_loss = 0.0

    all_preds = []
    all_labels = []

    for x, y in train_dataloader:
        for k, v in x.items():
            x[k] = v.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type):
            outputs = model(x)

            outputs = outputs.permute(0, 2, 1)

            loss = loss_fn(outputs, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(train_dataloader)

    return avg_loss, all_preds, all_labels


def evaluate(model, val_dataloader, loss_fn, device):
    model.eval()

    total_loss = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in val_dataloader:
            for k, v in x.items():
                x[k] = v.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type):
                outputs = model(x)
                outputs = outputs.permute(0, 2, 1)
                loss = loss_fn(outputs, y)

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(val_dataloader)

    return avg_loss, all_preds, all_labels


import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def calculate_metrics(predictions, true_labels, no_domain_encoded_id):
    """
    Calculates various classification metrics, excluding padded regions.

    Args:
        predictions (torch.Tensor or np.array): Model's predicted classes (e.g., after argmax).
                                                Expected shape: (N,) where N is total residues across batch/epoch.
        true_labels (torch.Tensor or np.array): True class labels.
                                                Expected shape: (N,) where N is total residues across batch/epoch.
        no_domain_encoded_id (int): The numerical ID used for padding 'no domain' regions.
                                    Labels with this ID will be excluded from metric calculation.

    Returns:
        dict: A dictionary of computed metrics.
    """
    # Ensure inputs are converted to numpy arrays if they are not already.
    # This handles both torch.Tensor and plain Python list inputs.
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    elif not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)

    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    elif not isinstance(true_labels, np.ndarray):
        true_labels = np.array(true_labels)

    # Create a mask to identify non-padded positions
    # We filter based on true_labels not being the padding value.
    # This ensures we only evaluate where there's actual domain information.
    mask = (true_labels != no_domain_encoded_id)

    # Apply the mask to both true_labels and predictions
    filtered_true_labels = true_labels[mask]
    filtered_predictions = predictions[mask]

    # Check if any non-padded labels remain after filtering
    if len(filtered_true_labels) == 0:
        print("Warning: After filtering padding, no true labels remain. Metrics will be 0.0.")
        return {
            'accuracy': 0.0,
            'f1_macro': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'f1_weighted': 0.0,
            'precision_weighted': 0.0,
            'recall_weighted': 0.0,
        }

    metrics = {}
    metrics['accuracy'] = accuracy_score(filtered_true_labels, filtered_predictions)
    # For F1, Precision, and Recall, specify 'average' to handle multiclass.
    # 'macro' calculates metrics for each label, and finds their unweighted mean.
    # 'weighted' calculates metrics for each label, and finds their average weighted by support (the number of true instances for each label).
    # 'zero_division=0' handles cases where a class might have no true instances or no predicted instances, preventing division by zero.
    metrics['f1_macro'] = f1_score(filtered_true_labels, filtered_predictions, average='macro', zero_division=0)
    metrics['precision_macro'] = precision_score(filtered_true_labels, filtered_predictions, average='macro',
                                                 zero_division=0)
    metrics['recall_macro'] = recall_score(filtered_true_labels, filtered_predictions, average='macro', zero_division=0)

    metrics['f1_weighted'] = f1_score(filtered_true_labels, filtered_predictions, average='weighted', zero_division=0)
    metrics['precision_weighted'] = precision_score(filtered_true_labels, filtered_predictions, average='weighted',
                                                    zero_division=0)
    metrics['recall_weighted'] = recall_score(filtered_true_labels, filtered_predictions, average='weighted',
                                              zero_division=0)

    return metrics


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
    test.add_argument('-lr', '--learning-rate', default=1e-3, type=float,
                      help='Learning rate')
    test.add_argument('-wd', '--weight-decay', default=1e-2, type=float,
                      help='Weight decay (L2 penalty)')

    run = parser.add_argument_group(title='Prediction parameters',
                                    description='Parameters for Prediction using model')
    # run.add_argument('-t', '--target-label-cols', help='Output pytorch model',
    #                  choices=['class', 'class.architecture', 'class.architecture.topology',
    #                           'class.architecture.topology.homology'], default='class.architecture.topology.homology')
    run.add_argument('-o', '--output', help='Output pytorch model', default='output')
    run.add_argument('--overwrite', help='Overwrite files in output path', action='store_true', default=False)
    run.add_argument('-i', '--input_folder', help='Input data folder', default="datasets/v1")

    args = parser.parse_args()

    if os.path.exists(args.output):
        if args.overwrite:
            print("Output folder already exists. Overwriting")
        else:
            print("Output folder already exists. Specify a new output folder or add --overwrite flag")
            return
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU) for training.")
    else:
        device = torch.device("cpu")
        print("MPS not available, falling back to CPU for training.")
    input_folder = args.input_folder
    train_path, val_path, test_path = get_train_val_test_paths(input_folder)
    print("--------------------------------------------------\nData loading")
    label_encoder = LabelEncoder()

    train_dataset = CathPredPerResidueDataset(train_path, label_encoder, fit=True)

    val_dataset = CathPredPerResidueDataset(val_path, label_encoder)

    num_classes = len(label_encoder.classes_)
    print("Number of classes: {}".format(num_classes))
    # initiate the model
    model = CathPredEnn2(num_classes=num_classes)
    model.to(device)
    max_protein_length = calculate_max_protein_length(input_folder)

    collate_fn = create_protein_collate_fn(max_protein_length, train_dataset.no_domain_encoded_id)
    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch, collate_fn=collate_fn)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    loss_fn = torch.nn.CrossEntropyLoss()

    print("--------------------------------------------------\nTraining")

    output_path = args.output
    os.makedirs(output_path, exist_ok=True)

    hyperparameters = vars(args)
    params_file_path = os.path.join(output_path, 'params.json')
    with open(params_file_path, 'w') as f:
        json.dump(hyperparameters, f, indent=4)
    tqdm.write(f"Hyperparameters saved to: {params_file_path}")

    label_encoder_file_path = os.path.join(output_path, 'label_encoder.pkl')
    with open(label_encoder_file_path, "wb") as f:
        pickle.dump(label_encoder, f)
    tqdm.write(f"Label encoder saved to: {label_encoder_file_path}")

    # Training loop. 1 epoch = 1 Loop over the dataset:
    metrics_columns = [
        "epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy",
        "train_f1_macro", "train_precision_macro", "train_recall_macro",
        "train_f1_weighted", "train_precision_weighted", "train_recall_weighted",
        "val_f1_macro", "val_precision_macro", "val_recall_macro",
        "val_f1_weighted", "val_precision_weighted", "val_recall_weighted"
    ]
    metrics_df = pd.DataFrame(columns=metrics_columns)

    best_loss = float('inf')
    best_epoch = -1

    for epoch in tqdm(range(args.epoch), desc="Training Progress"):
        train_loss, train_preds, train_labels = train(model, train_dataloader, optimizer, loss_fn, device)
        val_loss, val_preds, val_labels = evaluate(model, val_dataloader, loss_fn, device)
        tqdm.write(f"Training Loss: {train_loss}, Validation Loss: {val_loss}")

        # Calculate metrics for the current epoch
        train_metrics = calculate_metrics(train_preds, train_labels,
                                          no_domain_encoded_id=train_dataset.no_domain_encoded_id)
        val_metrics = calculate_metrics(val_preds, val_labels, no_domain_encoded_id=val_dataset.no_domain_encoded_id)

        # Create a dictionary for the current epoch's metrics
        current_epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics['accuracy'],
            "val_accuracy": val_metrics['accuracy'],
            "train_f1_macro": train_metrics['f1_macro'],
            "train_precision_macro": train_metrics['precision_macro'],
            "train_recall_macro": train_metrics['recall_macro'],
            "train_f1_weighted": train_metrics['f1_weighted'],
            "train_precision_weighted": train_metrics['precision_weighted'],
            "train_recall_weighted": train_metrics['recall_weighted'],
            "val_f1_macro": val_metrics['f1_macro'],
            "val_precision_macro": val_metrics['precision_macro'],
            "val_recall_macro": val_metrics['recall_macro'],
            "val_f1_weighted": val_metrics['f1_weighted'],
            "val_precision_weighted": val_metrics['precision_weighted'],
            "val_recall_weighted": val_metrics['recall_weighted'],
        }

        # Append the current epoch's metrics as a new row to the DataFrame
        metrics_df = pd.concat([metrics_df, pd.DataFrame([current_epoch_metrics])], ignore_index=True)

        # Save the DataFrame to CSV after each epoch
        metrics_df.to_csv(os.path.join(output_path, 'metrics_df.csv'), index=False)

        # Update tqdm description with current epoch metrics using tqdm.write
        tqdm.write(f"Epoch {epoch + 1}/{args.epoch}")
        tqdm.write(f"Train Loss: {train_loss:.4f}, Train Acc: {train_metrics['accuracy']:.4f}, "
                   f"Train F1 (Weighted): {train_metrics['f1_weighted']:.4f}")
        tqdm.write(f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
                   f"Val F1 (Weighted): {val_metrics['f1_weighted']:.4f}")

        # Save best performance
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(output_path, "best_model.pt"))
            tqdm.write(f"Saved best model at epoch {epoch + 1} with Val Loss: {val_loss:.4f}")

        # early stopping if loss does not improve for a patience of 10 epochs
        if (epoch - best_epoch) == 10:
            tqdm.write(f"Early stopping at epoch {epoch + 1}")
            break

    tqdm.write("\nTraining complete. All epoch data saved to metrics_df.csv")


if __name__ == '__main__':
    main()
