import argparse
import json
import os
import pickle

import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter

import numpy as np


from dataset import CathPredDomainDataset
from model import CathPred


def train(model, train_dataloader, optimizer, loss_fn, device):
    # Set model to training mode
    model.train()
    # Initialize running loss value
    total_loss = 0.0

    # Lists to store all predictions and true labels from the epoch
    all_preds = []
    all_labels = []

    # Loop over the batches of data
    for x, y in train_dataloader:
        x = x["embedding"].to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass: get outputs by passing inputs to the model
        with torch.autocast(device_type=device.type):
            outputs = model(x)
            # Compute loss: compare outputs with labels
            loss = loss_fn(outputs, y)

        # Backward pass: compute gradients of the loss with respect to model parameters
        loss.backward()
        # Update parameters: perform a single optimization step (parameter update)
        optimizer.step()

        # Record statistics
        total_loss += loss.item()  # Accumulate loss for the epoch

        # Collect predictions and true labels
        # Assuming a classification task where model outputs are logits
        # You'll likely need to apply argmax to get the predicted class.
        # Adjust this line if your model output or task is different (e.g., regression).
        preds = torch.argmax(outputs, dim=1)

        # Move tensors to CPU and convert to numpy for compatibility with sklearn metrics
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    # Calculate average loss for the epoch
    # It's usually better to average by the number of batches, not dataset size,
    # if `loss_fn` already averages loss per item in the batch.
    # If your `loss_fn` returns sum of losses per batch, then dividing by dataset size is correct.
    # I'll stick to averaging by number of batches as it's common with mean-reduction losses.
    avg_loss = total_loss / len(train_dataloader)

    # Return the average loss, and the collected predictions and true labels
    return avg_loss, all_preds, all_labels


def evaluate(model, val_dataloader, loss_fn, device):
    # Set model to evaluation mode
    model.eval()
    # Initialize validation loss
    total_loss = 0.0

    # Lists to store all predictions and true labels from the epoch
    all_preds = []
    all_labels = []

    # Disable gradient computation for efficiency and to prevent accidental updates
    with torch.no_grad():
        for x, y in val_dataloader:
            x = x["embedding"].to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type):
                outputs = model(x)
                loss = loss_fn(outputs, y)

            # Record statistics
            total_loss += loss.item()

            # Collect predictions and true labels
            # Assuming classification: get the predicted class.
            # Adjust this line if your model output or task is different (e.g., regression).
            preds = torch.argmax(outputs, dim=1)

            # Move tensors to CPU and convert to numpy for compatibility with sklearn metrics
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Calculate average loss for the epoch
    # As discussed, averaging by the number of batches is common when loss_fn
    # already averages loss per item in the batch.
    avg_loss = total_loss / len(val_dataloader)

    # Return the average loss, and the collected predictions and true labels
    return avg_loss, all_preds, all_labels


def protein_collate_fn(batch):
    """
    Pads the batches for the Dataloader
    Args:
        batch: List of tuples, each containing:
            - dict: Dictionary with 'embedding' tensor (protein_length x 1024)
            - torch.Tensor: Encoded numerical label

    Returns:
        tuple: (batch_dict, labels)
            - batch_dict: dict with 'embedding' tensor of shape (batch_size, max_protein_length, 1024)
            - labels: torch.Tensor of shape (batch_size,)
    """
    embeddings = []
    labels = []

    for embedding_dict, label in batch:
        embeddings.append(embedding_dict["embedding"])
        labels.append(label)

    batch_embeddings = pad_sequence(embeddings, batch_first=True, padding_value=0.0)

    batch_labels = torch.stack(labels, dim=0)

    return {"embedding": batch_embeddings}, batch_labels


def calculate_metrics(predictions, true_labels, label_encoder):
    """
    Calculates various classification metrics.
    Args:
        predictions (torch.Tensor or np.array): Model's predicted classes (e.g., after argmax).
        true_labels (torch.Tensor or np.array): True class labels.
    Returns:
        dict: A dictionary of computed metrics.
    """
    # Ensure inputs are on CPU and converted to numpy for sklearn
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()

    metrics = {}
    metrics['accuracy'] = accuracy_score(true_labels, predictions)
    metrics['f1_macro'] = f1_score(true_labels, predictions, average='macro', zero_division=0)
    metrics['precision_macro'] = precision_score(true_labels, predictions, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(true_labels, predictions, average='macro', zero_division=0)

    metrics['f1_weighted'] = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    metrics['precision_weighted'] = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(true_labels, predictions, average='weighted', zero_division=0)

    if label_encoder:
        # per level prediction
        prediction_cath = label_encoder.inverse_transform(predictions)
        true_cath = label_encoder.inverse_transform(true_labels)

        max_levels = 4

        for level in range(1, max_levels + 1):
            # Get hierarchy up to current level
            pred_level = ['.'.join(label.split('.')[:level]) for label in prediction_cath]
            true_level = ['.'.join(label.split('.')[:level]) for label in true_cath]

            # Calculate accuracy
            acc = accuracy_score(true_level, pred_level)
            metrics[f'accuracy_level_{level}'] = acc

    return metrics

def showSamplePredictions(true_cath, prediction_cath):
    # Log 20 sample pairs with green highlighting
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS (Green = Correct)")
    print("=" * 60)

    sample_size = min(100, len(prediction_cath))
    for i in range(sample_size):
        pred = prediction_cath[i]
        true = true_cath[i]

        pred_parts = pred.split('.')
        true_parts = true.split('.')

        # Find matching levels
        match_level = 0
        for j in range(min(len(pred_parts), len(true_parts))):
            if pred_parts[j] == true_parts[j]:
                match_level += 1
            else:
                break

        # Create colored output
        pred_colored = ""
        true_colored = ""

        for j, part in enumerate(pred_parts):
            if j < match_level:
                pred_colored += f"\033[92m{part}\033[0m"  # Green
            else:
                pred_colored += f"\033[91m{part}\033[0m"  # Red
            if j < len(pred_parts) - 1:
                pred_colored += "."

        for j, part in enumerate(true_parts):
            if j < match_level:
                true_colored += f"\033[92m{part}\033[0m"  # Green
            else:
                true_colored += f"\033[91m{part}\033[0m"  # Red
            if j < len(true_parts) - 1:
                true_colored += "."

        print(f"{i + 1:2d}. Pred: {pred_colored}")
        print(f"    True: {true_colored}")
        print()

def bootstrapping(test_preds, test_labels, label_encoder):
    n_iterations = 1000
    n_size = len(test_labels)

    all_boot_metrics = []

    # Convert labels and preds to numpy arrays for indexing
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)

    for i in range(n_iterations):
        indices = np.random.choice(range(n_size), size=n_size, replace=True)

        sample_preds = test_preds[indices]
        sample_labels = test_labels[indices]

        # Calculate metrics for this bootstrap sample
        boot_metrics = calculate_metrics(sample_preds, sample_labels, label_encoder)

        all_boot_metrics.append(boot_metrics)

    # === Summarize Bootstrapped Metrics ===
    # Example: summarizing accuracy

    # accuracy, accuracy_level_1, ... 4
    accuracy_keys = ['accuracy', 'accuracy_level_1', 'accuracy_level_2', 'accuracy_level_3', 'accuracy_level_4']

    confidence_intervals = {}

    for acc_key in accuracy_keys:
        # Extract accuracy values for this level
        accuracies = [m[acc_key] for m in all_boot_metrics if acc_key in m]

        if accuracies:  # Only calculate if this accuracy level exists
            mean_acc = np.mean(accuracies)
            lower_acc = np.percentile(accuracies, 2.5)
            upper_acc = np.percentile(accuracies, 97.5)

            confidence_intervals[acc_key] = {
                'mean': mean_acc,
                'lower_ci': lower_acc,
                'upper_ci': upper_acc
            }

            # Print results
            level_name = acc_key.replace('_', ' ').title()
            print(f"Bootstrapped {level_name}: {mean_acc:.4f} (95% CI: {lower_acc:.4f} - {upper_acc:.4f})")

    return confidence_intervals

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



    run = parser.add_argument_group(title='Prediction parameters',
                                    description='Parameters for Prediction using model')
    run.add_argument('-t', '--target-label-cols', help='Output pytorch model',
                     choices=['class', 'class.architecture', 'class.architecture.topology',
                              'class.architecture.topology.homology'], default='class.architecture.topology.homology')
    run.add_argument('-o', '--output', help='Output pytorch model')
    run.add_argument('--overwrite', help='Overwrite files in output path', action='store_true', default=False)
    run.add_argument('--test_mode', help='Testing the best model on the test set', action='store_true', default=False)
    run.add_argument('-i', '--input_folder', help='Input data folder', default="datasets/v1")
    run.add_argument('--one_hot', default=False, type=bool)
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
    train_path, val_path, test_path = os.path.join(input_folder, 'train_split.csv'), os.path.join(input_folder,
                                                                                                  'val_split.csv'), os.path.join(
        input_folder, 'test_split.csv')
    print("--------------------------------------------------\nData loading")

    label_encoder = LabelEncoder()

    train_dataset = CathPredDomainDataset(train_path, label_encoder, args.target_label_cols, one_hot=args.one_hot, fit=True)

    val_dataset = CathPredDomainDataset(val_path, label_encoder, args.target_label_cols)

    num_classes = len(label_encoder.classes_)
    print("Number of classes: {}".format(num_classes))
    # initiate the model

    if args.one_hot:

        print("One-hot encoding")

        model = CathPred(num_classes=num_classes, in_channels= 21, out_channels=21)
        model.to(device)

    else:
        model = CathPred(num_classes=num_classes)
        model.to(device)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=1,
                                  collate_fn=protein_collate_fn)

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch, num_workers=1, collate_fn=protein_collate_fn)

    # Define optimizer with a scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning, weight_decay=args.weight_decay)
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, total_iters=args.warmup)
    train_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01,
                                                  total_iters=(args.epoch - args.warmup))
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, train_scheduler], [args.warmup])

    # Define loss criterion
    loss_fn = torch.nn.CrossEntropyLoss()

    output_path = args.output
    label_encoder_file_path = os.path.join(output_path, 'label_encoder.pkl')

    if args.test_mode:
        label_encoder = pickle.load(open(label_encoder_file_path, 'rb'))
        test_dataset = CathPredDomainDataset(test_path, label_encoder, args.target_label_cols)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch, num_workers=1, collate_fn=protein_collate_fn)
        model.load_state_dict(torch.load(os.path.join(output_path, "best_model.pt")))
        test_loss, test_preds, test_labels = evaluate(model, test_dataloader, loss_fn, device)
        test_metrics = calculate_metrics(test_preds, test_labels, label_encoder)
        print(test_metrics)
        bootstrapping(test_preds, test_labels, label_encoder)
        return

    print("--------------------------------------------------\nTraining")

    os.makedirs(output_path, exist_ok=True)
    print("output/tensorboard_logs/" + output_path.split("/")[-1])
    writer = SummaryWriter(log_dir="output/tensorboard_logs/" + output_path.split("/")[-1])

    with open(label_encoder_file_path, "wb") as f:
        pickle.dump(label_encoder, f)

    hyperparameters = vars(args)
    params_file_path = os.path.join(output_path, 'params.json')
    with open(params_file_path, 'w') as f:
        json.dump(hyperparameters, f, indent=4)
    tqdm.write(f"Hyperparameters saved to: {params_file_path}")

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

        if epoch >= 0:
            val_loss, val_preds, val_labels = evaluate(model, val_dataloader, loss_fn, device)
        else:
            val_loss, val_preds, val_labels = train_loss, train_preds, train_labels

        scheduler.step()

        # Calculate metrics for the current epoch
        train_metrics = calculate_metrics(train_preds, train_labels, label_encoder)
        val_metrics = calculate_metrics(val_preds, val_labels, label_encoder)

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

        writer.add_scalars('Loss', {'Train': train_loss, 'Validation': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'Train': train_metrics['accuracy'], 'Validation': val_metrics['accuracy']},
                           epoch)
        writer.add_scalars('F1_Weighted',
                           {'Train': train_metrics['f1_weighted'], 'Validation': val_metrics['f1_weighted']}, epoch)
        writer.add_scalars('F1_Macro', {'Train': train_metrics['f1_macro'], 'Validation': val_metrics['f1_macro']},
                           epoch)

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

    writer.close()

    # --- After the loop: Save all collected data ---
    print("\nTraining complete. Saving all epoch data...")


if __name__ == '__main__':
    main()
