import argparse
import json
import os

import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm

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
        for k, v in x.items():
            x[k] = v.to(device, non_blocking=True)
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
            for k, v in x.items():
                x[k] = v.to(device, non_blocking=True)
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


def calculate_metrics(predictions, true_labels):
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
    metrics['f1_macro'] = f1_score(true_labels, predictions, average='macro')
    metrics['precision_macro'] = precision_score(true_labels, predictions, average='macro')
    metrics['recall_macro'] = recall_score(true_labels, predictions, average='macro')

    metrics['f1_weighted'] = f1_score(true_labels, predictions, average='weighted')
    metrics['precision_weighted'] = precision_score(true_labels, predictions, average='weighted')
    metrics['recall_weighted'] = recall_score(true_labels, predictions, average='weighted')

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
    train_path, val_path, test_path = os.path.join(input_folder, 'train_split.csv'), os.path.join(input_folder,
                                                                                                  'val_split.csv'), os.path.join(
        input_folder, 'test_split.csv')
    print("--------------------------------------------------\nData loading")
    label_encoder = LabelEncoder()
    train_dataset = CathPredDomainDataset(train_path, label_encoder, args.target_label_cols, fit=True)

    val_dataset = CathPredDomainDataset(val_path, label_encoder, args.target_label_cols)

    num_classes = len(label_encoder.classes_)
    # initiate the model
    model = CathPred(num_classes=num_classes)
    model.to(device)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=11)

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch, num_workers=11)

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
    os.makedirs(output_path, exist_ok=True)

    hyperparameters = vars(args)
    params_file_path = os.path.join(args.output, 'params.json')
    with open(params_file_path, 'w') as f:
        json.dump(hyperparameters, f, indent=4)
    tqdm.write(f"Hyperparameters saved to: {params_file_path}")

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

        scheduler.step()

        # Calculate metrics for the current epoch
        train_metrics = calculate_metrics(train_preds, train_labels)
        val_metrics = calculate_metrics(val_preds, val_labels)

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
            if args.save:
                torch.save(model.state_dict(), os.path.join(output_path, "best_model.pt"))
                tqdm.write(f"Saved best model at epoch {epoch + 1} with Val Loss: {val_loss:.4f}")

        # early stopping if loss does not improve for a patience of 10 epochs
        if (epoch - best_epoch) == 10:
            tqdm.write(f"Early stopping at epoch {epoch + 1}")
            break

    tqdm.write("\nTraining complete. All epoch data saved to metrics_df.csv")

    # --- After the loop: Save all collected data ---
    print("\nTraining complete. Saving all epoch data...")
    # # load best model
    # model.load_state_dict(torch.load(os.path.join(output_path, "best_model.pt")))
    # test_dataset = CathPredDomainDataset(test_path, label_encoder, args.target_label_cols, fit=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=True, num_workers=11)
    # # output all the correlations
    # out_tensors_test, target_tensors_test = make_prediction(model, test_dataloader, device)
    # testdf = create_prediction_df(test_dataset, out_tensors_test, target_tensors_test)
    # testdf.to_parquet(os.path.join(output_path, 'model_prediction.parquet'))
    # print(f"Test set predictions saved to: {os.path.join(output_path, 'model_prediction.parquet')}")
    #
    # train_unshuffled_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=False, pin_memory=True,
    #                                          num_workers=14)
    # out_tensors_train, target_tensors_train = make_prediction(model, train_unshuffled_dataloader, device)
    # traindf = create_prediction_df(train_dataset, out_tensors_train, target_tensors_train)
    # traindf.to_parquet(os.path.join(output_path, 'model_trainingset.parquet'))
    # print(f"Training set predictions saved to: {os.path.join(output_path, 'model_trainingset.parquet')}")
    #
    # print("--------------------------------------------------\nFinished!")


if __name__ == '__main__':
    main()
