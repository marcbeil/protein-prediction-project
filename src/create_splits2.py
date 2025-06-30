import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
# You may need to install scikit-multilearn:
# pip install scikit-multilearn
from skmultilearn.model_selection import IterativeStratification


def stratify_protein_domain_dataset(
        df: pd.DataFrame,
        stratify_on: str = 'homology',
        n_splits: int = 3,
        split_ratios: tuple = (0.7, 0.15, 0.15),
        min_samples_per_class: int = 3
) -> list[pd.DataFrame]:
    """
    Performs a grouped, stratified split on a protein domain dataset.

    This function ensures that:
    1. All domains from the same protein_id are in the same split.
    2. The distribution of a specified CATH hierarchy level is preserved across splits.
    3. Each split is guaranteed to have at least one sample of each class,
       assuming the class exists in at least `n_splits` different proteins.
    4. Classes with fewer than `min_samples_per_class` proteins are excluded.

    Args:
        df (pd.DataFrame): The input dataframe containing domain information.
        stratify_on (str): The column name to stratify on.
        n_splits (int): The number of splits to generate.
        split_ratios (tuple): A tuple defining the ratio for each split.
        min_samples_per_class (int): Minimum number of proteins for a class to be included.

    Returns:
        list[pd.DataFrame]: A list of dataframes, one for each split.
    """
    # --- Input Validation ---
    if stratify_on not in ['class', 'architecture', 'topology', 'homology']:
        raise ValueError("`stratify_on` must be one of 'class', 'architecture', 'topology', 'homology'")
    if len(split_ratios) != n_splits or not np.isclose(sum(split_ratios), 1.0):
        raise ValueError(f"`split_ratios` must have {n_splits} elements and sum to 1.0")
    if 'domain_id' not in df.columns or stratify_on not in df.columns:
        raise ValueError(f"Dataframe must contain 'protein_id' and '{stratify_on}' columns.")
    if "protein_chain_id" not in df.columns:
        df["protein_chain_id"] = df["domain_id"].str[:5]
    print(f"--- Starting Stratified Split on '{stratify_on}' ---")
    print(f"Original dataset shape: {df.shape}")
    print(f"Original number of unique protein chains: {df['protein_chain_id'].nunique()}")
    if stratify_on != 'homology':
        raise NotImplementedError("Not implemented")
    stratify_on = "cath"
    # --- Step 0: Filter by Minimum Samples Per Class ---
    # Count unique proteins per class
    protein_class_counts = df.groupby(stratify_on)['protein_chain_id'].nunique()
    # Identify classes that meet the minimum threshold
    valid_classes = protein_class_counts[protein_class_counts >= min_samples_per_class].index

    original_class_count = df[stratify_on].nunique()
    if len(valid_classes) < original_class_count:
        print(f"\n--- Filtering classes with fewer than {min_samples_per_class} samples ---")
        print(f"Removed {original_class_count - len(valid_classes)} out of {original_class_count} classes.")

    # Filter the dataframe to only include proteins that belong to the valid classes.
    # A protein is included if at least one of its domains is in a valid class.
    df = df[df[stratify_on].isin(valid_classes)].copy()

    print(f"\nFiltered dataset shape: {df.shape}")
    print(f"Number of unique protein chains after filtering: {df['protein_chain_id'].nunique()}")
    print(f"Number of unique classes in '{stratify_on}' after filtering: {df[stratify_on].nunique()}")

    # --- Step 1: Create a Group-Level Representation ---
    protein_to_label = df.groupby('protein_chain_id')[stratify_on].apply(lambda x: x.mode()[0]).reset_index()
    protein_to_label.columns = ['protein_chain_id', 'label']

    # --- Step 2: One-Hot Encode the Labels for the Stratifier ---
    mlb = MultiLabelBinarizer(classes=sorted(valid_classes.tolist()))
    X = protein_to_label[['protein_chain_id']].values
    y = mlb.fit_transform(protein_to_label['label'].apply(lambda x: [x]))

    print(f"\nCreated protein-level dataset for stratification with shape: {X.shape}")
    print(f"Created one-hot label matrix for stratification with shape: {y.shape}")

    # --- Step 3: Perform Iterative Stratification ---
    stratifier = IterativeStratification(n_splits=n_splits, order=1, sample_distribution_per_fold=split_ratios)
    split_indices = list(stratifier.split(X, y))

    # --- Step 4: Map Protein Splits Back to the Original DataFrame ---
    final_splits = []
    split_names = ['Train', 'Validation', 'Test'] if n_splits == 3 else [f'Split {i + 1}' for i in range(n_splits)]

    for i, (_, split_idx) in enumerate(split_indices):
        split_protein_chain_ids = X[split_idx].flatten()
        split_df = df[df['protein_chain_id'].isin(split_protein_chain_ids)].copy()
        final_splits.append(split_df)

        print(f"\n--- {split_names[i]} Split ---")
        print(f"Shape: {split_df.shape}")
        print(f"Number of unique proteins: {split_df['protein_chain_id'].nunique()}")

    # --- Step 5: Verification ---
    print("\n--- Verification ---")
    all_labels = set(df[stratify_on])
    for i, split_df in enumerate(final_splits):
        labels_in_split = set(split_df[stratify_on])
        if labels_in_split == all_labels:
            print(f"✅ {split_names[i]} split contains all {len(all_labels)} classes.")
        else:
            missing = all_labels - labels_in_split
            print(f"⚠️ WARNING: {split_names[i]} split is missing {len(missing)} classes: {missing}")

    all_seen_proteins = set()
    for i, split_df in enumerate(final_splits):
        current_proteins = set(split_df['protein_chain_id'])
        if not all_seen_proteins.isdisjoint(current_proteins):
            print(f"❌ ERROR: Protein ID leakage detected in {split_names[i]} split!")
            break
        all_seen_proteins.update(current_proteins)
    else:
        print("✅ No protein ID leakage detected across splits.")

    return final_splits


def main():
    """Main function to run the script from the command line."""
    parser = argparse.ArgumentParser(
        description="Perform a grouped, stratified split on a CATH domain dataset."
    )
    parser.add_argument(
        "-i", "--input-csv",
        required=True,
        help="Path to the input CSV file."
    )
    parser.add_argument(
        "-o", "--output-dir",
        required=True,
        help="Directory to save the output CSV splits (train.csv, validation.csv, test.csv)."
    )
    parser.add_argument(
        "-s", "--stratify-by",
        default='homology',
        choices=['class', 'architecture', 'topology', 'homology'],
        help="CATH level to stratify on (default: topology)."
    )
    parser.add_argument(
        "--val-size",
        type=float, default=0.15,
        help="Proportion of the dataset for the validation set (default: 0.15)."
    )
    parser.add_argument(
        "--test-size",
        type=float, default=0.15,
        help="Proportion of the dataset for the test set (default: 0.15)."
    )
    parser.add_argument(
        "--min-samples-per-class",
        type=int, default=3,
        help="Minimum number of unique proteins required for a class to be included (default: 3)."
    )

    args = parser.parse_args()

    # --- Argument validation ---
    if args.val_size + args.test_size >= 1.0:
        raise ValueError("The sum of val_size and test_size must be less than 1.0.")

    # --- Execution ---
    print(f"Loading data from: {args.input_csv}")
    try:
        df = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_csv}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output will be saved to: {args.output_dir}")

    train_size = 1.0 - args.val_size - args.test_size
    split_ratios = (train_size, args.val_size, args.test_size)
    print(f"Splitting data into Train/Validation/Test with ratios: {split_ratios}")

    splits = stratify_protein_domain_dataset(
        df=df,
        stratify_on=args.stratify_by,
        n_splits=3,
        split_ratios=split_ratios,
        min_samples_per_class=args.min_samples_per_class
    )

    if len(splits) == 3:
        train_df, val_df, test_df = splits

        # Save files
        train_path = os.path.join(args.output_dir, 'train_split.csv')
        val_path = os.path.join(args.output_dir, 'val_split.csv')
        test_path = os.path.join(args.output_dir, 'test_split.csv')

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        params = vars(args)
        with open(os.path.join(args.output_dir, 'params.json'), 'w') as f:
            json.dump(params, f)

        print(f"\n✅ Successfully saved splits to:")
        print(f"  - Train: {train_path}")
        print(f"  - Validation: {val_path}")
        print(f"  - Test: {test_path}")


if __name__ == '__main__':
    main()
