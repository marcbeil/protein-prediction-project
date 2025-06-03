import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(
        description="Create train, validation, and test splits for protein domain data.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input parameters
    input_group = parser.add_argument_group(title='Input',
                                            description='Input data parameters')
    input_group.add_argument('-i', '--input-file', type=str, required=True,
                             help='Path to the input CSV file containing protein domain data (e.g., domain_id, class, architecture, homology).')

    # Output parameters
    output_group = parser.add_argument_group(title='Output',
                                             description='Output parameters')
    output_group.add_argument('-o', '--output-folder', type=str, required=True,
                              help='Path to the output folder where split DataFrames will be saved as .csv files.')

    # Splitting parameters
    split_group = parser.add_argument_group(title='Splitting',
                                            description='Parameters for data splitting')
    split_group.add_argument('--test-size', type=float, default=0.2,
                             help='Proportion of the dataset to include in the test split (default: 0.2).')
    split_group.add_argument('--val-size', type=float, default=0.25,
                             help='Proportion of the training data to include in the validation split. '
                                  'Set to 0.0 for no separate validation set (default: 0.0). '
                                  'Note: This is a proportion of the *remaining* data after test split.')
    split_group.add_argument('--random-state', type=int, default=42,
                             help='Random seed for reproducibility of splits (default: 42).')
    split_group.add_argument('--stratify-by', type=str, default='class.architecture.topology.homology',
                             choices=['class', 'class.architecture', 'class.architecture.topology',
                                      'class.architecture.topology.homology', 'none'],
                             help='Column(s) to use for stratification. '
                                  'Choose from "class", "class.architecture", "class.architecture.topology", "class.architecture.topology.homology", or "none" to disable stratification. '
                                  'Default: "class.architecture".')
    split_group.add_argument('--min-samples-per-class', type=int, default=5,
                             help='Minimum number of samples required for a class to be included in the dataset. '
                                  'Classes with fewer samples will be dropped. (default: 5).')

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = os.path.abspath(args.output_folder)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory set to: {output_dir}")

    # Load the dataset
    try:
        df = pd.read_csv(os.path.abspath(args.input_file))
        print(f"Successfully loaded data from: {args.input_file}")
        print(f"Total samples in dataset: {len(df)}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_file}")
        return
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    # Ensure required 'domain_id' column exists
    required_id_col = 'domain_id'
    if required_id_col not in df.columns:
        print(f"Error: Input CSV must contain '{required_id_col}' column.")
        return

    # Convert domain_start, domain_end, cath_domain_start, cath_domain_end, length to integers
    # This is done here to ensure these columns are integers before any filtering or splitting
    columns_to_int_initial = ['domain_start', 'domain_end', 'cath_domain_start', 'cath_domain_end', 'length']
    for col in columns_to_int_initial:
        if col in df.columns:
            # Convert to numeric first to handle potential strings like '1.0'
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Drop rows where conversion failed, or fill NaNs if appropriate for your data
            if df[col].isnull().any():
                print(
                    f"Warning: NaN values found in '{col}' column after numeric conversion. Dropping rows with NaNs in '{col}'.")
                df.dropna(subset=[col], inplace=True)
            df[col] = df[col].astype('Int64')  # Use nullable integer type
            print(f"Converted '{col}' to integer type.")
        else:
            print(f"Warning: Column '{col}' not found in the input CSV. Skipping type conversion for it.")

    # Handle stratification target dynamically based on --stratify-by argument
    stratify_target = None
    stratify_col_name = args.stratify_by
    initial_df_len = len(df)  # Store initial length for reporting dropped samples

    if stratify_col_name == 'none':
        print("Stratification is disabled as requested.")
    else:
        # Split the stratify_by string to get individual column names
        stratify_columns = stratify_col_name.split('.')

        # Check if all required columns for stratification exist in the DataFrame
        if not all(col in df.columns for col in stratify_columns):
            missing_cols = [col for col in stratify_columns if col not in df.columns]
            print(
                f"Error: Input CSV is missing required column(s) for '{stratify_col_name}' stratification: {missing_cols}")
            return

        # Create the combined stratification column
        # Convert all relevant columns to string and concatenate them
        df['__stratify_temp_col__'] = df[stratify_columns[0]].astype(str)
        for col in stratify_columns[1:]:
            df['__stratify_temp_col__'] = df['__stratify_temp_col__'] + '.' + df[col].astype(str)

        stratify_target = df['__stratify_temp_col__']
        print(f"Stratifying splits by '{stratify_col_name}'.")

        # --- Drop samples with rare classes ---
        class_counts = stratify_target.value_counts()
        rare_classes = class_counts[class_counts < args.min_samples_per_class].index

        if not rare_classes.empty:
            df = df[
                ~df['__stratify_temp_col__'].isin(rare_classes)].copy()  # Use .copy() to avoid SettingWithCopyWarning
            print(
                f"Dropped {initial_df_len - len(df)} samples belonging to classes with less than {args.min_samples_per_class} occurrences.")
            print(f"Remaining samples after dropping rare classes: {len(df)}")
            # Update stratify_target after dropping rows
            stratify_target = df['__stratify_temp_col__']
            class_counts = stratify_target.value_counts()  # Recalculate counts after dropping

        # Check if stratification is possible (at least 2 samples per class in the smallest split)
        # This check is now more robust after dropping rare classes
        min_samples_per_class_after_drop = class_counts.min()
        if min_samples_per_class_after_drop < 2:  # Changed from 3 to 2, as sklearn requires at least 2 for stratification
            print(
                f"Warning: Some '{stratify_col_name}' groups still have fewer than 2 samples ({min_samples_per_class_after_drop}). "
                f"Stratification might fail or result in uneven splits for these rare groups. "
                f"Consider using '--stratify-by none' or reviewing your data.")
        print(f"Class counts after filtering: {list(sorted(class_counts))}")  # Print counts after filtering

    # --- Step 1: Split into Training + Validation and Test sets ---
    if args.test_size > 0:
        # Ensure test_size is valid for stratification
        if stratify_target is not None:
            unique_classes = stratify_target.nunique()
            # If the number of unique classes is greater than 1, and the test set size
            # would be smaller than the number of unique classes, stratification might fail.
            if unique_classes > 1 and len(df) * args.test_size < unique_classes:
                print(f"Warning: Test size ({args.test_size}) might be too small for stratification, "
                      f"as it results in fewer samples ({len(df) * args.test_size:.0f}) than unique '{stratify_col_name}' groups ({unique_classes}) in the test set. "
                      f"Consider increasing test-size or using '--stratify-by none'.")

        train_val_df, test_df = train_test_split(
            df,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=stratify_target
        )
        print(f"Split into Training+Validation ({len(train_val_df)} samples) and Test ({len(test_df)} samples).")
    else:
        train_val_df = df.copy()
        test_df = pd.DataFrame()  # Empty DataFrame if no test split
        print("No test split requested (test-size is 0). All data will be used for training/validation.")

    # --- Step 2: Split Training + Validation into Training and Validation sets (if val_size > 0) ---
    if args.val_size > 0 and len(train_val_df) > 0:
        # Recalculate stratify target for train_val_df
        stratify_target_val = None
        if stratify_col_name != 'none':
            # Use the temporary column created earlier for stratification
            stratify_target_val = train_val_df['__stratify_temp_col__']

        # Check if stratification is possible for the validation split
        if stratify_target_val is not None:
            class_counts_val = stratify_target_val.value_counts()
            min_samples_per_class_val = class_counts_val.min()
            if min_samples_per_class_val < 2:
                print(
                    f"Warning: Some '{stratify_col_name}' groups in the training+validation set have fewer than 2 samples ({min_samples_per_class_val}). "
                    f"Stratification for validation split might fail or result in uneven splits for these rare groups.")

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=args.val_size,  # This is a proportion of train_val_df
            random_state=args.random_state,
            stratify=stratify_target_val
        )
        print(
            f"Further split Training+Validation into Training ({len(train_df)} samples) and Validation ({len(val_df)} samples).")
    else:
        train_df = train_val_df.copy()
        val_df = pd.DataFrame()  # Empty DataFrame if no validation split
        if args.val_size > 0:
            print("No validation split performed as training+validation set is empty or val-size is 0.")
        else:
            print("No validation split requested (val-size is 0).")

    # Drop the temporary stratification column before saving
    if '__stratify_temp_col__' in train_df.columns:
        train_df = train_df.drop(columns=['__stratify_temp_col__'])
    if '__stratify_temp_col__' in val_df.columns:
        val_df = val_df.drop(columns=['__stratify_temp_col__'])
    if '__stratify_temp_col__' in test_df.columns:
        test_df = test_df.drop(columns=['__stratify_temp_col__'])

    # --- Save full DataFrames to CSV files ---
    def save_split_df(df_split, filename):
        if not df_split.empty:
            filepath = os.path.join(output_dir, filename)
            # Convert specified columns to integer type before saving
            columns_to_int_save = ['domain_start', 'domain_end', 'cath_domain_start', 'cath_domain_end', 'length']
            for col in columns_to_int_save:
                if col in df_split.columns:
                    # Use .astype('Int64') for nullable integers to handle potential NaNs
                    df_split[col] = df_split[col].astype('Int64')

            df_split.to_csv(filepath, index=False)
            print(f"Saved {len(df_split)} samples to {filepath}")
        else:
            print(f"No samples to save for {filename} (split is empty).")

    save_split_df(train_df, 'train_split.csv')
    save_split_df(val_df, 'val_split.csv')
    save_split_df(test_df, 'test_split.csv')

    print("\nSplit statistics:")
    print(f"Total samples: {initial_df_len}")  # Report original total
    print(f"Samples after filtering rare classes: {len(df)}")  # Report samples after filtering
    print(f"Train samples: {len(train_df)}")
    if not val_df.empty:
        print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    if stratify_col_name != 'none':
        print(f"\nDistribution in splits by '{stratify_col_name}':")
        # For printing distribution, use the original DataFrame's temporary column
        # before it was dropped, or re-create for display if needed.
        # Here, we'll use the temporary column if it exists in the original df
        # for consistent reporting, as the split DFs might not have it anymore.
        if '__stratify_temp_col__' in df.columns:  # Check df, not original df, as it's filtered
            if not train_df.empty:
                print("--- Train Set Distribution ---")
                temp_train_df = pd.read_csv(os.path.join(output_dir, 'train_split.csv'),
                                            dtype={'domain_start': 'Int64', 'domain_end': 'Int64',
                                                   "cath_domain_start": "Int64", "cath_domain_end": "Int64",
                                                   "length": "Int64"})
                if stratify_col_name != 'none':
                    stratify_columns_for_report = stratify_col_name.split('.')
                    temp_train_df['__stratify_temp_col__'] = temp_train_df[stratify_columns_for_report[0]].astype(str)
                    for col in stratify_columns_for_report[1:]:
                        temp_train_df['__stratify_temp_col__'] = temp_train_df['__stratify_temp_col__'] + '.' + \
                                                                 temp_train_df[col].astype(str)
                    print(temp_train_df['__stratify_temp_col__'].value_counts(normalize=True).sort_index())
            if not val_df.empty:
                print("--- Validation Set Distribution ---")
                temp_val_df = pd.read_csv(os.path.join(output_dir, 'val_split.csv'),
                                          dtype={'domain_start': 'Int64', 'domain_end': 'Int64',
                                                 "cath_domain_start": "Int64", "cath_domain_end": "Int64",
                                                 "length": "Int64"})  # Added dtype for val_split.csv
                if stratify_col_name != 'none':
                    stratify_columns_for_report = stratify_col_name.split('.')
                    temp_val_df['__stratify_temp_col__'] = temp_val_df[stratify_columns_for_report[0]].astype(str)
                    for col in stratify_columns_for_report[1:]:
                        temp_val_df['__stratify_temp_col__'] = temp_val_df['__stratify_temp_col__'] + '.' + temp_val_df[
                            col].astype(str)
                    print(temp_val_df['__stratify_temp_col__'].value_counts(normalize=True).sort_index())
            if not test_df.empty:
                print("--- Test Set Distribution ---")
                temp_test_df = pd.read_csv(os.path.join(output_dir, 'test_split.csv'),
                                           dtype={'domain_start': 'Int64', 'domain_end': 'Int64',
                                                  "cath_domain_start": "Int64", "cath_domain_end": "Int64",
                                                  "length": "Int64"})  # Added dtype for test_split.csv
                if stratify_col_name != 'none':
                    stratify_columns_for_report = stratify_col_name.split('.')
                    temp_test_df['__stratify_temp_col__'] = temp_test_df[stratify_columns_for_report[0]].astype(str)
                    for col in stratify_columns_for_report[1:]:
                        temp_test_df['__stratify_temp_col__'] = temp_test_df['__stratify_temp_col__'] + '.' + \
                                                                temp_test_df[col].astype(str)
                    print(temp_test_df['__stratify_temp_col__'].value_counts(normalize=True).sort_index())
        else:
            # Fallback if the temporary column was not created (e.g., no stratification attempted)
            print("Could not display stratified distribution as the temporary column was not found.")


if __name__ == '__main__':
    main()