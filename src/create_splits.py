import argparse
import json
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
    split_group.add_argument('--val-size', type=float, default=0.22,
                             help='Proportion of the training data to include in the validation split. '
                                  'Note: This is a proportion of the *remaining* data after test split. '
                                  'Set to 0.0 for no separate validation set (default: 0.0).')
    split_group.add_argument('--random-state', type=int, default=42,
                             help='Random seed for reproducibility of splits (default: 42).')
    split_group.add_argument('--stratify-by', type=str, default='class.architecture.topology.homology',
                             choices=['class', 'class.architecture', 'class.architecture.topology',
                                      'class.architecture.topology.homology', 'none'],
                             help='Column(s) to use for stratification. '
                                  'Choose from "class", "class.architecture", "class.architecture.topology", "class.architecture.topology.homology", or "none" to disable stratification. '
                                  'Default: "class.architecture.topology.homology".')
    split_group.add_argument('--min-samples-per-class', type=int, default=5,
                             help='Minimum number of samples (or proteins if --do-not-split-domains) required for a class to be included in the dataset. '
                                  'Classes with fewer samples/proteins will be dropped. (default: 5).')
    split_group.add_argument('--max-sequence-length', type=int, default=-1,
                             help='Maximum sequence length for each sequence to be included in the dataset. Defaults to inf (no limit).')
    # New argument added here
    split_group.add_argument('--do-not-split-domains', action='store_true',
                             help='If set, all domains of the same protein_id will be contained in the same split (train, val, or test).')

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = os.path.abspath(args.output_folder)
    os.makedirs(output_dir, exist_ok=False)
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

    # Ensure required 'domain_id' and 'protein_sequence' columns exist
    required_cols = ['domain_id', 'protein_sequence']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Error: Input CSV must contain all required columns: {missing}.")
        return

    # Convert domain_start, domain_end, cath_domain_start, cath_domain_end, length to integers
    columns_to_int_initial = ['domain_start', 'domain_end', 'cath_domain_start', 'cath_domain_end', 'protein_length']
    for col in columns_to_int_initial:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any():
                print(
                    f"Warning: NaN values found in '{col}' column after numeric conversion. Dropping rows with NaNs in '{col}'.")
                df.dropna(subset=[col], inplace=True)
            df[col] = df[col].astype('Int64')  # Use nullable integer type
            print(f"Converted '{col}' to integer type.")
        else:
            # For cath_domain_start/end, it's fine if they are missing if CATH domains are not present.
            # But domain_start/end and length are usually crucial.
            if col in ['domain_start', 'domain_end', 'length']:
                print(f"Error: Crucial column '{col}' not found in the input CSV. Please ensure it exists.")
                return
            print(f"Warning: Optional column '{col}' not found in the input CSV. Skipping type conversion for it.")

    target_label_col_list = ["class", "architecture", "topology", "homology"]
    df['cath'] = df[target_label_col_list[0]].astype(str)
    for col in target_label_col_list[1:]:
        df['cath'] = df['cath'] + '.' + df[col].astype(str)

    df["protein_length"] = df["protein_sequence"].str.len()

    if args.max_sequence_length > 0:
        print(f"Max sequence length is set to {args.max_sequence_length}.")
        df_size_before_len_filter = len(df)
        df = df[df['protein_length'] <= args.max_sequence_length].copy()  # Use <= not <
        print(
            f"Size before length filter: {df_size_before_len_filter} -> Size after dropping rows where protein length > {args.max_sequence_length}: {len(df)}.")

    initial_df_len = len(df)  # Store length after initial filtering

    # --- Prepare stratification target based on --stratify-by and --do-not-split-domains ---
    stratify_target_col = None  # This will be the temporary column name for stratification

    if args.stratify_by != 'none':
        # Split the stratify_by string to get individual column names
        stratify_columns = args.stratify_by.split('.')

        # Check if all required columns for stratification exist in the DataFrame
        if not all(col in df.columns for col in stratify_columns):
            missing_cols = [col for col in stratify_columns if col not in df.columns]
            print(
                f"Error: Input CSV is missing required column(s) for '{args.stratify_by}' stratification: {missing_cols}")
            return

        # Create the combined stratification column
        df['__stratify_temp_col__'] = df[stratify_columns[0]].astype(str)
        for col in stratify_columns[1:]:
            df['__stratify_temp_col__'] = df['__stratify_temp_col__'] + '.' + df[col].astype(str)
        stratify_target_col = '__stratify_temp_col__'
        print(f"Prepared stratification column '{stratify_target_col}' based on '{args.stratify_by}'.")

    # --- Determine the items to split (domains or proteins) and their stratification labels ---
    if args.do_not_split_domains:
        print("Splitting by unique protein IDs to ensure domains from the same protein stay together.")
        # Group by protein_id and get the most frequent stratification label for each protein
        protein_groups = df.groupby('protein_id')

        # For stratification, find the most common stratification class for each protein.
        # If a protein has no domains (e.g., after filtering), it won't be in unique_protein_ids anyway.
        if stratify_target_col:
            # Get the value counts of the stratification column within each protein, then pick the top one
            protein_strat_mapping = protein_groups[stratify_target_col].apply(
                lambda x: x.mode()[0] if not x.mode().empty else None).dropna()

            # Filter proteins with rare stratification classes
            protein_class_counts = protein_strat_mapping.value_counts()
            rare_protein_classes = protein_class_counts[protein_class_counts < args.min_samples_per_class].index

            if not rare_protein_classes.empty:
                proteins_to_drop_for_strat = protein_strat_mapping[
                    protein_strat_mapping.isin(rare_protein_classes)].index
                df = df[~df['protein_id'].isin(proteins_to_drop_for_strat)].copy()
                protein_groups = df.groupby('protein_id')  # Re-group after dropping
                protein_strat_mapping = protein_groups[stratify_target_col].apply(
                    lambda x: x.mode()[0] if not x.mode().empty else None).dropna()  # Recompute
                print(
                    f"Dropped {len(proteins_to_drop_for_strat)} proteins with less than {args.min_samples_per_class} occurrences of their most frequent '{args.stratify_by}' class.")
                print(f"Remaining proteins after filtering rare classes: {len(df['protein_id'].unique())}")

            items_to_split = protein_strat_mapping.index.tolist()
            stratify_for_split = protein_strat_mapping.loc[items_to_split]  # Ensure Series is aligned
            print(f"Protein-level class counts for stratification: {list(sorted(stratify_for_split.value_counts()))}")

            # Final check for stratification validity after rare class filtering
            if stratify_for_split.nunique() > 1 and stratify_for_split.value_counts().min() < 2:
                print(
                    f"Warning: Some protein '{args.stratify_by}' groups still have fewer than 2 proteins. "
                    f"Protein-level stratification might result in uneven splits for these rare groups. "
                    f"Consider using '--stratify-by none' or reviewing your data.")
        else:
            items_to_split = df['protein_id'].unique().tolist()
            stratify_for_split = None  # No stratification for proteins if stratify-by is 'none'

        # If after filtering, no items left to split
        if not items_to_split:
            print("No proteins left to split after applying filters (e.g., min-samples-per-class or length). Exiting.")
            return

        # Split the protein IDs
        train_val_items, test_items = train_test_split(
            items_to_split,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=stratify_for_split
        )
        print(f"Split proteins into Training+Validation ({len(train_val_items)}) and Test ({len(test_items)}).")

        # Map protein IDs back to domains
        train_val_df = df[df['protein_id'].isin(train_val_items)].copy()
        test_df = df[df['protein_id'].isin(test_items)].copy()

        # Step 2: Split Training + Validation proteins into Training and Validation sets
        if args.val_size > 0 and len(train_val_items) > 0:
            val_stratify_for_split = None
            if stratify_target_col:
                val_stratify_for_split = protein_strat_mapping.loc[train_val_items]

            train_items, val_items = train_test_split(
                train_val_items,
                test_size=args.val_size,
                random_state=args.random_state,
                stratify=val_stratify_for_split
            )
            print(f"Further split protein IDs into Training ({len(train_items)}) and Validation ({len(val_items)}).")

            train_df = df[df['protein_id'].isin(train_items)].copy()
            val_df = df[df['protein_id'].isin(val_items)].copy()
        else:
            train_df = train_val_df.copy()
            val_df = pd.DataFrame(columns=df.columns)  # Ensure columns match for empty df
            if args.val_size > 0:
                print("No validation split performed as training+validation protein set is empty or val-size is 0.")
            else:
                print("No validation split requested for proteins (val-size is 0).")

    else:  # Original splitting logic: split by domain_id
        print("Splitting individual domains (not enforcing domains from same protein in same split).")

        # Apply min-samples-per-class filtering directly to domains
        if stratify_target_col:
            domain_class_counts = df[stratify_target_col].value_counts()
            rare_domain_classes = domain_class_counts[domain_class_counts < args.min_samples_per_class].index

            if not rare_domain_classes.empty:
                df = df[~df[stratify_target_col].isin(rare_domain_classes)].copy()
                print(
                    f"Dropped {initial_df_len - len(df)} domains belonging to classes with less than {args.min_samples_per_class} occurrences.")
                print(f"Remaining domains after dropping rare classes: {len(df)}")

            # Re-assign stratify target for the filtered df
            stratify_for_split = df[stratify_target_col]
            if stratify_for_split.nunique() > 1 and stratify_for_split.value_counts().min() < 2:
                print(
                    f"Warning: Some domain '{args.stratify_by}' groups still have fewer than 2 samples ({stratify_for_split.value_counts().min()}). "
                    f"Domain-level stratification might fail or result in uneven splits for these rare groups. "
                    f"Consider using '--stratify-by none' or reviewing your data.")
            print(f"Domain-level class counts for stratification: {list(sorted(stratify_for_split.value_counts()))}")

        else:
            stratify_for_split = None  # No stratification for domains if stratify-by is 'none'

        # If after filtering, no items left to split
        if not df.empty:
            # --- Step 1: Split into Training + Validation and Test sets ---
            if args.test_size > 0:
                train_val_df, test_df = train_test_split(
                    df,
                    test_size=args.test_size,
                    random_state=args.random_state,
                    stratify=stratify_for_split
                )
                print(
                    f"Split domains into Training+Validation ({len(train_val_df)} samples) and Test ({len(test_df)} samples).")
            else:
                train_val_df = df.copy()
                test_df = pd.DataFrame(columns=df.columns)
                print("No test split requested (test-size is 0). All data will be used for training/validation.")

            # --- Step 2: Split Training + Validation into Training and Validation sets (if val_size > 0) ---
            if args.val_size > 0 and len(train_val_df) > 0:
                val_stratify_for_split = None
                if stratify_target_col:
                    val_stratify_for_split = train_val_df[stratify_target_col]

                train_df, val_df = train_test_split(
                    train_val_df,
                    test_size=args.val_size,
                    random_state=args.random_state,
                    stratify=val_stratify_for_split
                )
                print(
                    f"Further split domains into Training ({len(train_df)} samples) and Validation ({len(val_df)} samples).")
            else:
                train_df = train_val_df.copy()
                val_df = pd.DataFrame(columns=df.columns)
                if args.val_size > 0:
                    print("No validation split performed as training+validation set is empty or val-size is 0.")
                else:
                    print("No validation split requested (val-size is 0).")
        else:  # If df is empty after filtering
            train_df = pd.DataFrame(columns=df.columns)
            val_df = pd.DataFrame(columns=df.columns)
            test_df = pd.DataFrame(columns=df.columns)
            print("No domains left to split after applying filters (e.g., min-samples-per-class or length). Exiting.")
            return

    # Drop the temporary stratification column before saving
    for temp_df in [train_df, val_df, test_df]:
        if stratify_target_col and stratify_target_col in temp_df.columns:
            temp_df.drop(columns=[stratify_target_col], inplace=True)

    # --- Save full DataFrames to CSV files ---
    def save_split_df(df_split, filename):
        desired_column_order = [
            'protein_id', 'domain_id', 'domain_start', 'domain_end', 'protein_length',
            'cath', 'class', 'architecture', 'topology', 'homology', 'protein_sequence'
        ]
        if not df_split.empty:
            filepath = os.path.join(output_dir, filename)
            # Convert specified columns to integer type before saving
            columns_to_int_save = ['domain_start', 'domain_end', 'cath_domain_start', 'cath_domain_end', 'length']
            for col in columns_to_int_save:
                if col in df_split.columns:
                    df_split[col] = df_split[col].astype('Int64')  # Use nullable integers

            # Filter and reorder columns
            cols_to_keep = [col for col in desired_column_order if col in df_split.columns]
            df_split = df_split[cols_to_keep]

            df_split.to_csv(filepath, index=False)
            print(f"Saved {len(df_split)} samples to {filepath}")
        else:
            print(f"No samples to save for {filename} (split is empty).")

    save_split_df(train_df, 'train_split.csv')
    save_split_df(val_df, 'val_split.csv')
    save_split_df(test_df, 'test_split.csv')

    params = vars(args)
    with open(os.path.join(output_dir, 'params.json'), 'w') as f:
        json.dump(params, f)

    print("\nSplit statistics:")
    print(f"Total initial samples in input: {initial_df_len}")
    print(f"Total samples used for splitting (after initial filters): {len(df)}")
    print(f"Train samples: {len(train_df)}")
    if not val_df.empty:
        print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    # For reporting distribution, we need to reload the saved CSVs to get the __stratify_temp_col__ for protein_id/domain_id
    # Or, preferably, pass the stratified items directly, but reloading is safer given the drop operations.
    if args.stratify_by != 'none':
        print(f"\nDistribution in splits by '{args.stratify_by}':")
        stratify_columns_for_report = args.stratify_by.split('.')

        if not train_df.empty:
            print("--- Train Set Distribution ---")
            temp_train_df = pd.read_csv(os.path.join(output_dir, 'train_split.csv'),
                                        dtype={'domain_start': 'Int64', 'domain_end': 'Int64',
                                               "cath_domain_start": "Int64", "cath_domain_end": "Int64",
                                               "length": "Int64"})
            temp_train_df['__temp_report_col__'] = temp_train_df[stratify_columns_for_report[0]].astype(str)
            for col in stratify_columns_for_report[1:]:
                temp_train_df['__temp_report_col__'] = temp_train_df['__temp_report_col__'] + '.' + temp_train_df[
                    col].astype(str)
            print(temp_train_df['__temp_report_col__'].value_counts(normalize=True).sort_index())

        if not val_df.empty:
            print("--- Validation Set Distribution ---")
            temp_val_df = pd.read_csv(os.path.join(output_dir, 'val_split.csv'),
                                      dtype={'domain_start': 'Int64', 'domain_end': 'Int64',
                                             "cath_domain_start": "Int64", "cath_domain_end": "Int64",
                                             "length": "Int64"})
            temp_val_df['__temp_report_col__'] = temp_val_df[stratify_columns_for_report[0]].astype(str)
            for col in stratify_columns_for_report[1:]:
                temp_val_df['__temp_report_col__'] = temp_val_df['__temp_report_col__'] + '.' + temp_val_df[col].astype(
                    str)
            print(temp_val_df['__temp_report_col__'].value_counts(normalize=True).sort_index())

        if not test_df.empty:
            print("--- Test Set Distribution ---")
            temp_test_df = pd.read_csv(os.path.join(output_dir, 'test_split.csv'),
                                       dtype={'domain_start': 'Int64', 'domain_end': 'Int64',
                                              "cath_domain_start": "Int64", "cath_domain_end": "Int64",
                                              "length": "Int64"})
            temp_test_df['__temp_report_col__'] = temp_test_df[stratify_columns_for_report[0]].astype(str)
            for col in stratify_columns_for_report[1:]:
                temp_test_df['__temp_report_col__'] = temp_test_df['__temp_report_col__'] + '.' + temp_test_df[
                    col].astype(str)
            print(temp_test_df['__temp_report_col__'].value_counts(normalize=True).sort_index())


if __name__ == '__main__':
    main()
