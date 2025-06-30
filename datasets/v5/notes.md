(protein-prediction) marcbeil@Marcs-MacBook-Pro protein-prediction-project % python3 src/create_splits2.py -i
data/subset_protein_mapped_enhanced_limited_len_600.csv -o datasets/v5
Loading data from: data/subset_protein_mapped_enhanced_limited_len_600.csv
Output will be saved to: datasets/v5
Splitting data into Train/Validation/Test with ratios: (0.7, 0.15, 0.15)
--- Starting Stratified Split on 'homology' ---
Original dataset shape: (9210, 12)
Original number of unique protein chains: 8679

--- Filtering classes with fewer than 3 samples ---
Removed 63 out of 940 classes.

Filtered dataset shape: (9108, 12)
Number of unique protein chains after filtering: 8595
Number of unique classes in 'cath' after filtering: 877

Created protein-level dataset for stratification with shape: (8595, 1)
Created one-hot label matrix for stratification with shape: (8595, 877)

--- Train Split ---
Shape: (6298, 12)
Number of unique proteins: 5959

--- Validation Split ---
Shape: (1404, 12)
Number of unique proteins: 1324

--- Test Split ---
Shape: (1406, 12)
Number of unique proteins: 1312

--- Verification ---
✅ Train split contains all 877 classes.
⚠️ WARNING: Validation split is missing 53 classes: {'1.10.1140.10', '3.40.50.11260', '1.10.287.80', '2.100.10.10', '
2.30.110.20', '2.30.180.10', '2.40.300.10', '1.10.287.1490', '2.60.60.20', '2.20.25.30', '3.40.50.1000', '
3.10.320.10', '1.10.506.10', '1.20.5.1160', '3.40.50.10890', '3.30.10.20', '3.90.1640.10', '3.90.120.10', '
3.90.380.10', '3.40.50.2030', '1.10.287.770', '3.40.50.1260', '1.10.135.10', '3.30.1640.10', '2.10.10.10', '
3.30.160.110', '3.40.140.20', '1.10.132.70', '1.10.132.20', '3.40.50.10170', '1.10.3470.10', '3.30.1480.10', '
3.40.50.10130', '1.10.150.120', '3.20.140.10', '1.20.120.80', '1.20.870.10', '1.20.58.100', '3.20.180.10', '
1.20.142.10', '3.30.30.30', '2.60.40.2580', '3.30.2130.30', '3.40.1030.10', '1.50.10.160', '1.20.140.20', '
1.10.220.20', '3.30.2320.30', '3.30.70.980', '1.10.10.250', '1.20.120.790', '3.30.1460.20', '2.60.40.180'}
⚠️ WARNING: Test split is missing 47 classes: {'1.20.58.480', '3.40.50.11260', '2.60.40.2060', '2.60.40.1320', '
3.40.30.60', '2.60.40.4100', '3.40.462.10', '1.20.990.10', '3.30.1340.10', '2.160.20.20', '3.40.50.10090', '
2.140.10.10', '1.20.91.20', '3.90.380.10', '2.30.250.10', '3.30.1640.10', '2.60.40.290', '3.30.1360.30', '
1.10.1390.10', '1.10.437.20', '3.40.367.20', '3.40.30.120', '1.10.1670.10', '3.40.1570.10', '3.40.50.10740', '
1.20.141.10', '1.20.870.10', '3.30.1550.10', '1.20.5.640', '3.90.930.1', '1.20.5.730', '3.40.50.10420', '2.60.40.690', '
3.90.1170.30', '3.90.920.10', '3.90.1170.10', '2.60.40.2020', '2.30.30.380', '2.60.220.30', '3.50.30.60', '
3.30.70.790', '3.40.50.10610', '3.40.50.10050', '1.10.220.30', '3.30.70.980', '3.30.460.20', '3.40.50.1450'}
✅ No protein ID leakage detected across splits.

✅ Successfully saved splits to:

- Train: datasets/v5/train_split.csv
- Validation: datasets/v5/val_split.csv
- Test: datasets/v5/test_split.csv



