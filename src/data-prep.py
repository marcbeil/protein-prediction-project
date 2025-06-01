import re
import pandas as pd

def parse_cath(cath_domain_list_path="../data/cath-domain-list.txt", cath_domain_seqs_path="../data/cath-domain-seqs.fa.txt"):
    columns = [
        "domain_id", "class", "architecture", "topology", "homology",
        "s35", "s60", "s95", "s100", "s100_count", "length", "resolution"
    ]

    with open(cath_domain_list_path, 'r') as f:
        lines = [line.strip() for line in f if not line.startswith('#') and line.strip()]

    data = [line.split() for line in lines]

    cath_domains_df = pd.DataFrame(data, columns=columns)

    print(f"Parsed cath classes for {len(cath_domains_df)} domains from  {cath_domain_list_path}")

    domain_ranges_df = parse_fasta_domain_ranges(cath_domain_seqs_path)

    print(f"Parsed domain ranges for {len(domain_ranges_df)} domains from {cath_domain_seqs_path}")

    merged = pd.merge(cath_domains_df, domain_ranges_df, on='domain_id', how='inner')

    print(f"Merging resulted in {len(merged)} domains")

    # Calculate length from domain_start and domain_end instead of using original value
    merged["length"] = merged["cath_domain_end"] - merged["cath_domain_start"] + 1

    for col in columns[1:]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors='coerce')

    return merged

def clean_coord_part(part):
    # Remove letters and brackets, keep digits and minus
    return re.sub(r"[^\d\-]", "", part)

def parse_fasta_domain_ranges(fasta_file):
    records = []
    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith(">"):
                line = line.strip()
                domain_part = line.split("|")[-1]  # e.g. 101mA00/1(A)-80 or 101mA00/-5-153
                domain_id = domain_part.split("/")[0]

                # Default values
                start = end = None

                if "/" in domain_part:
                    range_part = domain_part.split("/")[1]
                    if "-" in range_part:
                        start_raw, end_raw = range_part.split("-", 1)
                        try:
                            start = int(clean_coord_part(start_raw))
                            end = int(clean_coord_part(end_raw))
                        except ValueError:
                            pass # leave as None if not integers

                records.append({
                    "domain_id": domain_id,
                    "cath_domain_start": start,
                    "cath_domain_end": end
                })

    df = pd.DataFrame(records)

    # Convert columns to integer type, keeping NaN for missing values
    df['cath_domain_start'] = df['cath_domain_start'].astype('Int64')  # nullable integer
    df['cath_domain_end'] = df['cath_domain_end'].astype('Int64')  # nullable integer

    return df

def generate_subset(cath_domains_df, homology_groups=100, samples_per_group=10,
                    min_domains=10, max_domains=200, out_path="../data/subset.csv"):
    hierarchy = ['class', 'architecture', 'topology', 'homology']

    # Remove duplicate s35 sequences within each homology group
    df_unique_s35 = cath_domains_df.drop_duplicates(subset=hierarchy + ['s35'])
    reduction_percent = ((len(df) - len(df_unique_s35)) / len(df)) * 100
    print(
        f"Removing s35 duplicates: {len(df):,} → {len(df_unique_s35):,} domains (reduced by {reduction_percent:.1f}%)")

    # Filter groups by size constraints
    group_sizes = df_unique_s35.groupby(hierarchy).size()
    valid_groups = group_sizes[(group_sizes >= min_domains) & (group_sizes <= max_domains)]
    print(f"Found {len(valid_groups)} homology groups with {min_domains}-{max_domains} domains")

    # Sample homology groups
    sampled_groups = valid_groups.sample(
        n=min(homology_groups, len(valid_groups)), random_state=42
    ).index.to_frame().reset_index(drop=True)

    # Get data for sampled groups
    sampled_df = pd.merge(sampled_groups, df_unique_s35, on=hierarchy)

    # Sample domains within each group
    subset_list = []
    for name, group in sampled_df.groupby(hierarchy):
        sampled_group = group.sample(n=min(samples_per_group, len(group)), random_state=42)
        subset_list.append(sampled_group)

    subset = pd.concat(subset_list, ignore_index=True)

    # Select only required columns in specified order
    output_df = subset[['domain_id', 'class', 'architecture', 'topology', 'homology', 'length', 'cath_domain_start', 'cath_domain_end']]

    print(
        f"Generated random subset with {homology_groups} homology groups and {samples_per_group} samples per group. "
        f"Total: {len(output_df)} samples. Saved at: {out_path}")
    output_df.to_csv(out_path, index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate CATH protein domain subset')
    parser.add_argument('--input-cath',  default="../data/cath-domain-list.txt",
                        help='Input CATH domain list file (default: ../data/cath-domain-list.txt)')
    parser.add_argument('--input-seq', default="../data/cath-domain-seqs.fa",
                        help='Input domain sequences list file (default: ../data/cath-domain-seqs.fa')
    parser.add_argument('--output', '-o', default="../data/subset.csv",
                        help='Output subset CSV file (default: ../data/subset.csv)')
    parser.add_argument('--homology-groups', '-g', type=int, default=1000,
                        help='Number of homology groups to sample (default: 1000)')
    parser.add_argument('--samples-per-group', '-s', type=int, default=20,
                        help='Number of samples per homology group (default: 20)')
    parser.add_argument('--min-domains', type=int, default=5,
                        help='Minimum domains per homology group (default: 5)')
    parser.add_argument('--max-domains', type=int, default=1000,
                        help='Maximum domains per homology group (default: 1000)')

    args = parser.parse_args()

    print("---------------- STARTING DATA PREP  ------------------")
    print(f"Parameters: {args.homology_groups} groups, {args.samples_per_group} samples/group, "
          f"domain range: {args.min_domains}-{args.max_domains}")

    df = parse_cath(args.input_cath, args.input_seq)
    generate_subset(df,
                    homology_groups=args.homology_groups,
                    samples_per_group=args.samples_per_group,
                    min_domains=args.min_domains,
                    max_domains=args.max_domains,
                    out_path=args.output)
    print("---------------- FINISHED DATA PREP  ------------------")
