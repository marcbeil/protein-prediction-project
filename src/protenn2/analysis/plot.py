import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_metric_with_ci(data, metric, plot_types=('raw', 'post', "baseline")):
    """
    Creates a grouped bar plot with confidence intervals for a specified metric.

    Args:
        data (dict): The input dictionary containing the metric data.
        metric (str): The name of the metric to plot (e.g., 'accuracy', 'f1_score_post_').
        plot_types (list): A list of types to include in the plot (e.g., ['raw', 'post']).
    """

    plot_order_prefix = ['C', 'A', 'T', 'H']
    plot_data = []

    for prefix in plot_order_prefix:
        for p_type in plot_types:
            key = f"{p_type}_{prefix}"
            if key in data and metric in data[key]:
                mean_val = data[key][metric]['mean']
                ci_lower = data[key][metric]['ci_lower']
                ci_upper = data[key][metric]['ci_upper']
                plot_data.append({
                    'Group': prefix,
                    'Type': p_type,
                    'Mean': mean_val,
                    'Error_Lower': mean_val - ci_lower,
                    'Error_Upper': ci_upper - mean_val
                })

    if not plot_data:
        print(f"No data found for metric '{metric}' with types {plot_types}.")
        return

    df = pd.DataFrame(plot_data)

    # Set a dark style for better visualization
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 7))

    group_labels = plot_order_prefix
    bar_width = 0.25
    x = np.arange(len(group_labels))

    raw_means, raw_errs = [], []
    post_means, post_errs = [], []
    baseline_means, baseline_errs = [], []

    for group in group_labels:
        for p_type in plot_types:
            row = df[(df['Group'] == group) & (df['Type'] == p_type)]
            if not row.empty:
                mean = row['Mean'].values[0]
                lower = row['Error_Lower'].values[0]
                upper = row['Error_Upper'].values[0]
                if p_type == 'raw':
                    raw_means.append(mean)
                    raw_errs.append([lower, upper])
                elif p_type == 'post':
                    post_means.append(mean)
                    post_errs.append([lower, upper])
                elif p_type == 'baseline' and p_type in plot_types:
                    baseline_means.append(mean)
                    baseline_errs.append([lower, upper])

    # Convert errors for matplotlib
    raw_errs = np.array(raw_errs).T if raw_errs else [[], []]
    post_errs = np.array(post_errs).T if post_errs else [[], []]
    baseline_errs = np.array(baseline_errs).T if baseline_errs else [[], []]

    # Plot grouped bars
    bars1 = ax.bar(x - bar_width / 2, raw_means, width=bar_width, label='Raw',
                   yerr=raw_errs, capsize=5, color='#6FB2E4', alpha=0.9,
                   error_kw={'ecolor': 'black', 'linewidth': 1.5})

    bars2 = ax.bar(x + bar_width / 2, post_means, width=bar_width, label='Post',
                   yerr=post_errs, capsize=5, color='darkblue', alpha=0.9,
                   error_kw={'ecolor': 'black', 'linewidth': 1.5})

    if "baseline" in plot_types:
        bars3 = ax.bar(x + bar_width * 1.5, baseline_means, width=bar_width, label='Baseline',
                       yerr=baseline_errs, capsize=5, color='#A0A0A0', alpha=0.9,
                       error_kw={'ecolor': 'black', 'linewidth': 1.5})

    # Labeling and formatting
    ax.set_xlabel('CATH Category', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'{metric.replace("_", " ").title()} (95% CI)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.tick_params(axis='y')
    ax.set_facecolor('white')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4, color='lightgray')
    ax.set_axisbelow(True)

    # Add value labels
    bars = list(bars1) + list(bars2)
    if "baseline" in plot_types:
        bars += list(bars3)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.3f}',
                ha='center', va='bottom', fontsize=9, color='black')

    ax.legend(facecolor='white', edgecolor='lightgray', labelcolor='black')
    plt.tight_layout()
    return plt
