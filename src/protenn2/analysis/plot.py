import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_metric_with_ci(data, metric, plot_types=('raw', 'post', 'two_tier', 'baseline')):
    """
    Creates a grouped bar plot with confidence intervals for a specified metric.

    Args:
        data (dict): The input dictionary containing the metric data.
        metric (str): The name of the metric to plot (e.g., 'accuracy', 'f1_score_post_').
        plot_types (tuple): A tuple of types to include in the plot, in the order they should appear.
    """

    plot_order_prefix = ['C', 'A', 'T', 'H']
    plot_data = []

    # Gather means and CIs for each combination of prefix and type
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

    plt.style.use('default')
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(8, 5))

    group_labels = plot_order_prefix
    bar_width = 0.2
    x = np.arange(len(group_labels))

    # Prepare dynamic containers
    means = {ptype: [] for ptype in plot_types}
    errs  = {ptype: [] for ptype in plot_types}

    # Populate the lists from the DataFrame
    for group in group_labels:
        for p_type in plot_types:
            row = df[(df['Group'] == group) & (df['Type'] == p_type)]
            if row.empty:
                means[p_type].append(0)
                errs[p_type].append([0, 0])
            else:
                m = row['Mean'].iloc[0]
                l = row['Error_Lower'].iloc[0]
                u = row['Error_Upper'].iloc[0]
                means[p_type].append(m)
                errs[p_type].append([l, u])

    colors = ['#666666', '#C43F7B', '#6FB2E4', '#64A0CD']  # darker variation of #6FB2E4

    # Plot grouped bars in the order of plot_types
    offsets = np.linspace(
        -bar_width * (len(plot_types) - 1) / 2,
        bar_width * (len(plot_types) - 1) / 2,
        len(plot_types)
    )
    all_bars = []
    for off, p_type, color in zip(offsets, plot_types, colors):
        ya = means[p_type]
        ye = np.array(errs[p_type]).T
        bars = ax.bar(
            x + off,
            ya,
            width=bar_width,
            label=p_type.replace('_', ' ').title(),
            yerr=ye,
            capsize=5,
            alpha=0.9,
            color=color,
            error_kw={'ecolor': 'black', 'linewidth': 1.5}
        )
        all_bars.extend(bars)

    # Labeling and formatting
    ax.set_ylabel(metric.replace('_', ' ').title())
    # ax.set_title(f'{metric.replace("_", " ").title()} (95% CI)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(["Class", "Architecture", "Topology", "Homology"])
    ax.grid(axis='y', linestyle='--', alpha=0.4, color='lightgray')
    ax.set_axisbelow(True)

    # Add value labels above bars
    for bar in all_bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f'{h:.3f}', ha='center', va='bottom')

    ax.legend(facecolor='white', edgecolor='lightgray', labelcolor='black')
    plt.tight_layout()
    return plt
