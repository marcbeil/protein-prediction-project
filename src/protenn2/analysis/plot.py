import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_metric_with_ci(data, metric, plot_types=['raw', 'post']):
    """
    Creates a bar plot with confidence intervals for a specified metric.

    Args:
        data (dict): The input dictionary containing the metric data.
        metric (str): The name of the metric to plot (e.g., 'accuracy', 'f1_score_post_').
        plot_types (list): A list of types to include in the plot (e.g., ['raw', 'post']).
                           Can include 'raw', 'post', or both.
    """

    plot_order_prefix = ['C', 'A', 'T', 'H']
    plot_data = []

    # Ensure the order is CATH, with raw/post paired for each letter
    for prefix in plot_order_prefix:
        for p_type in plot_types: # This will ensure raw_C, post_C, then raw_A, post_A, etc.
            key = f"{p_type}_{prefix}"
            if key in data and metric in data[key]:
                mean_val = data[key][metric]['mean']
                ci_lower = data[key][metric]['ci_lower']
                ci_upper = data[key][metric]['ci_upper']
                plot_data.append({'Key': key, 'Mean': mean_val, 'CI_Lower': ci_lower, 'CI_Upper': ci_upper})

    if not plot_data:
        print(f"No data found for metric '{metric}' with types {plot_types}.")
        return

    df = pd.DataFrame(plot_data)

    # Calculate error for confidence interval (mean - lower, upper - mean)
    df['Error_Lower'] = df['Mean'] - df['CI_Lower']
    df['Error_Upper'] = df['CI_Upper'] - df['Mean']

    # Set a dark style for better visualization
    plt.style.use('dark_background')

    fig, ax = plt.subplots(figsize=(12, 7)) # Increased figure size for better readability

    x = np.arange(len(df['Key']))
    bars = ax.bar(x, df['Mean'], yerr=[df['Error_Lower'], df['Error_Upper']], capsize=5,
                  color='#34495e', # Dark blue-gray for bars
                  edgecolor='white', # White edges for distinction
                  alpha=0.9,
                  error_kw={'ecolor': 'lightgray', 'linewidth': 1.5}) # Color and thickness for error bars

    ax.set_xlabel('Key', color='white', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), color='white', fontsize=12)
    ax.set_title(f'{metric.replace("_", " ").title()} with 95% Confidence Intervals', color='white', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Key'], rotation=45, ha='right', color='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_facecolor('#2c3e50') # Darker background for the plot area

    # Customize grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.4, color='lightgray')
    ax.set_axisbelow(True) # Ensure grid is behind the bars

    # Add labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}',
                ha='center', va='bottom', fontsize=9, color='white')

    plt.tight_layout()
    return plt