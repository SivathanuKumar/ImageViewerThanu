import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os


def find_elbow_point(x_values, y_values):
    coords = np.column_stack((x_values, y_values))
    vectors = np.diff(coords, axis=0)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    angle_changes = np.diff(angles)
    elbow_idx = np.argmax(np.abs(angle_changes)) + 1
    return x_values[elbow_idx]


def analyze_trend(x_values, y_values):
    derivative = np.gradient(y_values, x_values)
    threshold = np.mean(derivative) + np.std(derivative)
    optimal_idx = np.where(derivative < threshold)[0][0]
    return x_values[optimal_idx]


def find_minimal_increase(x_values, y_values):
    pct_changes = np.diff(y_values) / y_values[:-1] * 100
    optimal_idx = np.argmin(pct_changes)
    return x_values[optimal_idx]


def find_optimal_combined(x_values, y_values):
    x_norm = (x_values - np.min(x_values)) / (
            np.max(x_values) - np.min(x_values))
    y_norm = (y_values - np.min(y_values)) / (
            np.max(y_values) - np.min(y_values))

    first_derivative = np.gradient(y_norm, x_norm)
    second_derivative = np.gradient(first_derivative, x_norm)

    scores = y_norm + np.abs(first_derivative) + np.abs(second_derivative)
    optimal_idx = np.argmin(scores)

    return x_values[optimal_idx]


def find_sign_change(x, y):
    for i in range(len(y) - 1):
        if y[i] <= 0 and y[i + 1] > 0:
            x0, x1 = x[i], x[i + 1]
            y0, y1 = y[i], y[i + 1]
            crossing_x = x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
            return crossing_x
    return None


def analyze_and_plot(file_path, include_gaussian_sd=False):
    # Read CSV file
    df = pd.read_csv(file_path)
    save_dir = os.path.dirname(file_path)

    # Define the metrics to analyze
    metrics = {
        'SD': 'Standard Deviation',
        'Mean': 'Mean',
        'CV': 'Coefficient of Variation',
        'Skewness': 'Skewness'
    }

    # Add Gaussian_SD to metrics if it exists in the CSV and include_gaussian_sd is True
    if include_gaussian_sd and 'Gaussian_SD' in df.columns:
        metrics['Gaussian_SD'] = 'Gaussian Standard Deviation'

    gamma_values = df['Parameter'].values

    for metric, title in metrics.items():
        # Skip if the metric column doesn't exist
        if metric not in df.columns:
            print(f"Warning: {metric} not found in CSV file. Skipping...")
            continue

        plt.figure(figsize=(12, 7))
        values = df[metric].values

        # Plot main curve
        plt.plot(gamma_values, values, 'b-', label=metric, linewidth=2)

        # Find and mark minimum point
        min_idx = np.argmin(values)
        min_gamma = gamma_values[min_idx]
        min_value = values[min_idx]

        plt.plot(min_gamma, min_value, 'ro',
                 label=f'Minimum (γ={min_gamma:.2f})')

        # Calculate analytical points
        elbow_gamma = find_elbow_point(gamma_values, values)
        trend_gamma = analyze_trend(gamma_values, values)
        min_increase_gamma = find_minimal_increase(gamma_values, values)
        combined_gamma = find_optimal_combined(gamma_values, values)

        # Plot analytical points
        plt.axvline(x=elbow_gamma, color='g', linestyle='--',
                    label=f'Elbow Point (γ={elbow_gamma:.2f})')
        plt.axvline(x=trend_gamma, color='m', linestyle='--',
                    label=f'Rate Change (γ={trend_gamma:.2f})')
        plt.axvline(x=min_increase_gamma, color='c', linestyle='--',
                    label=f'Min Increase (γ={min_increase_gamma:.2f})')
        plt.axvline(x=combined_gamma, color='y', linestyle='--',
                    label=f'Combined (γ={combined_gamma:.2f})')

        # For skewness, add sign change point
        if metric == 'Skewness':
            crossing_point = find_sign_change(gamma_values, values)
            if crossing_point is not None:
                plt.axvline(x=crossing_point, color='k', linestyle='-.',
                            label=f'Sign Change (γ={crossing_point:.2f})')
                plt.plot(crossing_point, 0, 'ko')

        # Special check for Gaussian_SD: if minimum is at 4.0, mark it specially
        if metric == 'Gaussian_SD' and abs(min_gamma - 4.0) < 0.01:
            plt.plot(min_gamma, min_value, 'rx', markersize=12,
                     label=f'Warning: Minimum at max value (γ=4.0)')

        plt.xlabel('Gamma')
        plt.ylabel(title)
        plt.title(f'{title} vs Gamma Analysis')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Save plot with extra space for legend
        save_path = os.path.join(save_dir, f'0_{metric}_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Print analysis results
        print(f"\n{title} Analysis:")
        print(f"Minimum value: {min_value:.4f} at γ = {min_gamma:.2f}")
        print(f"Elbow point: γ = {elbow_gamma:.2f}")
        print(f"Rate change point: γ = {trend_gamma:.2f}")
        print(f"Minimal increase point: γ = {min_increase_gamma:.2f}")
        print(f"Combined analysis point: γ = {combined_gamma:.2f}")
        if metric == 'Skewness' and crossing_point is not None:
            print(f"Sign change occurs at γ = {crossing_point:.2f}")
        if metric == 'Gaussian_SD' and abs(min_gamma - 4.0) < 0.01:
            print(f"WARNING: Minimum Gaussian SD at maximum γ value (4.0). Consider using Skewness metric instead.")

    # Create a comparison plot for SD vs Gaussian_SD if both exist
    if include_gaussian_sd and 'SD' in df.columns and 'Gaussian_SD' in df.columns:
        plt.figure(figsize=(12, 7))

        # Normalize values for better comparison
        sd_values = df['SD'].values
        gaussian_sd_values = df['Gaussian_SD'].values

        # Plot both curves
        plt.plot(gamma_values, sd_values, 'b-', label='Standard Deviation', linewidth=2)
        plt.plot(gamma_values, gaussian_sd_values, 'r-', label='Gaussian Standard Deviation', linewidth=2)

        # Find and mark minimum points
        sd_min_idx = np.argmin(sd_values)
        sd_min_gamma = gamma_values[sd_min_idx]
        sd_min_value = sd_values[sd_min_idx]

        g_sd_min_idx = np.argmin(gaussian_sd_values)
        g_sd_min_gamma = gamma_values[g_sd_min_idx]
        g_sd_min_value = gaussian_sd_values[g_sd_min_idx]

        plt.plot(sd_min_gamma, sd_min_value, 'bo', label=f'SD Minimum (γ={sd_min_gamma:.2f})')
        plt.plot(g_sd_min_gamma, g_sd_min_value, 'ro', label=f'Gaussian SD Minimum (γ={g_sd_min_gamma:.2f})')

        # If Gaussian SD minimum is at 4.0, mark it specially
        if abs(g_sd_min_gamma - 4.0) < 0.01:
            plt.plot(g_sd_min_gamma, g_sd_min_value, 'rx', markersize=12,
                     label=f'Warning: G-SD Minimum at max value (γ=4.0)')

        # Add skewness sign change if available
        if 'Skewness' in df.columns:
            skewness_values = df['Skewness'].values
            crossing_point = find_sign_change(gamma_values, skewness_values)
            if crossing_point is not None:
                plt.axvline(x=crossing_point, color='k', linestyle='-.',
                            label=f'Skewness Sign Change (γ={crossing_point:.2f})')

        plt.xlabel('Gamma')
        plt.ylabel('Value')
        plt.title('SD vs Gaussian SD Comparison')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Save comparison plot
        save_path = os.path.join(save_dir, f'0_SD_GaussianSD_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print("\nCreated SD vs Gaussian SD comparison plot")


def select_and_analyze(path=None, include_gaussian_sd=False):
    if path is None:
        root = tk.Tk()
        root.withdraw()

        file_path = filedialog.askopenfilename(
            title='Select CSV file',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
        )

        if file_path:
            try:
                analyze_and_plot(file_path, include_gaussian_sd)
                print(f"\nAnalysis complete! Plots have been saved in: {os.path.dirname(file_path)}")
            except Exception as e:
                print(f"Error during analysis: {str(e)}")
        else:
            print("No file selected.")
    else:
        analyze_and_plot(path, include_gaussian_sd)


if __name__ == "__main__":
    select_and_analyze(include_gaussian_sd=True)