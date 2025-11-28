"""
Visualization functions for the MiniLab.
Creates plots for evaluating models and learning.

  - Visualization of model complexity (Features) and performance (Task 2.4)
  - Heatmap & curve plots for polynomial models (Task 3.2, 3.3)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Visualization of model complexity (Features) and performance (Task 2.4)
# ---------------------------------------------------------------------
def plot_feature_performance(results, output_dir="results", file_name="Task_2"):
    """
    Create and save a performance plot showing how model performance
    (R² and RMSE) changes with the number of features.

    Parameters
    ----------
    results : list of dict
        Example format:
        [
            {"n_features": 1, "features": ["livingSpace"], "r2": 0.70, "rmse": 250.0},
            {"n_features": 2, "features": ["livingSpace", "numberOfRooms"], "r2": 0.78, "rmse": 210.0},
            ...
        ]
    output_dir : str, default="results"
        Directory where the plot is saved.
    file_name : str, default="Task_2"
        Identifier used for the filename.

    Returns
    -------
    str
        Path to the saved plot file.
    """
    os.makedirs(output_dir, exist_ok=True)

    n_features = [r["n_features"] for r in results]
    r2_values = [r["r2"] for r in results]
    rmse_values = [r["rmse"] for r in results]
    r2_train_values = [r['r2_train'] for  r in results]
    rmse_train_values = [r['rmse_train'] for r in results]

    # --- Create figure ---
    fig, ax1 = plt.subplots(figsize=(6, 4))

    color_r2 = "tab:blue"
    color_rmse = "tab:red"

    ax1.set_xlabel("Number of features")
    ax1.set_ylabel("R² (val:green, train:blue)", color=color_r2)
    ax1.plot(n_features, r2_values, color_r2)
    ax1.plot(n_features,r2_train_values, color='tab:green')
    ax1.tick_params(axis="y", labelcolor=color_r2)
    

    ax2 = ax1.twinx()
    ax2.set_ylabel("RMSE (€) (val:pink, train:red)", color=color_rmse)
    ax2.plot(n_features, rmse_values, color_rmse)
    ax2.plot(n_features, rmse_train_values, color='tab:pink')
    ax2.tick_params(axis="y", labelcolor=color_rmse)

    plt.title("Model performance (validation) vs. number of features")
    fig.tight_layout()

    output_path = os.path.join(output_dir, f"{file_name}.pdf")
    plt.savefig(output_path)
    plt.close(fig)

    return output_path

# ---------------------------------------------------------------------
# Heatmap for polynomial performance (Matplotlib version, Task 3.2)
# ---------------------------------------------------------------------
def plot_heatmap_performance(results_list, output_dir="results", metric="r2_val", file_name="Task_3_heatmap"):
    """
    Visualize validation performance as a heatmap over polynomial degree x number of features.
    Works directly from flat list of dicts.
    """

    """
    # Creates a 3×6 heatmap with random values between 0 and 1.
    df_res = pd.DataFrame(results_list)
    df_res['n_features'] = df_res['features'].apply(len)
    heatmap_df = df_res.pivot(index= 'n_features', columns= 'degree', values='r2_val')
    heatmap_data = heatmap_df.to_numpy()
    print(heatmap_data)

    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(heatmap_data, cmap="viridis", origin="lower", aspect="auto")

    # Axis labels
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("Feature set index")

    ax.set_xticks(range(6))
    ax.set_xticklabels([1, 2, 3, 4, 5, 6])

    ax.set_yticks(range(3))
    ax.set_yticklabels([f"Set {i+1}" for i in range(3)])
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    if metric == "r2_val":
        data = {
            '1':{'1 Degree': results_list[0]['r2_val'], '2 Degree': results_list[1]['r2_val'], '3 Degree': results_list[2]['r2_val'], '4 Degree': results_list[3]['r2_val'], '5 Degree': results_list[4]['r2_val'], '6 Degree': results_list[5]['r2_val']}, 
            '2':{'1 Degree': results_list[6]['r2_val'], '2 Degree': results_list[7]['r2_val'], '3 Degree': results_list[8]['r2_val'], '4 Degree': results_list[9]['r2_val'], '5 Degree': results_list[10]['r2_val'], '6 Degree': results_list[11]['r2_val']}, 
            '3':{'1 Degree': results_list[12]['r2_val'], '2 Degree': results_list[13]['r2_val'], '3 Degree': results_list[14]['r2_val'], '4 Degree': results_list[15]['r2_val'], '5 Degree': results_list[16]['r2_val'], '6 Degree': results_list[17]['r2_val']}, 
        }
    else:
        data = {
            '1':{'1 Degree': results_list[0]['rmse_val'], '2 Degree': results_list[1]['rmse_val'], '3 Degree': results_list[2]['rmse_val'], '4 Degree': results_list[3]['rmse_val'], '5 Degree': results_list[4]['rmse_val'], '6 Degree': results_list[5]['rmse_val']}, 
            '2':{'1 Degree': results_list[6]['rmse_val'], '2 Degree': results_list[7]['rmse_val'], '3 Degree': results_list[8]['rmse_val'], '4 Degree': results_list[9]['rmse_val'], '5 Degree': results_list[10]['rmse_val'], '6 Degree': results_list[11]['rmse_val']}, 
            '3':{'1 Degree': results_list[12]['rmse_val'], '2 Degree': results_list[13]['rmse_val'], '3 Degree': results_list[14]['rmse_val'], '4 Degree': results_list[15]['rmse_val'], '5 Degree': results_list[16]['rmse_val'], '6 Degree': results_list[17]['rmse_val']}, 
        }

    heatmap_data = pd.DataFrame.from_dict(data, orient='index')
    print(heatmap_data)
    im = ax.imshow(heatmap_data, cmap="hot", origin="lower", aspect="auto")
    
    # Axis labels
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("Feature set index")

    ax.set_xticks(range(6))
    ax.set_xticklabels([1, 2, 3, 4, 5, 6])

    ax.set_yticks(range(3))
    ax.set_yticklabels([f"Set {i+1}" for i in range(3)])

    # Colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("R² value")

    plt.tight_layout()
	
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{file_name}.pdf")
    plt.savefig(path)
    plt.close(fig)
    #print(f"Saved heatmap to: {path}")
    return path


# ---------------------------------------------------------------------
# 2D plot: train vs. validation curves over degree (Task 3.3)
# ---------------------------------------------------------------------
def plot_polynomial_results(results_list, output_dir="results", file_name="Task_3_curves"):
    """
    Plot R² (train vs validation) over polynomial degree for a flat list of results.

    Parameters
    ----------
    results_list : list of dict
        Flat list from evaluate_polynomial_models().

    Dummy version: create exactly three PDF plots with names:
      Task_3_3_A.pdf
      Task_3_3_A_B.pdf
      Task_3_3_A_B_C.pdf
    """
    os.makedirs(output_dir, exist_ok=True)

    feature_sets = [results_list[0]["features"], results_list[8]["features"], results_list[13]["features"]]
    print (feature_sets)

    created_paths = []

    for fs in feature_sets:
        fig, ax = plt.subplots(figsize=(4, 3))

        # Dummy placeholder curve
        degrees = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        train = [r2_values["r2_train"] for r2_values in results_list[6 * (len(fs) - 1):6 * len(fs)]]
        val   = [r2_values["r2_val"] for r2_values in results_list[6 * (len(fs) - 1):6 * len(fs)]]
        ax.plot(degrees, train, marker="o", label="Train R²", ls="--")
        ax.plot(degrees, val, marker="o", label="Validate R²")

        ax.set_title("Polynomial curves")
        ax.set_xlabel("Degree")
        ax.set_ylabel("R²")
        ax.grid(True, linestyle=":")
        ax.legend(fontsize=6)

        # Build filename from base name + feature suffix
        suffix = "_".join(fs)
        path = os.path.join(output_dir, f"{file_name}_{suffix}.pdf")
        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)

        created_paths.append(path)

    return created_paths

# ---------------------------------------------------------------------
# Plot: Learning curves (RMSE) over training and fine-tuning (Task 5.3)
# ---------------------------------------------------------------------
def plot_learning_curve(model, output_dir="results", file_name="learning_curve", ref_rmse=None):
    """
    Plot RMSE learning curves from model.train_curve and model.val_curve.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_title("Learning Curve")
    ax.grid(True, linestyle=":")

    path = os.path.join(output_dir, f"{file_name}.pdf")
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)

    return path