# Import the required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

# Set the test problem and category of interest
category = 1  # Region of interest
instance_name = 'knee'  # 'knee', 'f2-minimal', or 'centroid'
testproblem = 'VehicleCrash'  # 'TwoBarTruss' or 'VehicleCrash'

# File paths for different problems
file_paths = {
    'TwoBarTruss': r"C:\Users\emmer\Dropbox\CODING\ShapleySimple\mydata\TwoBarTrussLabeled.csv",
    'VehicleCrash': r"C:\Users\emmer\Dropbox\CODING\ShapleySimple\mydata\VehicleCrashLabeled.csv"
}

# Load data
if testproblem not in file_paths:
    print(f"Invalid test problem '{testproblem}'. Defaulting to 'TwoBarTruss'.")
    testproblem = 'TwoBarTruss'

df = pd.read_csv(file_paths[testproblem])

# Normalize a DataFrame column
def normalize_column(df, column):
    return (df[column] - df[column].min()) / (df[column].max() - df[column].min())

# Compute objectives and instance selection
def compute_objectives_and_instances(df, testproblem):
    if testproblem == 'VehicleCrash':
        for col in ['Weight', 'Acceleration', 'Intrusion']:
            df[col] = normalize_column(df, col)

        df['Objective'] = 0.333 * df['Weight'] + 0.333 * df['Acceleration'] + 0.333 * df['Intrusion']
        knee_point = df.loc[df['Objective'].idxmin(), :]
        knee_point_objective = df.loc[df['Objective'].idxmin(), ['Weight', 'Acceleration', 'Intrusion']]
        return knee_point, knee_point_objective

# Compute instances and objectives
knee_point, knee_point_objective = compute_objectives_and_instances(df, testproblem)

# Plotting function for 3D Scatterplot
def plot_3d_scatter(df, instance_point, title):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter all points
    ax.scatter(df['Weight'], df['Acceleration'], df['Intrusion'], label='Data Points', alpha=0.6)
    
    # Highlight the instance in red
    ax.scatter(instance_point['Weight'], instance_point['Acceleration'], instance_point['Intrusion'], 
               color='red', s=100, label='Instance Point')
    
    ax.set_xlabel('Weight', labelpad=20)  # Add padding to prevent cropping
    ax.set_ylabel('Acceleration', labelpad=20)
    ax.set_zlabel('Intrusion', labelpad=20)
    ax.set_title(title, pad=30)  # Add padding for the title
    ax.legend()
    
    # Adjust layout to prevent cropping
    fig.tight_layout()
    plt.show()

# Plotting function for 2D Scatterplots
def plot_2d_projections(df, instance_point, title):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs[0, 0].scatter(df['Weight'], df['Acceleration'], alpha=0.6, label='Data Points')
    axs[0, 0].scatter(instance_point['Weight'], instance_point['Acceleration'], color='red', s=100, label='Instance Point')
    axs[0, 0].set_xlabel('Weight')
    axs[0, 0].set_ylabel('Acceleration')
    axs[0, 0].set_title('Weight vs Acceleration')
    axs[0, 0].legend()
    axs[0, 1].scatter(df['Weight'], df['Intrusion'], alpha=0.6, label='Data Points')
    axs[0, 1].scatter(instance_point['Weight'], instance_point['Intrusion'], color='red', s=100, label='Instance Point')
    axs[0, 1].set_xlabel('Weight')
    axs[0, 1].set_ylabel('Intrusion')
    axs[0, 1].set_title('Weight vs Intrusion')
    axs[0, 1].legend()
    axs[1, 0].scatter(df['Acceleration'], df['Intrusion'], alpha=0.6, label='Data Points')
    axs[1, 0].scatter(instance_point['Acceleration'], instance_point['Intrusion'], color='red', s=100, label='Instance Point')
    axs[1, 0].set_xlabel('Acceleration')
    axs[1, 0].set_ylabel('Intrusion')
    axs[1, 0].set_title('Acceleration vs Intrusion')
    axs[1, 0].legend()
    axs[1, 1].axis('off')
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# Call the functions for 3D and 2D plots
if testproblem == 'VehicleCrash':
    plot_3d_scatter(df, knee_point_objective, "Vehicle Crash Example - 3D Scatterplot")
    plot_2d_projections(df, knee_point_objective, "Vehicle Crash Example - 2D Projections")
