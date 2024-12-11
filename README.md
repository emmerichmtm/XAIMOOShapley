# SHAP Explanation for Multiobjective Optimization

This project demonstrates the application of SHAP (SHapley Additive exPlanations) to explain machine learning models trained on data from multiobjective optimization problems. The current implementation supports two test problems: **TwoBarTruss** and **VehicleCrash**.

## Features
- Normalization of features for better interpretability.
- Identification of specific regions of interest:
  - **Knee region**
  - **F2-minimal region**
  - **Centroid (mean) region**
- Visualization of results with scatter plots and SHAP summary plots.
- Global and local SHAP explanations for model predictions.

## Requirements
Install the following Python libraries:
- `pandas`
- `xgboost`
- `shap`
- `matplotlib`
- `scikit-learn`

Use the following command to install all dependencies:
```bash
pip install pandas xgboost shap matplotlib scikit-learn
```
## Repository Structure

### Scripts

1. **`shap-sandbox-example.py`**
   - A sandbox script for experimenting with SHAP values on binary classification models.
   - Demonstrates model explainability through SHAP plots.
   - Focused on smaller-scale datasets for quick experimentation.

2. **`shap-examples-main.py`**
   - The primary script for SHAP analysis configured for the two examples twoBarTruss and VehicleCrash.
   - Handles binary classification tasks across multiple regions of interest.
   - Includes functionality for both local and global model explainability.

3. **`scatterplot3d.py`**
   - Visualizes multidimensional datasets using 3D scatterplots and corresponding 2D projections.
   - Highlights specific instances in both 3D and 2D visualizations.
   - Designed for test problems such as `VehicleCrash`.

---

### `mydata` Folder

- **Description**: Contains the datasets required for the scripts.
- **CSV Files**:
  - **`TwoBarTrussLabeled.csv`**: Dataset for the "Two-Bar Truss" test problem.
  - **`VehicleCrashLabeled.csv`**: Dataset for the "Vehicle Crash" test problem.
- These files are normalized and processed within the scripts to compute objectives, highlight regions of interest, and demonstrate explainability.

---
## License

This project is licensed under the [Creative Commons Attribution 2.0 Generic (CC BY 2.0)](https://creativecommons.org/licenses/by/2.0/) license.

