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

