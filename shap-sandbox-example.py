import sys
sys.path.append(r"C:\Users\emmer\AppData\Local\Programs\Python\Python312\Lib\site-packages")
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Generate a random 2D dataset with binary labels
np.random.seed(42)
X = np.random.rand(200, 2)  # 200 points in 2D
y = (1.5 * X[:, 0] + X[:, 1] > 1.5).astype(int)  # Weight x1 more heavily in the decision boundary

# Create a DataFrame for consistency
df = pd.DataFrame(X, columns=["x1", "x2"])
df["label"] = y

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df[["x1", "x2"]], df["label"], test_size=0.2, random_state=42)

# Train an XGBoost model
model = xgb.XGBClassifier(objective="binary:logistic", use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# Explain the model predictions using SHAP
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Beeswarm plot
plt.figure()
shap.summary_plot(shap_values, X_test)
plt.savefig("beeswarm_plot_weighted_x1.png")
plt.show()

# Pick a single instance for waterfall plot
instance = X_test.iloc[0:1]
instance_shap_values = explainer(instance)

# Waterfall plot
plt.figure()
shap.waterfall_plot(instance_shap_values[0])
plt.savefig("waterfall_plot_weighted_x1.png")
plt.show()

# Visualize the dataset with updated colors
plt.figure()
# Plot points based on their labels with light grey for Class 0 and black for Class 1
plt.scatter(X[y == 0, 0], X[y == 0, 1], color="lightgrey", alpha=0.6, edgecolors="k", label="Class 0")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color="black", alpha=0.8, edgecolors="k", label="Class 1")

# Highlight the explained instance (ensure it's plotted last and visibly larger)
plt.scatter(
    instance.iloc[0, 0], instance.iloc[0, 1], color="red", edgecolor="black", s=200, label="Individual Instance", zorder=5
)

# Add labels, title, and legend
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Randomly Generated 2D Dataset with Binary Label")
plt.legend(loc="lower left", fontsize=10)  # Move legend to lower left
plt.savefig("dataset.png")
plt.show()

# Calculate mean absolute SHAP values
mean_shap_values = np.abs(shap_values.values).mean(axis=0)

# Confirm the computation
print(f"Mean SHAP values: {mean_shap_values}")

# Plot mean SHAP values
plt.figure(figsize=(8, 5))
plt.bar(X_test.columns, mean_shap_values, color="blue", alpha=0.7)
plt.xlabel("Features")
plt.ylabel("Mean Absolute SHAP Value")
plt.title("Mean Absolute SHAP Values for Features")
plt.savefig("mean_shap_values.png")
plt.show()
