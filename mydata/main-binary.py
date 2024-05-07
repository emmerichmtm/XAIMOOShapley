# import the required libraries

import pandas as pd
import xgboost as xgb
import shap
from shap.plots import waterfall
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# load data into a pandas DataFrame
df = pd.read_csv(r"C:\Users\emmer\Dropbox\CODING\ShapleySimple\mydata\TwoBarTrussLabeled.csv")
# define the category of interest: 1 for knee point, 2 for f2-minimal point (MaxStress)
category = 2
instance_name = ('knee')
testproblem = 'TwoBarTruss'

# Load data into a pandas DataFrame
if testproblem=='TwoBarTruss':
    df = pd.read_csv(r"C:\Users\emmer\Dropbox\CODING\ShapleySimple\mydata\TwoBarTrussLabeled.csv")
elif testproblem=='VehicleCrash':
    df = pd.read_csv(r"C:\Users\emmer\Dropbox\CODING\ShapleySimple\mydata\VehicleCrashLabeled.csv")
else:
    print("Problem not found, using TwoBarTruss as default.")
    testproblem = 'TwoBarTruss'
    df = pd.read_csv(r"C:\Users\emmer\Dropbox\CODING\ShapleySimple\mydata\TwoBarTrussLabeled.csv")

# For simplicity have in the following hard coded the testproblems, but the code is repeated for each problem
# Bonus: It could be refactored to make it more general and avoid code repetition
if testproblem=='TwoBarTruss':
    # Compute some normalization of the objective function values
    Volume = df['Volume'] # also referred to as f1-objective, extremely good in region 3
    MaxStress = df['MaxStress'] # also referred to as f2-objective, extremely good in region 2
    # Normalize the MaxStress and Volume columns to 0-1 scale
    df['MaxStress'] = (MaxStress - MaxStress.min()) / (MaxStress.max() - MaxStress.min())
    df['Volume'] = (Volume - Volume.min()) / (Volume.max() - Volume.min())
    MaxStress = df['MaxStress']
    Volume = df['Volume']

    # Compute a decision vector in the knee point region
    df['Objective'] = 0.5 * MaxStress + 0.5 * Volume
    # Make a vector of the point with the lowest value of 0.5 *MaxStress + 0.5 *Volume
    knee_point = df.loc[df['Objective'].idxmin(), ['x1', 'x2', 'y']]
    # Find the knee point in the MaxStress and Volume columns and call it knee_point_objective
    knee_point_objective = df.loc[df['Objective'].idxmin(), ['MaxStress', 'Volume']]
    print("The knee point is:", knee_point, "with objective values:", knee_point_objective)

    # Compute a decision vector that minimizes f2
    df['Objective'] = MaxStress
    f2_point = df.loc[df['Objective'].idxmin(), ['x1', 'x2', 'y']]
    f2_point_objective = df.loc[df['Objective'].idxmin(), ['MaxStress', 'Volume']]
    print("The f2 (Volume) minimal point is:", f2_point, "with objective values:", f2_point_objective)

    # Compute a closest to the mean point in the MaxStress and Volume columns
    mean_point = [0.3, 0.3]
    # closest to the mean point in the MaxStress and Volume columns
    df['Objective'] = (df['MaxStress'] - mean_point[0]) ** 2 + (df['Volume'] - mean_point[1]) ** 2
    centroid_point = df.loc[df['Objective'].idxmin(), ['x1', 'x2', 'y']]
    centroid_point_objective = df.loc[df['Objective'].idxmin(), ['MaxStress', 'Volume']]
    print("The mean point is:", centroid_point, "with objective values:", centroid_point)

    # make a scatter plot of MaxStress and Volume columns and mark the sample point in big red
    plt.scatter(Volume, MaxStress)
    if instance_name == 'knee':
        plt.scatter(knee_point_objective['Volume'],knee_point_objective['MaxStress'], color='red', s=100)
    elif instance_name == 'f2-minimal':
         plt.scatter(f2_point_objective['Volume'],f2_point_objective['MaxStress'], color='red', s=100)
    elif instance_name == 'centroid':
     plt.scatter(centroid_point_objective['Volume'], centroid_point_objective['MaxStress'], color='red', s=100)
    else:
     plt.scatter(instance['Volume'], instance['MaxStress'], color='red', s=100)
    plt.xlabel('MaxStress')
    plt.ylabel('Volume')
    plt.show()

    # Assuming you have already loaded your data into a DataFrame 'df'
    input_columns = ['x1','x2', 'y']  # Specify your input feature columns
    output_column = 'category'  # Specify the name of your target variable column
    """
    Global explanation for the binary classification problem of region of interest (knee point = 1, f2-minimal point = 3
    """

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df[input_columns], df[output_column], test_size=0.2, random_state=42)

    # transform the target variable to a binary variable: 1 for category 1, and 0 for category 4

    # Category: 1 = knee point, 3 = f2-minimal point
    y_train_binary = (y_train == category).astype(int)
    y_test_binary = (y_test == category).astype(int)

    # Define and train the XGBoost model with binary logistic regression
    model_binary = xgb.XGBClassifier(objective='binary:logistic')
    model_binary.fit(X_train, y_train_binary)

    # Use SHAP to explain the model's predictions for the binary classification problem
    explainer_binary = shap.Explainer(model_binary)
    shap_values_binary = explainer_binary.shap_values(X_test)

    # What is the expected value of the model? (= the log odds of the probability of the point being in the region of interest)
    print("Global expected value of log odds:", explainer_binary.expected_value)
    global_base_value = explainer_binary.expected_value

    # Generate a mean importance plot for the binary classification problem and save as PDF
    shap.summary_plot(shap_values_binary, X_test, plot_type='bar')
    plt.savefig('shap_summary_mean_importance.png')

    # Add a beeswarm plot for the binary classification problem and save as PDF
    shap.summary_plot(shap_values_binary, X_test)
    plt.savefig('shap_summary_plot_beeswarm.png')

    """
    Local explanation for the binary classification problem of region of interest (knee point = 1, f2-minimal point = 3
    """

    # Use instance (see computation above) to explain the samples
    if (instance_name == 'knee'):
        instance = pd.DataFrame([knee_point], columns=['x1', 'x2', 'y'])
    elif (instance_name == 'f2-minimal'):
        instance = pd.DataFrame([f2_point], columns=['x1', 'x2', 'y'])
        print("Instance f2 min:", instance)
    elif (instance_name == 'centroid'):
        instance = pd.DataFrame([centroid_point], columns=['x1', 'x2', 'y'])
        print("Instance centroid:", instance)
    else:
        instance = X_test.iloc[0:1]

    model_binary.predict(instance)
    shap_values_instance = explainer_binary.shap_values(instance)

    # print name of columns (features) and their shape values
    print("Variables:", instance.columns)
    print("Variable Values", instance)
    print("SHAP values for the instance:", shap_values_instance[0])

    # Generate a force plot for the binary classification problem (does not work well with the current version of SHAP)
    shap.force_plot(explainer_binary.expected_value, shap_values_instance, instance, matplotlib=True)

    """
    Generate a waterfall plot for the binary classification problem of region of interest 
    (knee point = 1, f2-minimal point = 3)
    """
    # Given SHAP values
    # use shapley values from the instance
    shap_values = shap_values_instance[0]
    print('shapvalues', shap_values)

    # Features
    features = ['x1', 'x2', 'y']

    # Initialize cumulative sum
    cumulative_sum=global_base_value
    # Initialize plot
    plt.figure(figsize=(8, 6))

    # increase font size
    plt.rcParams.update({'font.size': 16})
    # Plot bars for each feature.
    n= len(shap_values)-1
    for i in range(len(shap_values)):
        plt.barh(features[i], shap_values[i], left=cumulative_sum, color='lightblue' if shap_values[i] >= 0 else 'red')
        # add name of the feature above
        plt.text(cumulative_sum + shap_values[i] / 2, i, features[i], ha='right', va='top', color='black')
        # add a dashed vertical line at the end of the bar
        xcoord=cumulative_sum + shap_values[i]
        #plt.axvline(x=xcoord-1.4, color='gray', linestyle='dashed', linewidth=1)
        cumulative_sum += shap_values[i]

    # Add labels and title
    plt.xlabel('SHAP Value')
    plt.ylabel('Feature')
    plt.title('Waterfall-plot')
    # remove y-axis ticks
    plt.yticks([])
    # remove! all vertical lines in bar plot
    plt.grid(False)
    plt.show()
elif testproblem=='VehicleCrash':
    #Vehicle crash worthiness:
    #
    # Category ‘1’ -- Knee: Weight <= 1685 & Acceleration <= 8.5 & Intrusion <= 0.10
    # Category ‘2’ -- F1Extreme: Weight >= 1695
    # Category ‘3’ -- F2Extreme: Acceleration >= 10.5 & Weight <= 1685
    # Category ‘4’ -- F3Extreme: Intrusion >= 0.20
    # Category ‘5’ -- Dominated: Acceleration >= 9.0 & Intrusion >= 0.15
    # Category ‘-1’ – Rest of the points
    # Compute some normalization of the objective function values
    # Weight	Acceleration	Intrusion
    Weight = df['Weight']
    Acceleration = df['Acceleration']
    Intrusion = df['Intrusion']
    # Normalize the Weight, Acceleration and Intrusion columns to 0-1 scale
    df['Weight'] = (Weight - Weight.min()) / (Weight.max() - Weight.min())
    df['Acceleration'] = (Acceleration - Acceleration.min()) / (Acceleration.max() - Acceleration.min())
    df['Intrusion'] = (Intrusion - Intrusion.min()) / (Intrusion.max() - Intrusion.min())

    # Compute a decision vector in the knee point region
    df['Objective'] = 0.5 * Weight + 0.5 * Acceleration + 0.5 * Intrusion

    # Make a vector of the point with the lowest value of 0.5 *MaxStress + 0.5 *
    # features (decision variables) are x1, x2, x3, x4, x5
    knee_point = df.loc[df['Objective'].idxmin(), ['x1', 'x2', 'x3', 'x4','x5']]

    # Find the knee point in the MaxStress and Volume columns and call it knee_point_objective
    knee_point_objective = df.loc[df['Objective'].idxmin(), ['Weight', 'Acceleration', 'Intrusion']]
    print("The knee point is:", knee_point, "with objective values:", knee_point_objective)

    # Compute a decision vector that minimizes f2
    df['Objective'] = Acceleration
    f2_point = df.loc[df['Objective'].idxmin(), ['x1', 'x2', 'x3', 'x4','x5']]
    f2_point_objective = df.loc[df['Objective'].idxmin(), ['Weight', 'Acceleration', 'Intrusion']]
    print("The f2 (Volume) minimal point is:", f2_point, "with objective values:", f2_point_objective)

    # Compute a closest to the mean point in the MaxStress and Volume columns
    mean_point = [0.3, 0.3]
    # closest to the mean point in the MaxStress and Volume columns
    df['Objective'] = (df['Weight'] - mean_point[0]) ** 2 + (df['Acceleration'] - mean_point[1]) ** 2 + (df['Intrusion'] - mean_point[2]) ** 2
    centroid_point = df.loc[df['Objective'].idxmin(), ['x1', 'x2', 'x3', 'x4','x5']]
    centroid_point_objective = df.loc[df['Objective'].idxmin(), ['Weight', 'Acceleration', 'Intrusion']]
    print("The mean point is:", centroid_point, "with objective values:", centroid_point)

    # make a 3d scatter plot of Weight, Acceleration, and Intrusion columns and mark the sample point in big red
    plt.scatter(Weight, Acceleration, Intrusion)
    if instance_name == 'knee':
        plt.scatter(knee_point_objective['Weight'], knee_point_objective['Acceleration'],  knee_point_objective['Intrusion'], color='red', s=100)
    elif instance_name == 'f2-minimal':
        plt.scatter(f2_point_objective['Weight'], f2_point_objective['Acceleration'], knee_point_objective['Intrusion'],  color='red', s=100)
    elif instance_name == 'centroid':
        plt.scatter(centroid_point_objective['Weight'], centroid_point_objective['Acceleration'], centroid_point['Intrusion'], color='red', s=100)
    else:
        print("Instance not found!")
    plt.xlabel('Weight')
    plt.ylabel('Acceleration')
    plt.zlabel('Intrusion')
    plt.show()

    # Assuming you have already loaded your data into a DataFrame 'df'
    input_columns = ['x1','x2','x3','x4','x5']  # Specify your input feature columns
    output_column = 'category'  # Specify the name of your target variable column

    """
    Global explanation for the binary classification problem of region of interest (knee point = 1, f2-minimal point = 3
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df[input_columns], df[output_column], test_size=0.2,
                                                        random_state=42)

    # transform the target variable to a binary variable: 1 for category 1, and 0 for category 4

    # Category: 1 = knee point, 3 = f2-minimal point
    y_train_binary = (y_train == category).astype(int)
    y_test_binary = (y_test == category).astype(int)

    # Define and train the XGBoost model with binary logistic regression
    model_binary = xgb.XGBClassifier(objective='binary:logistic')
    model_binary.fit(X_train, y_train_binary)

    # Use SHAP to explain the model's predictions for the binary classification problem
    explainer_binary = shap.Explainer(model_binary)
    shap_values_binary = explainer_binary.shap_values(X_test)

    # What is the expected value of the model? (= the log odds of the probability of the point being in the region of interest)
    print("Global expected value of log odds:", explainer_binary.expected_value)
    global_base_value = explainer_binary.expected_value

    # Generate a mean importance plot for the binary classification problem and save as PDF
    shap.summary_plot(shap_values_binary, X_test, plot_type='bar')
    plt.savefig('shap_summary_mean_importance.png')

    # Add a beeswarm plot for the binary classification problem and save as PDF
    shap.summary_plot(shap_values_binary, X_test)
    plt.savefig('shap_summary_plot_beeswarm.png')

    """
    Local explanation for the binary classification problem of region of interest (knee point = 1, f2-minimal point = 3
    """

    # Use instance (see computation above) to explain the samples
    if (instance_name == 'knee'):
        instance = pd.DataFrame([knee_point], columns=['x1','x2','x3','x4','x5'])
    elif (instance_name == 'f2-minimal'):
        instance = pd.DataFrame([f2_point], columns=['x1','x2','x3','x4','x5'])
        print("Instance f2 min:", instance)
    elif (instance_name == 'centroid'):
        instance = pd.DataFrame([centroid_point], columns=['x1','x2','x3','x4','x5'])
        print("Instance centroid:", instance)
    else:
        instance = X_test.iloc[0:1]

    model_binary.predict(instance)
    shap_values_instance = explainer_binary.shap_values(instance)

    # print name of columns (features) and their shape values
    print("Variables:", instance.columns)
    print("Variable Values", instance)
    print("SHAP values for the instance:", shap_values_instance[0])

    # Generate a force plot for the binary classification problem (does not work well with the current version of SHAP)
    shap.force_plot(explainer_binary.expected_value, shap_values_instance, instance, matplotlib=True)

    """
    Generate a waterfall plot for the binary classification problem of region of interest 
    (knee point = 1, f2-minimal point = 3)
    """
    # Given SHAP values
    # use shapley values from the instance
    shap_values = shap_values_instance[0]
    print('shapvalues', shap_values)

    # Features
    features = ['x1', 'x2', 'x3', 'x4', 'x5']

    # Initialize cumulative sum
    cumulative_sum = global_base_value
    # Initialize plot
    plt.figure(figsize=(8, 6))

    # increase font size
    plt.rcParams.update({'font.size': 16})
    # Plot bars for each feature.
    n = len(shap_values) - 1
    for i in range(len(shap_values)):
        plt.barh(features[i], shap_values[i], left=cumulative_sum, color='lightblue' if shap_values[i] >= 0 else 'red')
        # add name of the feature above
        plt.text(cumulative_sum + shap_values[i] / 2, i, features[i], ha='right', va='top', color='black')
        # add a dashed vertical line at the end of the bar
        xcoord = cumulative_sum + shap_values[i]
        # plt.axvline(x=xcoord-1.4, color='gray', linestyle='dashed', linewidth=1)
        cumulative_sum += shap_values[i]

    # Add labels and title
    plt.xlabel('SHAP Value')
    plt.ylabel('Feature')
    plt.title('Waterfall-plot')
    # remove y-axis ticks
    plt.yticks([])
    # remove! all vertical lines in bar plot
    plt.grid(False)
    plt.show()
