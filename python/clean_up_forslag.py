import pandas as pd
import numpy as np
from scipy import stats  # Import stats from scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load the dataset
df = pd.read_csv("data.csv")

# Clean the dataset using the prepare_data function
def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    # Remove rows with null values
    data = data[data.notnull().all(axis=1)]

    # Remove consecutive duplicates
    not_duplicate = data.diff(-1).any(axis=1)
    not_duplicate[not_duplicate.size - 1] = True
    data = data[not_duplicate.values]

    # Remove outliers
    non_static_columns = data.diff(-1).any(axis=0)
    data = data[(np.abs(stats.zscore(data[data.columns[non_static_columns]])) < 3).all(axis=1)]

    return data

# Apply cleaning
cleaned_df = prepare_data(df)
print(f"Cleaned data, {len(cleaned_df)} rows remaining.")

# Exclude the first column (index) and select features
x = cleaned_df.iloc[:, 1:-2].to_numpy()  # Exclude the first (index) and last two columns (decay coefficients)

# Standardize features
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Loop over the two decay coefficients (last two columns)
for i, target_column in enumerate(cleaned_df.columns[-2:], start=1):
    print(f"Analyzing Decay Coefficient {i}: {target_column}")

    # Define the target variable (binary based on threshold)
    y = cleaned_df[target_column].to_numpy()
    y_bool = list(map(lambda v: v < 0.99, y))

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y_bool, test_size=0.2, random_state=42)

    # Train logistic regression model
    model = LogisticRegression(solver='liblinear', C=0.05, random_state=0)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1]

    # --- Plot Feature Importance ---
    feature_importance = model.coef_[0]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.title(f"Feature Importance for Decay Coefficient {i}", fontsize=16)
    plt.xlabel("Feature Index", fontsize=14)
    plt.ylabel("Coefficient Value", fontsize=14)
    plt.show()

    # --- Plot Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix for Decay Coefficient {i}', fontsize=16)
    plt.xlabel('Predicted Labels', fontsize=14)
    plt.ylabel('True Labels', fontsize=14)
    plt.show()

    # --- Plot ROC Curve ---
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.title(f'ROC Curve for Decay Coefficient {i}', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(loc='lower right')
    plt.show()

    # Print classification report
    print(f"Classification Report for Decay Coefficient {i}:\n", classification_report(y_test, y_pred))

    # --- Scatter Plot of Decay Coefficient vs True Labels ---
    plt.figure(figsize=(8, 6))
    plt.scatter(cleaned_df[target_column], y_bool, alpha=0.5)
    plt.axvline(x=0.99, color='red', linestyle='--', label="Threshold = 0.99")
    plt.title(f"Decay Coefficient {i} vs True Labels", fontsize=16)
    plt.xlabel(f"Decay Coefficient {i}", fontsize=14)
    plt.ylabel("True/False Label", fontsize=14)
    plt.legend()
    plt.show()
