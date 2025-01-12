import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import numpy as np

def load_data(path: Path) -> pd.DataFrame:
    """Load the dataset from a CSV file."""
    return pd.read_csv(path)

def remove_consecutive_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """Remove consecutive duplicate rows."""
    return data.loc[(data != data.shift()).any(axis=1)]

def remove_outliers(data: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    """Remove rows with outliers based on z-scores for all numeric columns."""
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(data[numeric_cols]))
    return data[(z_scores < z_threshold).all(axis=1)]

def prepare_data(data: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    """Clean the dataset by removing duplicates and outliers."""
    data = remove_consecutive_duplicates(data)
    data = remove_outliers(data, z_threshold=z_threshold)
    return data

def visualize_data(data: pd.DataFrame):
    """Generate relevant plots to visualize data patterns and outliers."""
    # Histograms for each numeric column
    data.hist(bins=20, figsize=(20, 15))
    plt.suptitle("Histograms of Numeric Features", fontsize=20)
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation = data.corr()
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap", fontsize=18)
    plt.show()

    # Boxplots for detecting outliers
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=data[col])
        plt.title(f"Boxplot for {col}", fontsize=15)
        plt.show()

    # Pairplot to visualize relationships
    sns.pairplot(data.sample(min(1000, len(data))), diag_kind="kde")  # Limit to 1000 samples for efficiency
    plt.suptitle("Pairplot of Features", fontsize=20, y=1.02)
    plt.show()

#Example of usage:
# def main() -> None:
#     data_path = Path("data.csv")  # Example path to data
#     data = load_data(data_path)
#     print(f"Loaded {len(data)} rows of data")
#     clean_data = prepare_data(data)
#     print(f"Cleaned data contains {len(clean_data)} rows")
#     visualize_data(clean_data)

# if __name__ == "__main__":
#     main()
