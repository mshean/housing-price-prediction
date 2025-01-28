import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    """Load the dataset and display basic info."""
    data = pd.read_csv(filepath)
    print("First 5 rows:")
    print(data.head())
    print("\nDataset info:")
    print(data.info())
    print("\nSummary statistics:")
    print(data.describe())
    return data

# Step 2: Clean and Explore Data
def clean_and_explore(data):
    """Clean the dataset and explore relationships."""
    # Handle missing values
    data['total_bedrooms'] = data['total_bedrooms'].fillna(data['total_bedrooms'].median())

    # Remove duplicates
    data = data.drop_duplicates()

    # Remove capped house prices ($500,000)
    data = data[data['median_house_value'] < 500000]

    # One-hot encode categorical data
    data = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)

    # Visualize house price distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data['median_house_value'], bins=50, kde=True, color='green')
    plt.title('House Prices (After Removing Capped Values)')
    plt.xlabel('House Prices')
    plt.ylabel('Frequency')
    plt.show()

    # Visualize median income distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data['median_income'], bins=30, kde=True, color='purple')
    plt.title('Median Income Distribution (Scaled)')
    plt.xlabel('Median Income (0-15 scale)')
    plt.ylabel('Frequency')
    plt.show()

    # Scatter plot: Income vs. House Prices
    plt.figure(figsize=(10, 6))
    sns.regplot(
        x='median_income', 
        y='median_house_value', 
        data=data, 
        scatter_kws={'alpha': 0.3, 'color': 'blue'}, 
        line_kws={'color': 'red'}
    )
    plt.title('Income vs. House Prices (No Capped Values)')
    plt.xlabel('Median Income (scaled)')
    plt.ylabel('House Prices')
    plt.grid(True)
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()

    return data

# Main Function
def main():
    # Step 1: Load data
    filepath = os.path.join(os.path.expanduser("~"), "Downloads", "housing.csv")
    data = load_data(filepath)

    # Step 2: Clean and explore data
    data = clean_and_explore(data)

    # Save the cleaned dataset
    cleaned_filepath = os.path.join(os.path.expanduser("~"), "Downloads", "cleaned_housing.csv")
    data.to_csv(cleaned_filepath, index=False)

if __name__ == "__main__":
    main()