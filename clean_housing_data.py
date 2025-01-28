import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Construct the path to the Downloads folder
downloads_path = os.path.join(os.path.expanduser("~"), "Downloads", "housing.csv")

# Read the CSV file from the Downloads folder
data = pd.read_csv(downloads_path)

# Fill missing values in 'total_bedrooms' with the median
data['total_bedrooms'] = data['total_bedrooms'].fillna(data['total_bedrooms'].median())

# Remove duplicates
data = data.drop_duplicates()

# Check how many rows are capped
print("\nHouses priced at $500k:", data[data['median_house_value'] == 500000].shape[0])

# Remove capped values
data = data[data['median_house_value'] < 500000]

# Plot updated distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['median_house_value'], bins=50, kde=True, color='green')
plt.title('House Prices (After Removing Capped Values)')
plt.xlabel('House Prices')
plt.ylabel('Frequency')
plt.show()

# Check categories
print("\nOcean proximity categories:")
print(data['ocean_proximity'].value_counts())

# One-hot encode categorical column
data = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)

# Visualize house price distribution
sns.histplot(data['median_house_value'], bins=50, kde=True)
plt.title('Distribution of House Prices')
plt.show()

# Visualize median income distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['median_income'], bins=30, kde=True, color='purple')
plt.title('Median Income Distribution (Scaled)')
plt.xlabel('Median Income (0-15 scale)')
plt.ylabel('Frequency')
plt.show()

# Scatter plot: Median income vs. House prices
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

# Correlation matrix
plt.figure(figsize=(12, 8))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Cap house prices at the 95th percentile
upper_limit = data['median_house_value'].quantile(0.95)
data['median_house_value'] = data['median_house_value'].clip(upper=upper_limit)

# Rooms per household
data['rooms_per_household'] = data['total_rooms'] / data['households']

# Bedrooms per room
data['bedrooms_per_room'] = data['total_bedrooms'] / data['total_rooms']

# Population per household
data['population_per_household'] = data['population'] / data['households']

# Save the cleaned dataset
data.to_csv('cleaned_housing.csv', index=False)