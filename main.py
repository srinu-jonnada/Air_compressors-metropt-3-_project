import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load the dataset
df = pd.read_csv('MetroPT3(AirCompressor).csv')

# Display the first few rows of the dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Convert 'Unnamed: 0' to datetime if it's a timestamp (assuming it is a timestamp column)
df['timestamp'] = pd.to_datetime(df['Unnamed: 0'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month

# Drop the original timestamp column and the index column if not needed
df = df.drop(['Unnamed: 0'], axis=1)

# Handle infinite values by converting them to NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Check for any remaining NaN values
print(df.isnull().sum())

# Drop rows with NaN values (or you could fill them with a specific value if more appropriate)
df = df.dropna()

# Drop the 'timestamp' column as it's not needed for model training
df = df.drop(['timestamp'], axis=1)

# Display the first few rows of the updated dataframe
print(df.head())

# Step 3: Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 6))
sns.histplot(df['DV_pressure'], bins=30, kde=True)
plt.title('Distribution of DV_pressure')
plt.xlabel('DV_pressure')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='COMP', y='DV_pressure', data=df)
plt.title('DV_pressure by COMP Status')
plt.xlabel('COMP Status')
plt.ylabel('DV_pressure')
plt.show()

# Step 4: Build a Predictive Model
X = df.drop('COMP', axis=1)
y = df['COMP']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

importance = model.feature_importances_
feature_importance = pd.Series(importance, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feature_importance.plot(kind='bar')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()
