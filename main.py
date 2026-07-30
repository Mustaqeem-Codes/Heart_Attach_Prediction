import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

def plot_distributions(data, columns, stage_title):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    fig.suptitle(f"{stage_title} Data Distributions", fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for i, col in enumerate(columns):
        if col in data.columns:
            sns.histplot(data[col], kde=True, ax=axes[i], color='teal')
            axes[i].set_title(f"Distribution of {col}")
            axes[i].set_ylabel('')
    
    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.show()

def prepare_heart_pipeline(filepath='heart.csv'):
    df = pd.read_csv(filepath)
    
    print("--- STEP 1: INITIAL DATA INSPECTION ---")
    print(f"Original Data Shape: {df.shape}")
    print("Null Values Check:")
    print(df.isna().sum())
    
    numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    
    print("\n--- STEP 2: BEFORE VISUALIZATION ---")
    plot_distributions(df, numeric_cols, "Before Pipeline")
    
    print("\n--- STEP 3: DATA CLEANING & IMPUTATION ---")
    print(f"Zero values in Cholesterol before: {(df['Cholesterol'] == 0).sum()}")
    chol_mean = df[df['Cholesterol'] != 0]['Cholesterol'].mean()
    df['Cholesterol'] = df['Cholesterol'].replace(0, chol_mean).round(2)
    print(f"Zero values in Cholesterol after: {(df['Cholesterol'] == 0).sum()}")
    
    print("\n--- STEP 4: FEATURE ENGINEERING ---")
    df_encoded = pd.get_dummies(df, drop_first=True)
    bool_cols = df_encoded.select_dtypes(include=['bool']).columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
    print(f"Encoded DataFrame Shape: {df_encoded.shape}")
    print("Sample of Encoded Data:")
    print(df_encoded.sample(3))
    
    print("\n--- STEP 5: DATA SPLITTING ---")
    X = df_encoded.drop('HeartDisease', axis=1)
    y = df_encoded['HeartDisease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training Features Shape: {X_train.shape}")
    print(f"Testing Features Shape: {X_test.shape}")
    
    print("\n--- STEP 6: FEATURE SCALING ---")
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    print("StandardScaler fitted on training data and applied to both train and test sets.")
    
    print("\n--- STEP 7: AFTER VISUALIZATION ---")
    plot_distributions(X_train, numeric_cols, "After Pipeline (Scaled Training Data)")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_heart_pipeline('heart.csv')