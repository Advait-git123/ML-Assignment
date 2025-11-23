import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

print("--- Starting Assignment Code ---")

# 1. Load Data
try:
    df = pd.read_csv('RailwayTicketConfirmation.csv')
    print("✅ Data Loaded Successfully!")
except FileNotFoundError:
    print("❌ Error: 'RailwayTicketConfirmation.csv' not found.")
    print("   Make sure the CSV is in the same folder as this script.")
    exit()

# 2. Feature Engineering
# Convert Target
target_map = {'Confirmed': 1, 'Not Confirmed': 0}
df['target'] = df['Confirmation Status'].map(target_map)

# Calculate Days in Advance
df['Date of Journey'] = pd.to_datetime(df['Date of Journey'])
df['Booking Date'] = pd.to_datetime(df['Booking Date'])
df['Days_Advance'] = (df['Date of Journey'] - df['Booking Date']).dt.days

# Drop columns (including leakers)
cols_to_drop = [
    'PNR Number', 'Train Number', 'Source Station', 'Destination Station', 
    'Date of Journey', 'Booking Date', 'Current Status', 'Waitlist Position', 
    'Confirmation Status', 'target'
]

X = df.drop(columns=cols_to_drop)
y = df['target']

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# 3. Setup Pipelines
print("   Setting up models...")
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_cols),
        ('cat', cat_transformer, categorical_cols)
    ])

# Define Models
baseline_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

sophisticated_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
])

# 4. Evaluation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate(model, name):
    print(f"   Running evaluation for {name} (this may take a moment)...")
    acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()
    f1 = cross_val_score(model, X, y, cv=cv, scoring='f1').mean()
    roc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc').mean()
    return [acc, f1, roc]

# Run tests
scores_base = evaluate(baseline_model, "Logistic Regression")
scores_soph = evaluate(sophisticated_model, "Random Forest")

# 5. Output Results
results_df = pd.DataFrame({
    'Metric': ['Accuracy', 'F1 Score', 'ROC-AUC'],
    'Logistic Regression': scores_base,
    'Random Forest': scores_soph
})

print("\n" + "="*30)
print("FINAL RESULTS TABLE")
print("="*30)
print(results_df)
print("="*30)

# 6. Save Plot
print("   Generating plot...")
results_df.set_index('Metric').plot(kind='bar', figsize=(10, 6))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--')
plt.xticks(rotation=0)
plt.tight_layout()

# SAVE the plot to a file
plt.savefig('result_graph.png')
print("✅ Graph saved as 'result_graph.png'. Open this file to see the chart.")
print("--- Done ---")