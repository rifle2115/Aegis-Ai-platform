"""
One-time script to train the LightGBM model locally and save it as a .pkl file.
Run this once: python train_hospital_model.py
"""
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import joblib
import os

# --- Load Dataset ---
file_path = 'data/predictive_analytics/hospital_delay/COVID-19-Hospitals-Treatment-Plan.csv'
df = pd.read_csv(file_path)

# --- Data Cleaning ---
df = df.drop(columns=['case_id', 'patientid'], errors='ignore')
df['Bed_Grade'] = df['Bed_Grade'].fillna(df['Bed_Grade'].mode()[0])
df['City_Code_Patient'] = df['City_Code_Patient'].fillna(df['City_Code_Patient'].mode()[0])

# --- Stay mappings ---
stay_to_days_mapping = {
    '0-10': 5, '11-20': 15.5, '21-30': 25.5, '31-40': 35.5, '41-50': 45.5,
    '51-60': 55.5, '61-70': 65.5, '71-80': 75.5, '81-90': 85.5, '91-100': 95.5,
    'More than 100 Days': 110
}
df['Stay_Numeric'] = df['Stay_Days'].map(stay_to_days_mapping)

# --- Features/Target ---
X = df.drop(['Stay_Days', 'Stay_Numeric'], axis=1)
y = df['Stay_Numeric']

# Convert categorical columns to category dtype for LightGBM
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    X[col] = X[col].astype('category')

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Train LightGBM ---
model = lgb.LGBMRegressor(
    objective='regression',
    num_leaves=64,
    learning_rate=0.05,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='l1',
    categorical_feature=categorical_cols,
)

# --- Save model and metadata ---
output_dir = 'models'
os.makedirs(output_dir, exist_ok=True)

artifact = {
    'model': model,
    'categorical_cols': categorical_cols,
    'stay_to_days_mapping': stay_to_days_mapping,
    'days_to_stay_mapping': {v: k for k, v in stay_to_days_mapping.items()},
}

output_path = os.path.join(output_dir, 'hospital_lgbm.pkl')
joblib.dump(artifact, output_path)
print(f"Model saved to {output_path}")
print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")
