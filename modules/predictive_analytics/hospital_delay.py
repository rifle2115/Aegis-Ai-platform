import streamlit as st
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Cache the model training
@st.cache_resource(show_spinner="Training LightGBM Regressor...")
def get_trained_model_and_mappings():
    # --- Load Dataset ---
    file_path = 'data/predictive_analytics/hospital_delay/COVID-19-Hospitals-Treatment-Plan.csv'
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Data file {file_path} not found. Please ensure it exists.")
        st.stop()

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
    days_to_stay_mapping = {v: k for k, v in stay_to_days_mapping.items()}
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
    
    return model, categorical_cols, stay_to_days_mapping, days_to_stay_mapping

# --- Cost Schedule ---
cost_schedule = {
    ('gynecology', 'Emergency'): 8000, ('gynecology', 'Trauma'): 7000, ('gynecology', 'Urgent'): 6500,
    ('radiotherapy', 'Emergency'): 15000, ('radiotherapy', 'Trauma'): 12000, ('radiotherapy', 'Urgent'): 10000,
    ('anesthesia', 'Emergency'): 11000, ('anesthesia', 'Trauma'): 9500, ('anesthesia', 'Urgent'): 8500,
    ('TB & Chest disease', 'Emergency'): 9000, ('TB & Chest disease', 'Trauma'): 7500, ('TB & Chest disease', 'Urgent'): 7000,
    ('surgery', 'Emergency'): 13000, ('surgery', 'Trauma'): 11000, ('surgery', 'Urgent'): 9000
}
administrative_day_rate = 2500

def calculate_administrative_delay_score(medical_cost, severity, admission_type, age):
    delay_score = 0
    if medical_cost > 200000: delay_score += 5
    elif medical_cost > 125000: delay_score += 3
    elif medical_cost > 75000: delay_score += 1
    if severity == 'Extreme': delay_score += 3
    elif severity == 'Moderate': delay_score += 1
    if admission_type == 'Emergency': delay_score += 2
    elif admission_type == 'Trauma': delay_score += 1
    if age in ['61-70','71-80','81-90','91-100']: delay_score += 1
    return delay_score

st.title("🏥 COVID-19 Hospital Treatment Plan & Delay Predictor")
st.write("Enter patient details below to predict hospital stay duration and total cost.")

# Load model
model, categorical_cols, stay_to_days_mapping, days_to_stay_mapping = get_trained_model_and_mappings()

# --- Form UI ---
with st.form("patient_form"):
    col1, col2 = st.columns(2)
    with col1:
        department = st.selectbox("Select Department", ['gynecology', 'radiotherapy', 'anesthesia', 'TB & Chest disease', 'surgery'])
        admission = st.selectbox("Select Admission Type", ['Emergency', 'Trauma', 'Urgent'])
        severity = st.selectbox("Select Illness Severity", ['Minor', 'Moderate', 'Extreme'])
    with col2:
        age = st.selectbox("Select Age Bracket", ['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100'])
        visitors = st.number_input("Enter Number of Visitors", min_value=0, value=2)
        admission_deposit = st.number_input("Enter Admission Deposit amount (₹)", min_value=0.0, value=5000.0)
    
    submit = st.form_submit_button("Predict Stay & Cost")

if submit:
    new_patient_data = {
        'Hospital': [8], 'Hospital_type': [2], 'Hospital_city': [3], 'Hospital_region': [2],
        'Available_Extra_Rooms_in_Hospital': [3], 'Department': [department], 'Ward_Type': ['R'],
        'Ward_Facility': ['F'], 'Bed_Grade': [2.0], 'City_Code_Patient': [7.0],
        'Type_of_Admission': [admission], 'Illness_Severity': [severity],
        'Patient_Visitors': [visitors], 'Age': [age], 'Admission_Deposit': [admission_deposit]
    }
    new_patient_df = pd.DataFrame(new_patient_data)
    
    # Convert categorical fields
    for col in categorical_cols:
        new_patient_df[col] = new_patient_df[col].astype('category')

    # Predict stay days
    predicted_days = model.predict(new_patient_df)[0]
    
    # Map to nearest bin
    closest = min(stay_to_days_mapping.values(), key=lambda x: abs(x - predicted_days))
    predicted_category = days_to_stay_mapping[closest]

    # Calculate medical cost
    lookup_key = (department, admission)
    daily_rate = cost_schedule.get(lookup_key, 6000)
    medical_cost = closest * daily_rate

    # Administrative delay
    admin_delay_days = calculate_administrative_delay_score(medical_cost, severity, admission, age)
    administrative_cost = admin_delay_days * administrative_day_rate

    # Totals
    total_stay_days = closest + admin_delay_days
    total_final_cost = medical_cost + administrative_cost

    # Display Report
    st.divider()
    st.header("📋 Final Patient Stay and Cost Report")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Medical Estimate")
        st.write(f"**Predicted Stay Category:** {predicted_category} ({closest} days)")
        st.write(f"**Estimated Daily Rate:** ₹{daily_rate:,.2f}")
        st.write(f"**Estimated Medical Cost:** ₹{medical_cost:,.2f}")
        
    with col_b:
        st.subheader("Administrative Delay")
        st.write(f"**Admin Delay Days:** {admin_delay_days}")
        st.write(f"**Administrative Cost:** ₹{administrative_cost:,.2f}")
        
    st.info(f"### FINAL SUMMARY\n**Total Estimated Stay:** {total_stay_days} days\n\n**Total Estimated Cost:** ₹{total_final_cost:,.2f}")
