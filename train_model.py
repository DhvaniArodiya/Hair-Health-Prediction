import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = ("Predict Hair Fall.csv")
df = pd.read_csv(file_path)

# Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("&", "and")

# List of categorical columns to encode
categorical_columns = ["genetics", "hormonal_changes", "stress", "poor_hair_care_habits",
                       "environmental_factors", "smoking", "weight_loss",
                       "medical_conditions", "medications_and_treatments", "nutritional_deficiencies"]

# Apply Label Encoding
label_encoders = {}
for col in categorical_columns:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Define features and target variable
X = df.drop(columns=["id", "hair_loss"], errors="ignore")
y = df["hair_loss"]

# Convert to numeric and handle NaNs
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "hair_model.pkl")

print("Model trained and saved successfully as 'hair_model.pkl'!")
