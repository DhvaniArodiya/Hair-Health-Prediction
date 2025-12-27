from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("hair_model.pkl")

# Home page
@app.route('/')
def home():
    return render_template("index.html")  # Make sure index.html exists

# Prediction route
@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
        # Extract form data
        genetics = int(request.form["genetics"])
        hormonal_changes = int(request.form["hormonal_changes"])
        medical_conditions = int(request.form["medical_conditions"])
        medications_and_treatments = request.form.get('medications_and_treatments', '0')  
        nutritional_deficiencies = int(request.form["nutritional_deficiencies"])
        stress_mapping = {"low": 0, "moderate": 1, "high": 2}

        stress_level = request.form["stress"].lower()  # Convert input to lowercase
        stress = stress_mapping.get(stress_level, 1)  # Default to "moderate" if invalid

        age = int(request.form["age"])
        poor_hair_care_habits = int(request.form["poor_hair_care_habits"])
        environmental_factors = int(request.form["environmental_factors"])
        smoking = int(request.form["smoking"])
        weight_loss = int(request.form["weight_loss"])

        # Prepare data for model
        features = [[genetics, hormonal_changes, medical_conditions, 
                     medications_and_treatments, nutritional_deficiencies, 
                     stress, age, poor_hair_care_habits, 
                     environmental_factors, smoking, weight_loss]]
        features_df = pd.DataFrame(features)

        # Make prediction
        prediction = model.predict(features_df)
        result = "Hair Loss" if prediction[0] == 1 else "No Hair Loss"

        return render_template("result.html", prediction=result)  # Ensure result.html exists

if __name__ == "__main__":
    app.run(debug=True)
