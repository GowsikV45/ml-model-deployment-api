import joblib
import pandas as pd

# 1. Load the saved model
model_path = "ml/saved_model/model.joblib"
print(f"Loading model from {model_path}...")
model = joblib.load(model_path)

# 2. Create dummy data (Iris பூவின் அளவுகள்)
sample_data = pd.DataFrame([{
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2
}])

# 3. Predict using the loaded model
prediction = model.predict(sample_data)

print(f"Prediction Output: {prediction[0]}")
print("(0 = Setosa, 1 = Versicolor, 2 = Virginica)")
print("✅ Model Loading and Prediction is working perfectly!")