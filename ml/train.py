import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# 1. Load Data
print("Loading Iris dataset...")
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 2. Split Data (Train & Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Model
print("Training RandomForest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Test Model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 5. Save Model
os.makedirs("ml/saved_model", exist_ok=True)
model_path = "ml/saved_model/model.joblib"
joblib.dump(model, model_path)
print(f"Model successfully saved to {model_path} ✅")