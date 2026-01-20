
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib, os

os.makedirs("models", exist_ok=True)

data = pd.read_csv("data/nsl_kdd_processed.csv")
X = data.drop("label", axis=1)
y = data["label"]

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)
joblib.dump(model, "models/static_model.pkl")

print("Static ML Firewall trained successfully")
