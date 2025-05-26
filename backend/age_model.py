import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

print(os.getcwd())


df = pd.read_csv("age_data.csv")

X = df[["Age"]]
y = df["PD"]

age_model = LogisticRegression()
age_model.fit(X, y)

# Save the model
joblib.dump(age_model, "age_model.pkl")
