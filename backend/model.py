import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.chdir(r"C:\Users\sachd\OneDrive\Desktop\Final_Project")
# 1. Load the dataset
df = pd.read_csv("audio_Data.csv")



# 2. Drop non-numeric/non-feature column
df = df.drop(columns=['name'])

# 3. Separate features and target
X = df.drop(columns=['status'])
y = df['status']

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train a classifier (Random Forest)
clf = RandomForestClassifier(random_state=42, max_depth=5)

clf.fit(X_train_scaled, y_train)

# 7. Make predictions
y_pred = clf.predict(X_test_scaled)

# 8. Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# 9. Feature Importance Plot
importances = clf.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=feature_names[indices])
plt.title('Feature Importances')
plt.tight_layout()
plt.show()
# Save features after training
joblib.dump(clf, 'parkinson_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'feature_names.pkl')

from sklearn.metrics import roc_curve, roc_auc_score

# Predict probabilities
y_prob = clf.predict_proba(X_test_scaled)[:, 1]

# Compute ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

# Plot ROC
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Accuracy comparison plot
train_accuracy = accuracy_score(y_train, clf.predict(X_train_scaled))
test_accuracy = accuracy_score(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.barplot(x=['Train Accuracy', 'Test Accuracy'], y=[train_accuracy, test_accuracy], palette='viridis')
plt.ylim(0, 1)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()

