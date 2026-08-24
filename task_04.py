import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import make_classification

# 1. Dataset Simulation / Hand Gesture Landmark Features
# Simulating 10 gesture classes (e.g., Palm, Fist, Thumbs Up, Peace, etc.)
# 63 features correspond to 21 hand landmarks in 3D space (x, y, z)
X, y = make_classification(
    n_samples=2000,
    n_features=63,
    n_informative=45,
    n_classes=10,
    random_state=42
)

# 2. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train Gesture Classification Model (Random Forest Classifier)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 4. Evaluate Model Performance
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Hand Gesture Classification Accuracy: {accuracy * 100:.2f}%\n")
print("--- Classification Report ---")
gesture_names = [f"Gesture_{i}" for i in range(10)]
print(classification_report(y_test, y_pred, target_names=gesture_names))
