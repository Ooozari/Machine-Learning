# Lab4_Task2_SVM_BrainTumorDetection_NoCV2.py

import kagglehub
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==============================
# Step 1: Download and Load Dataset
# ==============================
path = kagglehub.dataset_download("navoneel/brain-mri-images-for-brain-tumor-detection")
print("Path to dataset files:", path)

categories = ["yes", "no"]
img_size = 64  # Resize to 64x64

X = []
y = []

# ==============================
# Step 2: Preprocess Images
# ==============================
for label, category in enumerate(categories):
    folder = os.path.join(path, category)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        try:
            img = Image.open(img_path).convert("L")  # Convert to grayscale
            img = img.resize((img_size, img_size))   # Resize
            X.append(np.array(img).flatten())        # Flatten
            y.append(label)
        except Exception as e:
            pass  # skip unreadable files

X = np.array(X)
y = np.array(y)

print(f"Total samples: {len(X)}")
print(f"Feature shape: {X.shape}")

# ==============================
# Step 3: Split Data
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# Step 4: Normalize Features
# ==============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# Step 5: Train SVM Classifier
# ==============================
svm_clf = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
svm_clf.fit(X_train, y_train)

# ==============================
# Step 6: Evaluate Model
# ==============================
y_pred = svm_clf.predict(X_test)

print("\nModel Evaluation Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=categories))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ==============================
# Step 7: Visualize Predictions
# ==============================
plt.figure(figsize=(10, 6))
for i in range(6):
    idx = np.random.randint(0, len(X_test))
    img = X_test[idx].reshape(img_size, img_size)
    plt.subplot(2, 3, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Actual: {categories[y_test[idx]]}\nPredicted: {categories[y_pred[idx]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
