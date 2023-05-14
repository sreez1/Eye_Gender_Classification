import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Define path to train folder
train_path = "eye_morphology/train/"

# Define categories
categories = ["female", "male"]

# Define image size
img_size = 64

# Load images and labels from train folder
data = []
for category in categories:
    path = os.path.join(train_path, category)
    class_num = categories.index(category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        img_array = cv2.resize(img_array, (img_size, img_size))
        data.append([img_array, class_num])

# Shuffle the data
np.random.shuffle(data)

# Split the data into features and labels
X = []
y = []
for features, label in data:
    X.append(features)
    y.append(label)

# Convert the data into numpy arrays
X = np.array(X).reshape(-1, img_size*img_size)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the SVM model
svm_model = SVC()

# Train the SVM model
svm_model.fit(X_train, y_train)

# Evaluate the SVM model
svm_pred = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)

# Print the accuracy of the SVM model
print("SVM Accuracy:", svm_acc)
print(svm_pred)

# Save the trained SVM model to a joblib file
joblib.dump(svm_model, "eye_morphology_svm_model.joblib")

