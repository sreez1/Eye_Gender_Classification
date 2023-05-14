# import required libraries
import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from sklearn.svm import SVC
from joblib import load
# initialize Flask app
app = Flask(__name__)

# Define path to train folder
train_path = "eye_morphology/train/"

# Define categories
categories = ["female", "male"]

# Define image size
img_size = 64

# Load the trained SVM model
svm_model = SVC(kernel='linear')

svm_model = load('eye_morphology_svm_model.joblib')


# define function to preprocess the input image
def preprocess_image(image):
    # convert the image to grayscale and resize
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (img_size, img_size))
    # convert the image into a 1D numpy array
    image = image.reshape(-1, img_size * img_size)
    return image

# define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# define route for image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # get the uploaded file
    file = request.files['file']
    # save the file
    filename = secure_filename(file.filename)
    filepath = os.path.join('uploads', filename)
    file.save(filepath)
    # read the uploaded image
    image = cv2.imread(filepath)
    # preprocess the image
    image = preprocess_image(image)
    # make a prediction using the SVM model
    prediction = svm_model.predict(image)[0]
    # get the category name of the prediction
    category = categories[prediction]
    # return the prediction as a json object
    return jsonify({'result': category})

if __name__ == '__main__':
    app.run(debug=True)
