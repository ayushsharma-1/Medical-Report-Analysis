from flask import Flask, render_template, request, redirect, url_for
from forms import InputForm
import os
import gdown
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Initialize the Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = "secret_key"

# Directory to save uploaded images in the static folder
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Google Drive file ID of your model.h5
DRIVE_FILE_ID = "1nDnSlx3yiJ0zKQ5B2VpBaw_j0uy-Y18A"
MODEL_PATH = "./model.h5"

# Function to download model if not present
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={DRIVE_FILE_ID}", MODEL_PATH, quiet=False)

# Ensure the upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Download and load the model
download_model()
print("Loading model...")
model = load_model(MODEL_PATH)

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
@app.route('/home')
def home():
    return render_template("home.html", title="Home")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = InputForm()
    if form.validate_on_submit():
        if 'image' not in request.files:
            return redirect(request.url)
        
        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)
        
        if file:
            # Save the image to the static folder
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            # Open and preprocess the image
            image = Image.open(filename)
            image = image.resize((150, 150))  # Resize the image
            image = np.array(image)

            # Convert grayscale to RGB if needed
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)

            # Normalize the image
            image = image / 255.0

            # Add a batch dimension
            image = np.expand_dims(image, axis=0)

            # Make the prediction
            prediction = model.predict(image)
            labels = ["COVID-19", "Pneumonia", "Normal", "Tuberculosis"]
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_class = labels[predicted_class_index]

            return render_template('prediction_result.html', prediction=predicted_class, image_url=url_for('static', filename=f'uploads/{file.filename}'))
    
    return render_template('predict.html', title="Predict", form=form)

if __name__ == "__main__":
    app.run(debug=True)