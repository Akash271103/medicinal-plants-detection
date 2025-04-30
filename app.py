import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
MODEL_PATH = 'model/plant_model.h5'
UPLOAD_FOLDER = 'static/uploads/'
model = load_model(MODEL_PATH)

# class labels (update with your plant names)
class_labels = ['Neem', 'Tulsi', 'Aloe Vera', 'Mint', 'Ashwagandha']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"
    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Preprocess
    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    predicted_class = class_labels[np.argmax(pred)]

    return render_template('result.html', plant=predicted_class, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
