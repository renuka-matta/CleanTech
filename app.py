from flask import Flask, render_template, request
import tensorflow as tf
from keras.utils import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
model = tf.keras.models.load_model("hematovision_model.keras")
class_labels = ["Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img = load_img(filepath, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    preds = model.predict(img_array)
    result = class_labels[np.argmax(preds)]

    return render_template("result.html", prediction=result, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
