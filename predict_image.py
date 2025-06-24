from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

model = load_model("hematovision_model.h5")

img_path = "data/test/monocyte/your_image.jpg"  # Change path
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
classes = ['eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']
print("Predicted class:", classes[np.argmax(pred)])
