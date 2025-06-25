from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

model = load_model("hematovision_model.h5")

img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.  # Ensure scaling matches training preprocessing

pred = model.predict(img_array)
classes = ['Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']
predicted_class = classes[np.argmax(pred)]
