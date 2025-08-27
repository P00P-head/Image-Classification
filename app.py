import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

MODEL_PATH = 'models/model_0.732.h5'
model = load_model(MODEL_PATH)


class_labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

st.title('Image Classification App')
st.write('Upload an image and the model will predict its class.')
st.write('*NOTE* : Keep labels within [ airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck])

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write('')
    st.write('Classifying...')

    img = image.resize((32, 32))
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)


    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    st.write(f'Prediction: **{predicted_class}**')
