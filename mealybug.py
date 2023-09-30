import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

# Load the pre-trained deep learning model
model_path = r'best_model.h5'
model = keras.models.load_model(model_path)

# Define a function to preprocess the user-uploaded image
def preprocess_image(image):
    # Resize the image to match the input size of your model
    image = image.resize((150, 150))
    # Convert the PIL image to a NumPy array
    image_array = np.array(image)
    # Normalize pixel values (assuming the model expects values between 0 and 1)
    image_array = image_array / 255.0
    # Expand the dimensions to match the input shape of your model
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Create a Streamlit web app
st.title('Mealybug Classifier')

# Create an upload button for the user to upload an image
uploaded_image = st.file_uploader('Upload an image', type=['jpg', 'png', 'jpeg'])

# Check if an image has been uploaded
if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the uploaded image
    processed_image = preprocess_image(Image.open(uploaded_image))

    # Make predictions using the loaded model
    prediction = model.predict(processed_image)

    # Display the results
    st.subheader('Prediction:')
    if prediction[0][0] > 0.5:
        st.write('This is a mealybug.')
    else:
        st.write('This is not a mealybug.')
