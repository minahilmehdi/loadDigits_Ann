import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your trained model (Assuming it's a TensorFlow/Keras model)
model = tf.keras.models.load_model('ann_model.h5')

# List of class names (Modify this with your actual class names if applicable)
class_names = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6", "Class 7", "Class 8", "Class 9"]  # For digits 0-9, adjust as needed

def preprocess_image(image):
    """Preprocess the image before passing it to the model."""
    try:
        image = image.convert('L')  # Convert image to grayscale
        image = image.resize((8, 8))  # Resize image to 8x8 pixels
        image = np.array(image) / 255.0   # Normalize the image to [0, 1] range
        image = image.flatten()  # Flatten the image to a 1D array of 64 features
        image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 64)
        return image
    except Exception as e:
        st.error(f"Error in image preprocessing: {e}")
        return None


def predict(image):
    """Run the model prediction on the image."""
    processed_image = preprocess_image(image)
    if processed_image is not None:
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)  # Get the class with the highest score
        return predicted_class, prediction
    else:
        return None, None

# Streamlit UI
st.title("Digit Image Classification App")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Classifying...")

        # Run the model prediction
        predicted_class, prediction = predict(image)
        
        if predicted_class is not None:
            st.write(f"Prediction: {class_names[predicted_class[0]]} ({prediction[0][predicted_class[0]]*100:.2f}%)")
        else:
            st.write("Prediction failed. Please try again.")
    except Exception as e:
        st.error(f"Error loading image: {e}")
