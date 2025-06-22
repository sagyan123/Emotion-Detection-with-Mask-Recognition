import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

# Load your trained models
@st.cache_resource
def load_emotion_model():
    try:
        model = tf.keras.models.load_model('ok.h5')  
        return model
    except Exception as e:
        st.error(f"Error loading emotion model: {e}")
        st.stop()

@st.cache_resource
def load_demographic_model():
    try:
        model = tf.keras.models.load_model('demographic_model.h5')  
        return model
    except Exception as e:
        st.error(f"Error loading demographic model: {e}")
        st.stop()

emotion_model = load_emotion_model()
demographic_model = load_demographic_model()

# Define class names
emotion_class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
demographic_class_names = ['Black', 'East Asian', 'Indian', 'Latino_Hispanic', 'Middle Eastern', 'Southeast Asian', 'White']

# Function to preprocess the image
def preprocess_image(image, target_size):
    """
    Preprocesses the input image to match the model's requirements.
    """
    if len(image.shape) == 2:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # Image with alpha channel
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    image = cv2.resize(image, target_size)  # Resize to model input size
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize pixel values
    return image

# Function to preprocess the image for demographic detection
def preprocess_image_demographic(image, target_size):
    """
    Preprocesses the input image for the demographic model, ensuring dimensions are adequate.
    """
    try:
        if len(image.shape) == 2:  # Grayscale image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # Image with alpha channel
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Resize the image to the required target size
        image = cv2.resize(image, target_size)

        # Pad the image to prevent excessive downsampling if necessary
        if target_size[0] < 64 or target_size[1] < 64:  # Example threshold
            st.warning("Padding the image to prevent dimension errors.")
            image = cv2.copyMakeBorder(image, 32, 32, 32, 32, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # Normalize pixel values
        image = np.array(image, dtype=np.float32) / 255.0
        return image
    except Exception as e:
        st.error(f"Error during image preprocessing for demographic model: {e}")
        return None

# Function to make predictions for emotion detection
def predict_emotion(image):
    """
    Predicts the emotion class of the input image using the emotion model.
    """
    try:
        image = preprocess_image(image, (256, 256))
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        predictions = emotion_model.predict(image)
        predicted_class_index = np.argmax(predictions[0])
        return emotion_class_names[predicted_class_index]
    except Exception as e:
        st.error(f"Error during emotion prediction: {e}")
        return None

# Function to make predictions for demographic detection
def predict_demographic(image):
    """
    Predicts the demographic class of the input image using the demographic model.
    """
    try:
        image = preprocess_image_demographic(image, (256, 256))
        if image is None:
            return None

        image = np.expand_dims(image, axis=0)  # Add batch dimension
        predictions = demographic_model.predict(image)
        predicted_class_index = np.argmax(predictions[0])
        return demographic_class_names[predicted_class_index]
    except Exception as e:
        st.error(f"Error during demographic prediction: {e}")
        return None

# Streamlit UI setup
st.title("Emotion and Demographic Detection ")
st.text("Detect emotions and demographics using uploaded images or webcam feeds.")

# Upload image from local device
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    image_array = np.array(image)  # Convert PIL image to numpy array

    # Buttons for predictions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Predict Emotion", key="emotion_btn"):
            predicted_emotion = predict_emotion(image_array)
            if predicted_emotion:
                st.image(image, caption="Uploaded Image", use_column_width=True)
                st.write(f"Predicted Emotion: **{predicted_emotion}**")
    with col2:
        if st.button("Predict Demographic", key="demographic_btn"):
            predicted_demographic = predict_demographic(image_array)
            if predicted_demographic:
                st.image(image, caption="Uploaded Image", use_column_width=True)
                st.write(f"Predicted Demographic: **{predicted_demographic}**")

# Start webcam feed
run_webcam = st.checkbox("Start Webcam")

if run_webcam:
    st.warning("Allow webcam access in your browser to use this feature.")
    cap = cv2.VideoCapture(0)

    stframe = st.empty()
    detect_emotion = st.checkbox("Detect Emotion from Webcam", key="detect_emotion_webcam")
    detect_demographic = st.checkbox("Detect Demographic from Webcam", key="detect_demographic_webcam")

    while run_webcam:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image from webcam.")
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Predictions on the webcam frame
        if detect_emotion:
            predicted_emotion = predict_emotion(frame_rgb)
            if predicted_emotion:
                cv2.putText(
                    frame_rgb,
                    f"Emotion: {predicted_emotion}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        if detect_demographic:
            predicted_demographic = predict_demographic(frame_rgb)
            if predicted_demographic:
                cv2.putText(
                    frame_rgb,
                    f"Demographic: {predicted_demographic}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        # Display the frame
        stframe.image(frame_rgb, channels="RGB")

    cap.release()
else:
    st.write("Check the box above to start the webcam.")
