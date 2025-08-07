import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from mtcnn import MTCNN

# Constants
RAF_DB_EMOTIONS = [
    "Surprise",
    "Fear",
    "Disgust",
    "Happiness",
    "Sadness",
    "Anger",
    "Neutral",
]


@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)


@st.cache_resource
def load_face_detector():
    return MTCNN()


def preprocess_image(image, target_size=(100, 100)):
    # Convert PIL to numpy
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Handle different image formats
    if len(image.shape) == 2:  # Grayscale
        image = np.stack([image, image, image], axis=-1)
    elif len(image.shape) == 3:
        if image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]
        elif image.shape[2] == 1:  # Single channel
            image = np.concatenate([image, image, image], axis=-1)

    # Resize to target size
    image = cv2.resize(image, target_size)

    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image


def detect_face(image, detector):
    # Convert PIL to numpy
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Ensure RGB format for MTCNN
    if len(image.shape) == 2:
        rgb_image = np.stack([image, image, image], axis=-1)
    elif len(image.shape) == 3:
        if image.shape[2] == 3:
            rgb_image = image.copy()
        elif image.shape[2] == 4:
            rgb_image = image[:, :, :3]
        else:
            return None
    else:
        return None

    try:
        # Detect faces
        results = detector.detect_faces(rgb_image)

        if results:
            # Get the largest face
            largest_face = max(results, key=lambda x: x["box"][2] * x["box"][3])
            x, y, w, h = largest_face["box"]

            # Add padding
            padding = int(max(w, h) * 0.2)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(rgb_image.shape[1], x + w + padding)
            y2 = min(rgb_image.shape[0], y + h + padding)

            # Extract face
            face = rgb_image[y1:y2, x1:x2]
            return face

        return None

    except Exception as e:
        st.error(f"Face detection error: {e}")
        return None


def predict_emotion(model, image):
    try:
        predictions = model.predict(image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        return RAF_DB_EMOTIONS[predicted_class], confidence, predictions[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, 0, None


# Main app
st.title("🎭 Emotion Detection")

# Sidebar
st.sidebar.header("Model Configuration")
model_file = st.sidebar.file_uploader("Upload .keras model", type=["keras"])

if model_file:
    # Save and load model
    with open("temp_model.keras", "wb") as f:
        f.write(model_file.getbuffer())

    try:
        model = load_model("temp_model.keras")
        st.sidebar.success("✅ Model loaded!")
        st.sidebar.info(f"Input shape: {model.input_shape}")

        # Extract target size from model input shape
        if len(model.input_shape) == 4:  # (batch, height, width, channels)
            target_h, target_w = model.input_shape[1], model.input_shape[2]
            target_size = (target_w, target_h)  # cv2.resize expects (width, height)
        else:
            target_size = (100, 100)  # Default fallback

        st.sidebar.info(f"Target size: {target_size}")

        # Detection settings
        st.sidebar.subheader("Settings")
        use_face_detection = st.sidebar.checkbox("Use face detection", value=True)

        # Main content
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            help="Supports both RGB and grayscale images",
        )

        if uploaded_file:
            image = Image.open(uploaded_file)

            # Display image info
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(image, caption="Input Image", use_column_width=True)
            with col2:
                st.write("**Image Info:**")
                st.write(f"Size: {image.size}")
                st.write(f"Mode: {image.mode}")
                if hasattr(image, "format"):
                    st.write(f"Format: {image.format}")

            if st.button(
                "🔍 Predict Emotion", type="primary", use_container_width=True
            ):
                with st.spinner("Processing..."):
                    processed_image = None
                    face_detected = False

                    if use_face_detection:
                        # Try face detection
                        detector = load_face_detector()
                        face = detect_face(image, detector)

                        if face is not None:
                            st.success("✅ Face detected!")

                            # Show detected face
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                st.image(
                                    face, caption="Detected Face", use_column_width=True
                                )

                            processed_image = preprocess_image(face, target_size)
                            face_detected = True
                        else:
                            st.warning("⚠️ No face detected, using full image")

                    # If no face detected or face detection disabled, use full image
                    if processed_image is None:
                        processed_image = preprocess_image(image, target_size)

                    # Predict emotion
                    emotion, confidence, all_preds = predict_emotion(
                        model, processed_image
                    )

                    if emotion is not None:
                        # Display results
                        st.header("🎯 Results")

                        # Main prediction
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.metric("🎭 Predicted Emotion", emotion)
                            st.metric("📊 Confidence", f"{confidence:.1%}")

                        with col2:
                            # Top 3 emotions
                            sorted_emotions = sorted(
                                zip(RAF_DB_EMOTIONS, all_preds),
                                key=lambda x: x[1],
                                reverse=True,
                            )[:3]

                            st.write("**Top 3 Predictions:**")
                            for i, (emo, prob) in enumerate(sorted_emotions):
                                st.write(f"{i+1}. {emo}: {prob:.2%}")

                        # All emotions with progress bars
                        st.subheader("All Emotion Probabilities")
                        for emo, prob in zip(RAF_DB_EMOTIONS, all_preds):
                            st.progress(float(prob), text=f"{emo}: {prob:.3f}")

                    else:
                        st.error("❌ Prediction failed")

    except Exception as e:
        st.sidebar.error(f"❌ Model loading failed: {e}")
        st.error("Please ensure you uploaded a valid .keras model file")

else:
    st.sidebar.warning("Please upload a .keras model file")
    st.info("👈 Upload your trained emotion detection model in the sidebar to begin")

# Footer
st.markdown("---")
st.markdown("*Emotion Detection using CNN + MTCNN | Built with Streamlit*")
