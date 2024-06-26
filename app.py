import streamlit as st
import librosa
import numpy as np
from keras.models import load_model

# Load the pre-trained model without loading optimizer weights
model = load_model('model.h5', compile=False)

# Compile the model with the same optimizer and settings
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Dictionary mapping indices to class labels
class_labels = {
    0: "disco",
    1: "metal",
    2: "reggae",
    3: "blues",
    4: "rock",
    5: "classical",
    6: "jazz",
    7: "hiphop",
    8: "country",
    9: "pop"
}

# Function to preprocess audio file
def preprocess_audio(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file, mono=True, duration=30)  # Load 30 seconds of audio
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    # Compute the target length for padding
    target_length = 128
    # Resize MFCCs to the target length
    resized_mfccs = np.zeros((20, target_length))
    if mfccs.shape[1] >= target_length:
        resized_mfccs = mfccs[:, :target_length]
    else:
        resized_mfccs[:, :mfccs.shape[1]] = mfccs
    # Reshape for model input
    reshaped_mfccs = resized_mfccs.reshape(1, 20, target_length, 1)
    return reshaped_mfccs

# Function to make predictions
def predict_genre(audio_file):
    preprocessed_audio = preprocess_audio(audio_file)
    predictions = model.predict(preprocessed_audio)
    predicted_index = np.argmax(predictions)
    predicted_label = class_labels[predicted_index]
    return predicted_label

# Custom CSS for white background
st.markdown(
    """
    <style>
    body {
        background-color: #ffffff; /* white */
        color: black; /* black text for better readability */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Music Genre Classifier")

uploaded_file = st.file_uploader("Upload Audio File", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    if st.button("Predict Genre"):
        predicted_genre = predict_genre(uploaded_file)
        st.success(f"Predicted Genre: {predicted_genre}")
