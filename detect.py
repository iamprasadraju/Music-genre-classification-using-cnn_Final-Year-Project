from keras.models import load_model
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained model without loading optimizer weights
model = load_model('model.h5')

# Compile the model with the same optimizer and settings
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Function to preprocess audio file
def preprocess_audio(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file, mono=True, duration=30)  # Load 30 seconds of audio
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    # Compute the target length for padding or trimming
    target_length = 128
    # Determine whether to pad or trim the features
    if mfccs.shape[1] >= target_length:
        # Trim the features if longer than the target length
        mfccs = mfccs[:, :target_length]
    else:
        # Pad the features if shorter than the target length
        pad_width = target_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    # Reshape for model input
    reshaped_mfccs = mfccs.reshape(1, 20, target_length, 1)
    return reshaped_mfccs

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

# Path to the audio file you want to classify
audio_file_path = 'sunflower-street-drumloop-85bpm-163900.mp3'
# Preprocess the audio file
preprocessed_audio = preprocess_audio(audio_file_path)

# Your code to make predictions
predictions = model.predict(preprocessed_audio)

# Get the probabilities for all genres
probabilities = predictions[0]

# Plot the probabilities
plt.figure(figsize=(10, 6))
plt.bar(range(len(probabilities)), probabilities, tick_label=[class_labels[i] for i in range(len(probabilities))])
plt.title("Predicted Probabilities for Each Genre")
plt.xlabel("Genre")
plt.ylabel("Probability")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Find the index of the maximum probability
predicted_index = np.argmax(predictions)

# Get the predicted class label using the dictionary
predicted_label = class_labels[predicted_index]

print("Predicted label:", predicted_label)
