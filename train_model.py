import os
import numpy as np
import librosa
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Extract MFCC features from audio file
def extract_features(file_path, n_mfcc=13):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Paths to your dataset
real_path = 'KAGGLE/AUDIO/REAL'
fake_path = 'KAGGLE/AUDIO/FAKE'

X = []
y = []

# Load REAL audio files
for file in os.listdir(real_path):
    if file.endswith(".wav"):
        features = extract_features(os.path.join(real_path, file))
        if features is not None:
            X.append(features)
            y.append(0)  # Label for real

# Load FAKE audio files
for file in os.listdir(fake_path):
    if file.endswith(".wav"):
        features = extract_features(os.path.join(fake_path, file))
        if features is not None:
            X.append(features)
            y.append(1)  # Label for fake

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train SVM model
model = SVC(kernel='linear', probability=True)
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Training complete. Files saved: svm_model.pkl, scaler.pkl")
