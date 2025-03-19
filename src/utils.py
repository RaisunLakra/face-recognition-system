import pickle
import os

def load_encodings(encodings_file="face_encodings.pkl"):
    """Load known face encodings from a file."""
    if not os.path.exists(encodings_file):
        raise FileNotFoundError(f"Encodings file not found: {encodings_file}")

    with open(encodings_file, "rb") as f:
        data = pickle.load(f)
    return data