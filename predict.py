import os
import json
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
import argparse

# --- ENVIRONMENT PATCHES (from cnn.py) ---
import collections
import collections.abc
import html.parser
for attr in ['MutableMapping', 'Iterable', 'Mapping', 'Sequence']:
    if not hasattr(collections, attr):
        setattr(collections, attr, getattr(collections.abc, attr))
if not hasattr(html.parser.HTMLParser, 'unescape'):
    import html
    html.parser.HTMLParser.unescape = staticmethod(html.unescape)

# --- CONFIG (from cnn.py) ---
LOGS_DIR = "training_logs"
METADATA_PATH = os.path.join(LOGS_DIR, "metadata.json")
MODEL_PATH = os.path.join(LOGS_DIR, "best_model.pth")

# Load metadata for preprocessing
with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)

CROP_SHAPE = (182, 182, 182) # Hardcoded based on cnn.py
TARGET_SHAPE = tuple(metadata['target_shape'])
T_MEAN = metadata['t_mean']
T_STD = metadata['t_std']

# --- MODEL (from cnn.py) ---
class LaptopCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=2, padding=1), 
            nn.BatchNorm3d(16), nn.LeakyReLU(0.1),
            
            nn.Conv3d(16, 32, 3, stride=2, padding=1), 
            nn.BatchNorm3d(32), nn.LeakyReLU(0.1),
            nn.Dropout3d(0.3),
            
            nn.Conv3d(32, 32, 3, stride=1, padding=1), 
            nn.BatchNorm3d(32), nn.LeakyReLU(0.1),
            
            nn.Conv3d(32, 64, 3, stride=2, padding=1), 
            nn.BatchNorm3d(64), nn.LeakyReLU(0.1),
            nn.Dropout3d(0.3),
            
            nn.Conv3d(64, 128, 3, stride=2, padding=1), 
            nn.BatchNorm3d(128), nn.LeakyReLU(0.1),
            
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 128), nn.LeakyReLU(0.1),
            nn.Dropout(0.4), 
            nn.Linear(128, 1)
        )
    def forward(self, x): return self.fc(self.features(x))

# --- PREPROCESSING FUNCTIONS (adapted from BrainAgeDataset in cnn.py) ---
def preprocess_image(nifti_path):
    try:
        data = nib.load(nifti_path).get_fdata().astype(np.float32)
        m_val = np.nanmax(data)
        if m_val == 0 or np.isnan(m_val):
            raise ValueError("Image data is all zeros or contains NaNs.")
        data = data / (m_val + 1e-8)
        
        # --- CENTER CROP ---
        orig_shape = data.shape
        starts = [max(0, (orig_shape[i] - CROP_SHAPE[i]) // 2) for i in range(3)]
        ends = [min(orig_shape[i], starts[i] + CROP_SHAPE[i]) for i in range(3)]
        data = data[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]

        # --- RESIZE ---
        tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
        tensor = interpolate(tensor, size=TARGET_SHAPE, mode='nearest').squeeze(0)
        
        return tensor
    except Exception as e:
        print(f"Error during preprocessing of {nifti_path}: {e}")
        return None

def predict(nifti_file_path, model, device):
    preprocessed_tensor = preprocess_image(nifti_file_path)
    if preprocessed_tensor is None:
        return None

    # Add batch dimension
    input_tensor = preprocessed_tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        normalized_prediction = model(input_tensor).cpu().numpy()
        # Denormalize the prediction
        predicted_age = (normalized_prediction * T_STD) + T_MEAN
    
    return predicted_age[0][0] # Assuming a single prediction value

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict brain age from a NIfTI image using a trained CNN model.")
    parser.add_argument("nifti_file", type=str, help="Path to the input NIfTI file (.nii.gz)")
    args = parser.parse_args()

    if not os.path.exists(args.nifti_file):
        print(f"Error: NIfTI file not found at {args.nifti_file}")
        exit(1)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained model not found at {MODEL_PATH}")
        exit(1)
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        exit(1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")

    model = LaptopCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    print(f"Making prediction for {args.nifti_file}...")
    predicted_age = predict(args.nifti_file, model, device)

    if predicted_age is not None:
        print(f"Predicted Brain Age: {predicted_age:.2f} years")
    else:
        print("Prediction failed.")
