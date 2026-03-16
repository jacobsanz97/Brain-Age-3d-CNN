import collections
import collections.abc
import html.parser
import random
import os
import glob
import json
import pandas as pd
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import interpolate
from tqdm import tqdm
from scipy.stats import pearsonr

# --- ENVIRONMENT PATCHES ---
# These patches handle potential compatibility issues with older Python/PyTorch versions
# related to abstract base classes in the 'collections' module.
for attr in ['MutableMapping', 'Iterable', 'Mapping', 'Sequence']:
    if not hasattr(collections, attr):
        setattr(collections, attr, getattr(collections.abc, attr))
# This patch ensures HTML unescaping works correctly across different Python versions.
if not hasattr(html.parser.HTMLParser, 'unescape'):
    import html
    html.parser.HTMLParser.unescape = staticmethod(html.unescape)

# --- CONFIG for HPC Optimized Model ---
# This section defines configuration parameters for the training process.
# All paths and shapes are kept consistent with the original cnn.py for input compatibility.
ROOT_DIR = "/home/jacob/Desktop/duckysets"
CSV_PATH = "/home/jacob/Desktop/duckysets/ml_dataset_with_age_sex.csv"
PIPELINE = "freesurfer8001ants243"
FILENAME = "aseg_MNI.nii.gz"
CROP_SHAPE = (182, 182, 182) # Consistent cropping shape
TARGET_SHAPE = (140, 140, 140) # Consistent target image resolution
BATCH_SIZE = 8               # Increased batch size for HPC to better utilize GPU memory
MAX_SAMPLES = None           # Use all available samples unless specified
EPOCHS = 100                 # Increased epochs for a larger model to converge
LEARNING_RATE = 1e-4         # Adjusted learning rate for deeper model
LOGS_DIR = "training_logs_hpc" # Separate log directory for HPC model runs
os.makedirs(LOGS_DIR, exist_ok=True)

# --- UTILS ---
# EarlyStopping class to prevent overfitting and save computation time on HPC.
class EarlyStopping:
    def __init__(self, patience=10, delta=0): # Slightly increased patience for a larger model
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_mae):
        score = -val_mae
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# BrainAgeDataset class for loading and preprocessing NIfTI images.
# This remains identical to cnn.py to ensure input compatibility.
class BrainAgeDataset(Dataset):
    def __init__(self, samples, age_mean, age_std):
        self.samples = samples
        self.age_mean = age_mean
        self.age_std = age_std

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, age = self.samples[idx]
        try:
            data = nib.load(path).get_fdata().astype(np.float32)
            m_val = np.nanmax(data)
            if m_val == 0 or np.isnan(m_val): return None
            data = data / (m_val + 1e-8)
            
            # --- CENTER CROP ---
            orig_shape = data.shape
            starts = [max(0, (orig_shape[i] - CROP_SHAPE[i]) // 2) for i in range(3)]
            ends = [min(orig_shape[i], starts[i] + CROP_SHAPE[i]) for i in range(3)]
            data = data[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]

            # --- RESIZE ---
            norm_age = (age - self.age_mean) / self.age_std
            tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
            tensor = interpolate(tensor, size=TARGET_SHAPE, mode='nearest').squeeze(0)
            
            return tensor, torch.tensor([norm_age], dtype=torch.float32), torch.tensor([age], dtype=torch.float32)
        except: return None

# collate_fn to handle None values from dataset (e.g., failed image loading).
# This is crucial for robust data loading, especially in large datasets.
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0: return torch.tensor([]), torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)

# --- HPC OPTIMIZED CNN ARCHITECTURE ---
# This is a beefed-up version of the LaptopCNN, designed for HPC environments.
# It features more layers and increased channel depths to learn more complex features,
# leveraging the greater computational resources of GPUs.
class HPC_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature extraction layers: Deeper and wider than LaptopCNN
        self.features = nn.Sequential(
            # Block 1: Increased channels
            nn.Conv3d(1, 32, 3, stride=2, padding=1), 
            nn.BatchNorm3d(32), 
            nn.LeakyReLU(0.1),
            nn.Dropout3d(0.2), # Added dropout for regularization

            # Block 2: Further increased channels
            nn.Conv3d(32, 64, 3, stride=2, padding=1), 
            nn.BatchNorm3d(64), 
            nn.LeakyReLU(0.1),
            nn.Dropout3d(0.3),

            # Block 3: Additional convolutional layer with increased channels
            nn.Conv3d(64, 128, 3, stride=1, padding=1), # No stride reduction to preserve resolution longer
            nn.BatchNorm3d(128), 
            nn.LeakyReLU(0.1),
            
            # Block 4: Deeper layer
            nn.Conv3d(128, 128, 3, stride=2, padding=1), 
            nn.BatchNorm3d(128), 
            nn.LeakyReLU(0.1),
            nn.Dropout3d(0.4), # Increased dropout

            # Block 5: Even deeper layer with more channels
            nn.Conv3d(128, 256, 3, stride=2, padding=1), 
            nn.BatchNorm3d(256), 
            nn.LeakyReLU(0.1),
            nn.Dropout3d(0.4), # Added another dropout layer

            # Block 6: Final feature extraction layer
            nn.Conv3d(256, 512, 3, stride=2, padding=1), # Significantly more channels
            nn.BatchNorm3d(512), 
            nn.LeakyReLU(0.1),
            
            # Global Average Pooling to reduce dimensions before the fully connected layers
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten()
        )
        # Fully connected layers: Also beefed up to handle more complex features
        self.fc = nn.Sequential(
            nn.Linear(512, 256), # Increased input features from previous layer
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5), # Stronger dropout for FC layers
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x): 
        # Defines the forward pass of the network.
        # Data flows through the feature extraction layers, then through the fully connected layers.
        return self.fc(self.features(x))

# Functions to plot and save training results.
# These remain identical, but will save outputs to the new LOGS_DIR.
def plot_and_save_results(history_df, test_df):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_df['epoch'], history_df['train_mse'], label='Train MSE')
    plt.title('Training Loss (Z-Normalized)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_df['epoch'], history_df['val_mae'], label='Val MAE', color='orange')
    plt.title('Validation MAE (Years)')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(LOGS_DIR, "learning_curves.png"))
    plt.close()

    plt.figure(figsize=(7, 7))
    sns.regplot(x='true_age', y='pred_age', data=test_df, scatter_kws={'alpha':0.4})
    plt.plot([test_df['true_age'].min(), test_df['true_age'].max()], 
             [test_df['true_age'].min(), test_df['true_age'].max()], 'r--')
    plt.title('Test Set: True vs Predicted Age')
    plt.savefig(os.path.join(LOGS_DIR, "test_regression.png"))
    plt.close()

# Main training function.
# Modified to use HPC_CNN and adjusted hyperparameters.
def run_training():
    # Device setup: Prioritizes CUDA (GPU) for HPC, then MPS (Apple Silicon), fallback to CPU.
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    
    # 1. Data Indexing: Identical to cnn.py
    df_all = pd.read_csv(CSV_PATH).dropna(subset=['age'])
    print("Indexing files...")
    search_pattern = os.path.join(ROOT_DIR, "**", PIPELINE, "**", FILENAME)
    found_files = glob.glob(search_pattern, recursive=True)
    
    path_map = {}
    for f in found_files:
        parts = f.split(os.sep)
        sub = next((p for p in parts if p.startswith("sub-")), None)
        ses = next((p for p in parts if p.startswith("ses-")), None)
        if sub and ses: path_map[(sub, ses)] = f
    
    sub_to_samples = collections.defaultdict(list)
    for _, row in df_all.iterrows():
        key = (row['subject'], row['session'])
        if key in path_map:
            sub_to_samples[row['subject']].append((path_map[key], row['age']))
    
    unique_subs = list(sub_to_samples.keys())
    random.seed(42) # Ensure reproducibility of data splits
    random.shuffle(unique_subs)
    if MAX_SAMPLES: unique_subs = unique_subs[:MAX_SAMPLES]

    # 2. Data Splitting: Identical to cnn.py
    n_train_s = int(0.7 * len(unique_subs))
    n_val_s = int(0.15 * len(unique_subs))
    train_subs, val_subs, test_subs = unique_subs[:n_train_s], unique_subs[n_train_s:n_train_s+n_val_s], unique_subs[n_train_s+n_val_s:]
    
    train_list = [s for sub in train_subs for s in sub_to_samples[sub]]
    val_list = [s for sub in val_subs for s in sub_to_samples[sub]]
    test_list = [s for sub in test_subs for s in sub_to_samples[sub]]
    
    t_mean, t_std = np.mean([s[1] for s in train_list]), np.std([s[1] for s in train_list])

    # Save metadata for future predictions.
    metadata = {'t_mean': float(t_mean), 't_std': float(t_std), 'target_shape': TARGET_SHAPE}
    with open(os.path.join(LOGS_DIR, "metadata.json"), 'w') as f:
        json.dump(metadata, f)

    print(f"
--- DATASET SUMMARY ---")
    print(f"Device:            {device}")
    print(f"Crop Shape:        {CROP_SHAPE}")
    print(f"Target Resolution: {TARGET_SHAPE}")
    print(f"Batch Size:        {BATCH_SIZE}")
    print(f"Total Subjects:    {len(unique_subs)}")
    print(f"Training Scans:    {len(train_list)}")
    print(f"Validation Scans:  {len(val_list)}")
    print(f"Test Scans:        {len(test_list)}")
    print(f"Age Normalization: Mean={t_mean:.2f}, Std={t_std:.2f}")
    print("-" * 25)

    # DataLoaders for efficient batch processing.
    train_loader = DataLoader(BrainAgeDataset(train_list, t_mean, t_std), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(BrainAgeDataset(val_list, t_mean, t_std), batch_size=BATCH_SIZE, collate_fn=collate_fn)
    test_loader = DataLoader(BrainAgeDataset(test_list, t_mean, t_std), batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    # Instantiate the HPC_CNN model and move it to the selected device (GPU if available).
    model = HPC_CNN().to(device)
    # Optimizer and loss function. Using Adam for its robustness.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) # Slightly less weight decay for larger model
    criterion = nn.MSELoss()
    # Learning rate scheduler to adjust learning rate during training.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5) # Increased patience
    early_stopper = EarlyStopping(patience=10) # Using HPC version of EarlyStopping
    
    history = []
    best_val_mae = float('inf')

    # --- TRAINING LOOP ---
    for epoch in range(EPOCHS):
        model.train() # Set model to training mode
        train_mse_run, count = 0, 0
        for imgs, norm_ages, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            if imgs.numel() == 0: continue # Skip empty batches
            imgs, norm_ages = imgs.to(device), norm_ages.to(device) # Move data to device
            optimizer.zero_grad() # Clear previous gradients
            preds = model(imgs) # Forward pass
            loss = criterion(preds, norm_ages) # Calculate loss
            loss.backward() # Backpropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip gradients to prevent exploding gradients
            optimizer.step() # Update model parameters
            train_mse_run += loss.item()
            count += 1
        
        avg_train_mse = train_mse_run / (count if count > 0 else 1)

        model.eval() # Set model to evaluation mode
        val_mae_sum, val_count = 0, 0
        with torch.no_grad(): # Disable gradient calculation for validation
            for imgs, _, real_ages in val_loader:
                if imgs.numel() == 0: continue
                imgs, real_ages = imgs.to(device), real_ages.to(device)
                pred_yrs = (model(imgs) * t_std) + t_mean # Denormalize predictions
                val_mae_sum += torch.abs(pred_yrs - real_ages).sum().item()
                val_count += imgs.size(0)
        
        avg_val_mae = val_mae_sum / (val_count if val_count > 0 else 1)
        history.append({'epoch': epoch+1, 'train_mse': avg_train_mse, 'val_mae': avg_val_mae})
        print(f"Epoch {epoch+1} | Train MSE: {avg_train_mse:.4f} | Val MAE: {avg_val_mae:.2f} yrs")
        
        # Save the best model based on validation MAE
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            torch.save(model.state_dict(), os.path.join(LOGS_DIR, "best_model.pth"))

        scheduler.step(avg_val_mae) # Update learning rate scheduler
        early_stopper(avg_val_mae) # Check for early stopping
        if early_stopper.early_stop: break

    # Save training history
    pd.DataFrame(history).to_csv(os.path.join(LOGS_DIR, "training_history.csv"), index=False)

    # --- FINAL TEST ---
    print("
--- RUNNING FINAL TEST ---")
    model.load_state_dict(torch.load(os.path.join(LOGS_DIR, "best_model.pth"), map_location=device)) # Load best model
    model.eval() # Set to evaluation mode
    test_results = []
    
    with torch.no_grad():
        for imgs, _, real_ages in tqdm(test_loader, desc="Testing"):
            if imgs.numel() == 0: continue
            imgs = imgs.to(device)
            preds = (model(imgs) * t_std) + t_mean # Denormalize predictions
            for t, p in zip(real_ages.cpu().numpy(), preds.cpu().numpy()):
                test_results.append({'true_age': float(t[0]), 'pred_age': float(p[0])})

    test_df = pd.DataFrame(test_results)
    
    # Calculate and print final metrics
    mae = np.mean(np.abs(test_df['true_age'] - test_df['pred_age']))
    correlation, _ = pearsonr(test_df['true_age'], test_df['pred_age'])
    
    print("
" + "="*30)
    print("FINAL TEST RESULTS (CLI)")
    print(f"Test MAE:         {mae:.3f} years")
    print(f"Pearson r:        {correlation:.3f}")
    print(f"Total Test Scans: {len(test_df)}")
    print("="*30 + "
")

    # Save test predictions and plot results
    summary_data = {'true_age': 'SUMMARY_MAE', 'pred_age': mae}
    test_df_with_summary = pd.concat([test_df, pd.DataFrame([summary_data])], ignore_index=True)
    test_df_with_summary.to_csv(os.path.join(LOGS_DIR, "test_predictions.csv"), index=False)
    
    plot_and_save_results(pd.DataFrame(history), test_df)

    print(f"Training Complete. Results saved to: {LOGS_DIR}/")

if __name__ == "__main__":
    run_training()
