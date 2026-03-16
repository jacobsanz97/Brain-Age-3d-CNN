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
for attr in ['MutableMapping', 'Iterable', 'Mapping', 'Sequence']:
    if not hasattr(collections, attr):
        setattr(collections, attr, getattr(collections.abc, attr))
if not hasattr(html.parser.HTMLParser, 'unescape'):
    import html
    html.parser.HTMLParser.unescape = staticmethod(html.unescape)

# --- CONFIG ---
ROOT_DIR = "/home/jacob/Desktop/duckysets"
CSV_PATH = "/home/jacob/Desktop/duckysets/ml_dataset_with_age_sex.csv"
PIPELINE = "freesurfer8001ants243"
FILENAME = "aseg_MNI.nii.gz"
CROP_SHAPE = (182, 182, 182)
TARGET_SHAPE = (140, 140, 140)
BATCH_SIZE = 4               
MAX_SAMPLES = None
EPOCHS = 50 
LEARNING_RATE = 5e-5
LOGS_DIR = "training_logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# --- UTILS ---
class EarlyStopping:
    def __init__(self, patience=7, delta=0):
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

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0: return torch.tensor([]), torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)

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

def run_training():
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    
    # 1. Indexing
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
    random.seed(42)
    random.shuffle(unique_subs)
    if MAX_SAMPLES: unique_subs = unique_subs[:MAX_SAMPLES]

    # 2. Split
    n_train_s = int(0.7 * len(unique_subs))
    n_val_s = int(0.15 * len(unique_subs))
    train_subs, val_subs, test_subs = unique_subs[:n_train_s], unique_subs[n_train_s:n_train_s+n_val_s], unique_subs[n_train_s+n_val_s:]
    
    train_list = [s for sub in train_subs for s in sub_to_samples[sub]]
    val_list = [s for sub in val_subs for s in sub_to_samples[sub]]
    test_list = [s for sub in test_subs for s in sub_to_samples[sub]]
    
    t_mean, t_std = np.mean([s[1] for s in train_list]), np.std([s[1] for s in train_list])

    metadata = {'t_mean': float(t_mean), 't_std': float(t_std), 'target_shape': TARGET_SHAPE}
    with open(os.path.join(LOGS_DIR, "metadata.json"), 'w') as f:
        json.dump(metadata, f)

    print(f"\n--- DATASET SUMMARY ---")
    print(f"Device:           {device}")
    print(f"Crop Shape:       {CROP_SHAPE}")
    print(f"Target Resolution: {TARGET_SHAPE}")
    print(f"Batch Size:       {BATCH_SIZE}")
    print(f"Total Subjects:   {len(unique_subs)}")
    print(f"Training Scans:   {len(train_list)}")
    print(f"Validation Scans: {len(val_list)}")
    print(f"Test Scans:       {len(test_list)}")
    print(f"Age Normalization: Mean={t_mean:.2f}, Std={t_std:.2f}")
    print("-" * 25)

    train_loader = DataLoader(BrainAgeDataset(train_list, t_mean, t_std), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(BrainAgeDataset(val_list, t_mean, t_std), batch_size=BATCH_SIZE, collate_fn=collate_fn)
    test_loader = DataLoader(BrainAgeDataset(test_list, t_mean, t_std), batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    model = LaptopCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    early_stopper = EarlyStopping(patience=7)
    
    history = []
    best_val_mae = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_mse_run, count = 0, 0
        for imgs, norm_ages, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            if imgs.numel() == 0: continue
            imgs, norm_ages = imgs.to(device), norm_ages.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, norm_ages)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_mse_run += loss.item()
            count += 1
        
        avg_train_mse = train_mse_run / (count if count > 0 else 1)

        model.eval()
        val_mae_sum, val_count = 0, 0
        with torch.no_grad():
            for imgs, _, real_ages in val_loader:
                if imgs.numel() == 0: continue
                imgs, real_ages = imgs.to(device), real_ages.to(device)
                pred_yrs = (model(imgs) * t_std) + t_mean
                val_mae_sum += torch.abs(pred_yrs - real_ages).sum().item()
                val_count += imgs.size(0)
        
        avg_val_mae = val_mae_sum / (val_count if val_count > 0 else 1)
        history.append({'epoch': epoch+1, 'train_mse': avg_train_mse, 'val_mae': avg_val_mae})
        print(f"Epoch {epoch+1} | Train MSE: {avg_train_mse:.4f} | Val MAE: {avg_val_mae:.2f} yrs")
        
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            torch.save(model.state_dict(), os.path.join(LOGS_DIR, "best_model.pth"))

        scheduler.step(avg_val_mae)
        early_stopper(avg_val_mae)
        if early_stopper.early_stop: break

    pd.DataFrame(history).to_csv(os.path.join(LOGS_DIR, "training_history.csv"), index=False)

    # --- FINAL TEST ---
    print("\n--- RUNNING FINAL TEST ---")
    model.load_state_dict(torch.load(os.path.join(LOGS_DIR, "best_model.pth")))
    model.eval()
    test_results = []
    
    with torch.no_grad():
        for imgs, _, real_ages in tqdm(test_loader, desc="Testing"):
            if imgs.numel() == 0: continue
            imgs = imgs.to(device)
            preds = (model(imgs) * t_std) + t_mean
            for t, p in zip(real_ages.cpu().numpy(), preds.cpu().numpy()):
                test_results.append({'true_age': float(t[0]), 'pred_age': float(p[0])})

    test_df = pd.DataFrame(test_results)
    
    # Calculate metrics for CLI
    mae = np.mean(np.abs(test_df['true_age'] - test_df['pred_age']))
    correlation, _ = pearsonr(test_df['true_age'], test_df['pred_age'])
    
    print("\n" + "="*30)
    print("FINAL TEST RESULTS (CLI)")
    print(f"Test MAE:         {mae:.3f} years")
    print(f"Pearson r:        {correlation:.3f}")
    print(f"Total Test Scans: {len(test_df)}")
    print("="*30 + "\n")

    # Add summary row to CSV before saving
    summary_data = {'true_age': 'SUMMARY_MAE', 'pred_age': mae}
    test_df_with_summary = pd.concat([test_df, pd.DataFrame([summary_data])], ignore_index=True)
    test_df_with_summary.to_csv(os.path.join(LOGS_DIR, "test_predictions.csv"), index=False)
    
    plot_and_save_results(pd.DataFrame(history), test_df)

    print(f"Training Complete. Results saved to: {LOGS_DIR}/")

if __name__ == "__main__":
    run_training()
