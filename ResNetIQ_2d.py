import os
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import hilbert, butter, sosfiltfilt
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# IQ
def extract_iq(signal):
    analytic = hilbert(signal, axis=1)  # shape: (4, N), complex
    I = np.real(analytic)
    Q = np.imag(analytic)
    iq_data = np.vstack([I, Q])  # concat I and Q shaped (8, N)
    return iq_data
# bandpass
def bandpass(data, fs):
    nyq = 0.5 * fs
    sos = butter(4, [1950 / nyq, 2050 / nyq], btype='band', output='sos')
    return np.array([sosfiltfilt(sos, ch) for ch in data])
# Dataset
class NPYDataset(Dataset):
    def __init__(self, root_dir, fs=4200):
        self.root_dir = root_dir
        self.fs = fs
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.root_dir, file_name)
        data = bandpass(np.load(file_path), fs=self.fs)
        iq_data = extract_iq(data)
        iq_data = torch.tensor(iq_data, dtype=torch.float32)
        angle = float(file_name.split('_')[0])
        angle = torch.tensor(angle / 360.0, dtype=torch.float32)
        return iq_data, angle
# ResNetIQ
class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out
class ResNetIQ2D(nn.Module):
    def __init__(self):
        super(ResNetIQ2D, self).__init__()
        # one conv2d with a maxpool to start
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(8,7), stride=(1,1), padding=(0,3), bias=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,1))

        # residualblocks
        self.layer1 = ResidualBlock2D(16, 32, kernel_size=(1,5), stride=(1,2), padding=(0,2))
        self.layer2 = ResidualBlock2D(32, 64, kernel_size=(1,3), stride=(1,2), padding=(0,1))
        self.layer3 = ResidualBlock2D(64, 128, kernel_size=(1,3), stride=(1,2), padding=(0,1))
        self.layer4 = ResidualBlock2D(128, 256, kernel_size=(1,1), stride=(1,2), padding=(0,0))

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [batch, 8, length] -> [batch, 1, 8, length]
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.maxpool1(x)
        x = F.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x).view(x.size(0), -1)
        x = self.fc(x).squeeze(-1)
        return x
# Huber loss
def angular_huber_loss(pred, target, delta=5.0):
    pred_deg = pred * 360
    target_deg = target * 360
    diff = pred_deg - target_deg
    angular_diff = torch.remainder(diff + 180, 360) - 180
    abs_diff = torch.abs(angular_diff)
    loss = torch.where(abs_diff < delta, 0.5 * angular_diff ** 2, delta * (abs_diff - 0.5 * delta))
    return torch.mean(loss)
# Training function
def train_model(model, train_loader, val_loader, epochs=200, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler()
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6,threshold=1e-5)

    best_rmse = float('inf')
    best_epoch = -1
    train_losses, val_maes, train_maes,train_rmses,val_rmses = [], [], [],[],[]

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_preds, train_targets = [], []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast():
                preds = model(x)
                loss = angular_huber_loss(preds, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            train_preds.extend(preds.detach().cpu().numpy())
            train_targets.extend(y.cpu().numpy())

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        train_preds_deg = np.array(train_preds) * 360
        train_targets_deg = np.array(train_targets) * 360
        train_diff = np.abs(train_preds_deg - train_targets_deg)
        train_mae = np.mean(np.minimum(train_diff, 360 - train_diff))
        train_rmse = np.sqrt(np.mean(np.minimum(train_diff, 360 - train_diff) ** 2))
        train_maes.append(train_mae)
        train_rmses.append(train_rmse)

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with autocast():
                    preds = model(x)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y.cpu().numpy())

        val_preds_deg = np.array(val_preds) * 360
        val_targets_deg = np.array(val_targets) * 360
        val_diff = np.abs(val_preds_deg - val_targets_deg)
        val_mae = np.mean(np.minimum(val_diff, 360 - val_diff))
        val_rmse = np.sqrt(np.mean(np.minimum(val_diff, 360 - val_diff) ** 2))
        val_maes.append(val_mae)
        val_rmses.append(val_rmse)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.1f}, MAE={train_mae:.2f}, RMSE={train_rmse:.2f} | Val MAE={val_mae:.2f}, RMSE={val_rmse:.2f}")

        lr_scheduler.step(val_rmse)

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_rmse': best_rmse,
            }, '/home/ResNetIQ_M4.pth')

    print(f"→ Best RMSE so far: {best_rmse:.2f} at Epoch {best_epoch}")

    # save data
    metrics = {
        'Epoch': list(range(1, epochs + 1)),
        'Train_Loss': train_losses,
        'Train_MAE': train_maes,
        'Train_RMSE': train_rmses,
        'Val_MAE': val_maes,
        'Val_RMSE': val_rmses,
    }
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv('/home/ResNetIQ_M4.csv', index=False)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
if __name__ == '__main__':
    seed = 12
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)

    batch_size = 256
    root_train = '/root/autodl-fs/data1/train5'
    root_val = '/root/autodl-fs/data1/valid5'

    train_dataset = NPYDataset(root_train)
    val_dataset = NPYDataset(root_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = ResNetIQ2D()
    train_model(model, train_loader, val_loader, epochs=200)
