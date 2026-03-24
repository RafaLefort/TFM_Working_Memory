import pickle
import torch

import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================
# 1. LOAD DATA
# =========================

path = "C:/Users/lefor/Desktop/OneDriveBackupFiles/Documentos/Q4 MUEI/TFM/GitHub/NN/"

BSL = pd.read_parquet(path + "BSL_allChan.parquet")
SENS = pd.read_parquet(path + "SENS_allChan.parquet")
DELAY = pd.read_parquet(path + "DELAY_allChan.parquet")

# =========================
# 2. RESHAPE PER SUBJECT
# =========================

def reshape_per_subject(df, expected_channels=64):
    eeg_cols = [c for c in df.columns if c.startswith("EEG.V")]
    data = {}

    grouped = df.groupby(["subjectID", "trialID"], sort=False)

    for (subj, trial), group in grouped:
        group = group.sort_values("channel")
        if len(group) != expected_channels:
            continue

        trial_data = group[eeg_cols].to_numpy(dtype=np.float32)
        label = group["y"].iloc[0]

        if subj not in data:
            data[subj] = {"X": [], "y": []}

        data[subj]["X"].append(trial_data)
        data[subj]["y"].append(label)

    # Convert to numpy & re-index labels to 0-based
    for subj in data:
        data[subj]["X"] = np.stack(data[subj]["X"])
        y = np.array(data[subj]["y"])
        data[subj]["y"] = np.array(data[subj]["y"])

    return data

bsl_data = reshape_per_subject(BSL)
sens_data = reshape_per_subject(SENS)
delay_data = reshape_per_subject(DELAY)

# =========================
# 3. MODEL (CTCNN)
# =========================

class CTCNN(nn.Module): 
    def __init__(self, n_channels=64, n_classes=3, dropout=0.3): 
        super().__init__()
        
        self.temp_conv1 = nn.Conv1d(n_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32) 
        self.temp_conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2) 
        self.bn2 = nn.BatchNorm1d(64) 
         
        self.cross_conv = nn.Conv1d(64, 128, kernel_size=1) 
        self.bn3 = nn.BatchNorm1d(128) 
        
        self.dropout = nn.Dropout(dropout) 
        self.pool = nn.AdaptiveAvgPool1d(1) 
        self.fc = nn.Linear(128, n_classes) 
        
    def forward(self, x): 
        x = F.relu(self.bn1(self.temp_conv1(x))) 
        x = F.relu(self.bn2(self.temp_conv2(x))) 
        x = F.relu(self.bn3(self.cross_conv(x)))

        x = self.dropout(x) 
        x = self.pool(x).squeeze(-1) 
        return self.fc(x)

# =========================
# 4. TRAIN FUNCTION
# =========================

def train_model(model, train_loader, val_loader, device, epochs=15):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)

            out = model(xb)
            preds = torch.argmax(out, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    return np.array(all_preds), np.array(all_labels)

# =========================
# 5. CROSS-VALIDATION
# =========================

def run_cv(X, y, n_splits=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    n_classes = len(np.unique(y))
    accuracies = []
    conf_matrix_total = np.zeros((n_classes, n_classes))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
#        print(f"  Fold {fold+1}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(y_train, dtype=torch.long))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                               torch.tensor(y_val, dtype=torch.long))

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

        model = CTCNN(n_channels=64, n_classes=n_classes).to(device)

        preds, labels = train_model(model, train_loader, val_loader, device)

        acc = accuracy_score(labels, preds)
        cm = confusion_matrix(labels, preds)

        accuracies.append(acc)
        conf_matrix_total += cm

    return np.mean(accuracies), conf_matrix_total

# =========================
# 6. RUN PER SUBJECT
# =========================

def run_all_subjects(data, name):
    print(f"\n===== {name} =====")

    results = {}

    with open(f"results_{name}.txt", "w") as f:

        f.write(f"===== {name} =====\n")

        for subj, d in data.items():
            print(f"\nSubject {subj}")
            f.write(f"\nSubject {subj}\n")

            X = d["X"]

            le = LabelEncoder()
            y = le.fit_transform(d["y"])

#            print("Unique labels:", np.unique(y)) 
            f.write(f"Unique labels: {np.unique(y)}\n")

            acc, cm = run_cv(X, y)

            print(f"Mean Accuracy: {acc:.4f}")
            print("Confusion Matrix:\n", cm)

            f.write(f"Mean Accuracy: {acc:.4f}\n")
            f.write(f"Confusion Matrix:\n{cm}\n")

            results[subj] = {
                "accuracy": acc,
                "conf_matrix": cm
            }

    return results

# =========================
# 7. EXECUTION
# =========================

results_bsl = run_all_subjects(bsl_data, "BSL")
results_sens = run_all_subjects(sens_data, "SENS")
results_delay = run_all_subjects(delay_data, "DELAY")

# =========================
# 8. RESULTS
# =========================

def summarize_results(results, name):
    accs = [res["accuracy"] for res in results.values()]
    print(f"\n{name} Mean Accuracy across subjects: {np.mean(accs):.4f}")

results_bsl_summary = summarize_results(results_bsl, "BSL")
results_sens_summary = summarize_results(results_sens, "SENS")
results_delay_summary = summarize_results(results_delay, "DELAY")

with open("results_bsl.pkl", "wb") as f:
    pickle.dump(results_bsl, f)

with open("results_sens.pkl", "wb") as f:
    pickle.dump(results_sens, f)

with open("results_delay.pkl", "wb") as f:
    pickle.dump(results_delay, f)