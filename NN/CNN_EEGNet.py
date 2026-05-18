import os
import gc
import json
import copy
import pickle
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import (
    DataLoader,
    TensorDataset)

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix)


# =========================================================
# INITIALIZATION
# =========================================================

BASE_PATH = "/home/rlefort/"

BSL_FOLDER = os.path.join(
    BASE_PATH,
    "BSL_subjects")

SENS_FOLDER = os.path.join(
    BASE_PATH,
    "SENS_subjects")

DELAY_FOLDER = os.path.join(
    BASE_PATH,
    "DELAY_subjects")

RESULTS_DIR = os.path.join(
    BASE_PATH,
    "results_CNN_EEGNet")

os.makedirs(
    RESULTS_DIR,
    exist_ok=True)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu")

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =========================================================
# LOAD SUBJECT CSV
# =========================================================

def load_subject_csv(
    file_path,
    expected_channels=64):

    df = pd.read_csv(file_path)

    eeg_cols = [
        c for c in df.columns
        if c.startswith("EEG.V")]

    X = []

    y = []

    grouped = df.groupby(
        ["trialID"],
        sort=False)

    for _, group in grouped:

        group = group.sort_values(
            "channel")

        if len(group) != expected_channels:
            continue

        trial = group[eeg_cols].to_numpy(
            dtype=np.float32)

        label = group["y"].iloc[0]

        X.append(trial)

        y.append(label)


    X = np.stack(X)

    y = np.array(y)


    del df

    gc.collect()

    return X, y


# =========================================================
# EEGNET
# =========================================================

class EEGNet(nn.Module):

    def __init__(
        self,
        n_channels=64,
        n_samples=250,
        n_classes=3,
        dropout_rate=0.5):

        super().__init__()

        # =================================================
        # EEGNET HYPERPARAMETERS
        # =================================================

        F1 = 8
        D = 2
        F2 = F1 * D


        # =================================================
        # BLOCK 1
        # Temporal Convolution
        # =================================================

        self.block1 = nn.Sequential(

            # ---------------------------------------------
            # Temporal convolution
            # ---------------------------------------------

            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, 64),
                padding=(0, 32),
                bias=False
            ),

            # ---------------------------------------------
            # BatchNorm
            # ---------------------------------------------

            nn.BatchNorm2d(F1),


            # ---------------------------------------------
            # Depthwise spatial convolution
            # ---------------------------------------------

            nn.Conv2d(
                in_channels=F1,
                out_channels=F1 * D,
                kernel_size=(n_channels, 1),
                groups=F1,
                bias=False
            ),

            # ---------------------------------------------
            # BatchNorm
            # ---------------------------------------------

            nn.BatchNorm2d(F1 * D),

            # ---------------------------------------------
            # ELU
            # ---------------------------------------------

            nn.ELU(),

            # ---------------------------------------------
            # Average Pooling
            # ---------------------------------------------

            nn.AvgPool2d(
                kernel_size=(1, 4)
            ),

            # ---------------------------------------------
            # Spatial Dropout
            # ---------------------------------------------

            nn.Dropout2d(
                p=dropout_rate
            )
        )


        # =================================================
        # BLOCK 2
        # Separable Convolution
        # =================================================

        self.block2 = nn.Sequential(

            # ---------------------------------------------
            # Depthwise temporal convolution
            # ---------------------------------------------

            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F1 * D,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=F1 * D,
                bias=False
            ),

            # ---------------------------------------------
            # Pointwise convolution
            # ---------------------------------------------

            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F2,
                kernel_size=(1, 1),
                bias=False
            ),

            # ---------------------------------------------
            # BatchNorm
            # ---------------------------------------------

            nn.BatchNorm2d(F2),

            # ---------------------------------------------
            # ELU
            # ---------------------------------------------

            nn.ELU(),

            # ---------------------------------------------
            # Average Pooling
            # ---------------------------------------------

            nn.AvgPool2d(
                kernel_size=(1, 8)
            ),

            # ---------------------------------------------
            # Spatial Dropout
            # ---------------------------------------------

            nn.Dropout2d(
                p=dropout_rate
            )
        )


        # =================================================
        # AUTOMATIC FLATTEN DIMENSION
        # =================================================

        with torch.no_grad():

            dummy = torch.zeros(
                1,
                1,
                n_channels,
                n_samples
            )

            x = self.block1(dummy)

            x = self.block2(x)

            flatten_dim = x.view(
                1,
                -1
            ).shape[1]


        # =================================================
        # CLASSIFIER
        # =================================================

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(F2, n_classes)


    # =====================================================
    # FORWARD
    # =====================================================

    def forward(self, x):

        x = x.unsqueeze(1)

        x = self.block1(x)
        x = self.block2(x)

        x = self.gap(x)          # (B, F2, 1, 1)

        x = x.view(x.size(0), -1)

        return self.classifier(x)


# =========================================================
# DATA AUGMENTATION
# =========================================================

def augment_eeg(x):

    # =====================================================
    # GAUSSIAN NOISE
    # =====================================================

    x = x + 0.01 * torch.randn_like(x)


    # =====================================================
    # TIME MASKING
    # =====================================================

    if torch.rand(1) < 0.5:

        t = x.size(-1)

        mask_size = int(t * 0.1)

        start = torch.randint(
            0,
            t - mask_size,
            (1,))

        x[:, :, start:start + mask_size] = 0


    # =====================================================
    # CHANNEL DROPOUT
    # =====================================================

    if torch.rand(1) < 0.5:

        c = x.size(1)

        drop_ch = torch.randperm(c)[
            :int(c * 0.1)]

        x[:, drop_ch, :] = 0

    return x


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(
    model,
    train_loader,
    val_loader,
    epochs=60):

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=5,
        factor=0.5)

    best_loss = np.inf

    best_state = copy.deepcopy(
        model.state_dict())

    patience = 12

    counter = 0

    train_losses = []

    val_losses = []


    for epoch in range(epochs):

        # =================================================
        # TRAIN
        # =================================================

        model.train()

        running_train = 0

        for xb, yb in train_loader:

            xb = xb.to(DEVICE)

            yb = yb.to(DEVICE)

            xb = augment_eeg(xb)

            optimizer.zero_grad()

            out = model(xb)

            loss = criterion(
                out,
                yb)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0)

            optimizer.step()

            running_train += loss.item()


        train_loss = (
            running_train / len(train_loader))

        train_losses.append(train_loss)


        # =================================================
        # VALIDATION
        # =================================================

        model.eval()

        running_val = 0

        with torch.no_grad():

            for xb, yb in val_loader:

                xb = xb.to(DEVICE)

                yb = yb.to(DEVICE)

                out = model(xb)

                loss = criterion(
                    out,
                    yb)

                running_val += loss.item()


        val_loss = (
            running_val / len(val_loader))

        val_losses.append(val_loss)

        scheduler.step(val_loss)


        # =================================================
        # EARLY STOPPING
        # =================================================

        if val_loss < best_loss:

            best_loss = val_loss

            best_state = copy.deepcopy(
                model.state_dict())

            counter = 0

        else:

            counter += 1


        if counter >= patience:
            break


    # =====================================================
    # LOAD BEST MODEL
    # =====================================================

    model.load_state_dict(best_state)


    # =====================================================
    # FINAL EVALUATION
    # =====================================================

    preds = []

    labels = []

    model.eval()

    with torch.no_grad():

        for xb, yb in val_loader:

            xb = xb.to(DEVICE)

            out = model(xb)

            pred = torch.argmax(
                out,
                dim=1).cpu().numpy()

            preds.extend(pred)

            labels.extend(yb.numpy())


    return (

        np.array(preds),

        np.array(labels),

        train_losses,

        val_losses
    )


# =========================================================
# CROSS VALIDATION
# =========================================================

def run_cv(
    X,
    y,
    n_splits=10):

    kf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED)

    n_classes = len(np.unique(y))

    accuracies = []

    conf_matrix_total = np.zeros(
        (n_classes, n_classes))

    all_train_losses = []

    all_val_losses = []


    for fold, (train_idx, val_idx) in enumerate(
        kf.split(X, y)):

        print(f"\nFold {fold+1}")


        X_train = X[train_idx]

        X_val = X[val_idx]

        y_train = y[train_idx]

        y_val = y[val_idx]


        # =================================================
        # NORMALIZATION USING TRAIN ONLY
        # =================================================

        mean = X_train.mean(
            axis=(0, 2),
            keepdims=True)

        std = X_train.std(
            axis=(0, 2),
            keepdims=True) + 1e-6

        X_train = (
            X_train - mean) / std

        X_val = (
            X_val - mean) / std


        train_ds = TensorDataset(

            torch.tensor(
                X_train,
                dtype=torch.float32),

            torch.tensor(
                y_train,
                dtype=torch.long)
        )

        val_ds = TensorDataset(

            torch.tensor(
                X_val,
                dtype=torch.float32),

            torch.tensor(
                y_val,
                dtype=torch.long)
        )


        train_loader = DataLoader(
            train_ds,
            batch_size=16,
            shuffle=True,
            num_workers=4,
            pin_memory=True)

        val_loader = DataLoader(
            val_ds,
            batch_size=16,
            shuffle=False,
            num_workers=4,
            pin_memory=True)


        model = EEGNet(
            n_channels=64,
            n_classes=n_classes).to(DEVICE)


        preds, labels, train_losses, val_losses = (
            train_model(
                model,
                train_loader,
                val_loader)
        )


        acc = accuracy_score(
            labels,
            preds)

        cm = confusion_matrix(
            labels,
            preds)


        accuracies.append(acc)

        conf_matrix_total += cm

        all_train_losses.append(train_losses)

        all_val_losses.append(val_losses)


        print(
            f"Fold Accuracy: {acc:.4f}")


        del model

        gc.collect()

        torch.cuda.empty_cache()


    return (

        np.mean(accuracies),

        conf_matrix_total,

        all_train_losses,

        all_val_losses
    )


# =========================================================
# RUN ALL SUBJECTS
# =========================================================

def run_all_subjects(
    folder,
    name):

    print(f"\n==================== {name} ====================")

    results = {}

    txt_path = os.path.join(
        RESULTS_DIR,
        f"results_{name}.txt")


    with open(txt_path, "w") as f:

        f.write(
            f"===== {name} =====\n")


        count = 1

        for file in sorted(os.listdir(folder)):

            if not file.endswith(".csv"):
                continue


            subj = file.split("_")[1].split(".")[0]

            print(
                f"\nSubject {subj} "
                f"Nº{count}/20")

            count += 1


            X, y_raw = load_subject_csv(
                os.path.join(
                    folder,
                    file))


            le = LabelEncoder()

            y = le.fit_transform(
                y_raw)


            f.write(
                f"\nSubject {subj}\n")

            f.write(
                f"Unique labels: "
                f"{np.unique(y)}\n")


            acc, cm, train_losses, val_losses = (
                run_cv(X, y)
            )


            print(
                f"Mean Accuracy: {acc:.4f}")

            print(
                "Confusion Matrix:\n",
                cm)


            f.write(
                f"Mean Accuracy: "
                f"{acc:.4f}\n")

            f.write(
                f"Confusion Matrix:\n"
                f"{cm}\n")


            results[subj] = {

                "accuracy": float(acc),

                "conf_matrix": cm.tolist(),

                "train_losses": train_losses,

                "val_losses": val_losses
            }


            del X, y

            gc.collect()


    return results


# =========================================================
# EXECUTION
# =========================================================

results_bsl = run_all_subjects(
    BSL_FOLDER,
    "BSL")

results_sens = run_all_subjects(
    SENS_FOLDER,
    "SENS")

results_delay = run_all_subjects(
    DELAY_FOLDER,
    "DELAY")


# =========================================================
# SUMMARY
# =========================================================

def summarize_results(
    results,
    name):

    accs = [
        r["accuracy"]
        for r in results.values()
    ]

    print(
        f"\n{name} Mean Accuracy "
        f"across subjects: "
        f"{np.mean(accs):.4f}"
    )


summarize_results(
    results_bsl,
    "BSL")

summarize_results(
    results_sens,
    "SENS")

summarize_results(
    results_delay,
    "DELAY")


# =========================================================
# SAVE RESULTS
# =========================================================

with open(
    os.path.join(
        RESULTS_DIR,
        "results_bsl.pkl"),
    "wb") as f:

    pickle.dump(
        results_bsl,
        f)


with open(
    os.path.join(
        RESULTS_DIR,
        "results_sens.pkl"),
    "wb") as f:

    pickle.dump(
        results_sens,
        f)


with open(
    os.path.join(
        RESULTS_DIR,
        "results_delay.pkl"),
    "wb") as f:

    pickle.dump(
        results_delay,
        f)