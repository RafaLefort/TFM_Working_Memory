import os
import gc
import json
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import (
    DataLoader,
    Dataset)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, confusion_matrix)



# =========================================================
# INICIALIZACIÓN Y ENTRADA DATOS
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
    "results_data_augmentation")

os.makedirs(
    RESULTS_DIR,
    exist_ok=True)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu")

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =========================================================
# CARGA DE SUJETOS
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


    # NORMALIZACIÓN POR CANAL

    mean = X.mean(axis=(0,2), keepdims=True)
    std = X.std(axis=(0,2), keepdims=True) + 1e-6

    X = (X - mean) / std

    del df
    gc.collect()

    return X, y



# =========================================================
# MIXUP
# =========================================================

def frequency_mixup(
    x1,
    x2,
    alpha=0.4):

    fft1 = np.fft.rfft(
        x1,
        axis=-1)

    fft2 = np.fft.rfft(
        x2,
        axis=-1)

    mixed_fft = (
        alpha * fft1
        +
        (1 - alpha) * fft2)

    mixed = np.fft.irfft(
        mixed_fft,
        axis=-1)

    return mixed.astype(np.float32)


# =========================================================
# DEFINICIÓN AUMENTO DE DATOS
# =========================================================

class data_augmentation(Dataset):

    def __init__(
        self,
        X,
        y,
        augment=False,
        alpha=0.4):

        self.X = X
        self.y = y
        self.augment = augment
        self.alpha = alpha

        # =============================================
        # CONTADORES
        # =============================================

        self.total_samples = 0
        self.mixup_count = 0
        self.no_aug_count = 0


    def __len__(self):

        return len(self.X)


    def __getitem__(self, idx):

        x1 = self.X[idx].copy()
        y1 = self.y[idx]

        self.total_samples += 1

        applied = False

        if self.augment and np.random.rand() < 0.5:

            j = np.random.randint(0, len(self.X) - 1)

            if j >= idx:
                j += 1

            x2 = self.X[j]
            y2 = self.y[j]

            lam = self.alpha

            x_mix = frequency_mixup(
                x1,
                x2,
                alpha=lam
            )

            self.mixup_count += 1
            applied = True

            return (

                torch.tensor(
                    x_mix,
                    dtype=torch.float32
                ),

                torch.tensor(
                    y1,
                    dtype=torch.long
                ),

                torch.tensor(
                    y2,
                    dtype=torch.long
                ),

                torch.tensor(
                    lam,
                    dtype=torch.float32
                )
            )

        if not applied:

            self.no_aug_count += 1

        return (

            torch.tensor(
                x1,
                dtype=torch.float32
            ),

            torch.tensor(
                y1,
                dtype=torch.long
            ),

            torch.tensor(
                y1,
                dtype=torch.long
            ),

            torch.tensor(
                1.0,
                dtype=torch.float32
            )
        )


# =========================================================
# DEFINICIÓN CNN
# =========================================================

class CNN(nn.Module):

    def __init__(
        self,
        n_channels=64,
        n_samples=250,
        n_classes=3):
        super().__init__()

        # =================================================
        # CONVOLUCIÓN TEMPORAL
        # =================================================

        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=(1,32),
            padding=(0,15),
            bias=False)


        # =================================================
        # CONVOLUCIÓN ESPACIAL
        # =================================================

        self.spatial_conv = nn.Conv2d(
            in_channels=6,
            out_channels=24,
            kernel_size=(n_channels,1),
            groups=6,
            bias=False)


        # =================================================
        # BATCH NORMALIZACIÓN
        # =================================================

        self.bn = nn.BatchNorm2d(24)


        # =================================================
        # EPD1
        # =================================================

        self.epd1 = nn.Sequential(

            nn.ELU(),

            nn.AvgPool2d(kernel_size=(1,8)),

            nn.Dropout(0.25))


        # =================================================
        # CONVOLUCIÓN SEPARABLE
        # =================================================

        self.separable_conv = nn.Sequential(

            nn.Conv2d(in_channels=24,
                out_channels=24,
                kernel_size=(1,16),
                padding=(0,8),
                groups=24,
                bias=False),

            nn.Conv2d(
        	24,
        	24,
        	kernel_size=1,
        	bias=False))	


        # =================================================
        # EPD2
        # =================================================

        self.epd2 = nn.Sequential(

            nn.ELU(),

            nn.AvgPool2d(
                kernel_size=(1,12)),

            nn.Dropout(0.5))


        # =================================================
        # CALCULA LA DIMENSIÓN PLANA
        # =================================================

        with torch.no_grad():

            dummy = torch.zeros(
                1,
                1,
                n_channels,
                n_samples)

            x = self.temporal_conv(dummy)

            x = self.spatial_conv(x)

            x = self.bn(x)

            x = self.epd1(x)

            x = self.separable_conv(x)

            x = self.epd2(x)

            flatten_dim = x.view(
                1,
                -1).shape[1]


        # =================================================
        # CLASIFICADOR
        # =================================================

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                flatten_dim,
                n_classes))


    def forward(self, x):

        x = x.unsqueeze(1)

        x = self.temporal_conv(x)

        x = self.spatial_conv(x)

        x = self.bn(x)

        x = self.epd1(x)

        x = self.separable_conv(x)

        x = self.epd2(x)

        x = self.classifier(x)

        return x



# =========================================================
# ENTRENO (TRAIN SET)
# =========================================================

def train_model(
    model,
    train_loader,
    val_loader,
    epochs=100):

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs)

    best_loss = np.inf

    best_state = model.state_dict()

    patience = 20

    counter = 0

    train_losses = []

    val_losses = []


    for epoch in range(epochs):

        # =================================================
        # TRAIN
        # =================================================

        model.train()

        running_train = 0

        for xb, y1, y2, lam in train_loader:

            xb = xb.to(DEVICE)

            y1 = y1.to(DEVICE)

            y2 = y2.to(DEVICE)

            lam = lam.to(DEVICE)

            optimizer.zero_grad()

            out = model(xb)

            loss = (
                lam * criterion(out, y1)
                +
                (1 - lam) * criterion(out, y2)).mean()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0)

            optimizer.step()

            running_train += loss.item()


        train_loss = (running_train / len(train_loader))

        train_losses.append(train_loss)


        # =================================================
        # VALIDATION
        # =================================================

        model.eval()

        running_val = 0

        with torch.no_grad():

            for xb, y1, _, _ in val_loader:

                xb = xb.to(DEVICE)

                y1 = y1.to(DEVICE)

                out = model(xb)

                loss = criterion(
                    out,
                    y1)

                running_val += loss.item()


        val_loss = (running_val / len(val_loader))

        val_losses.append(val_loss)

        scheduler.step()


        # =================================================
        # EARLY STOPPING
        # =================================================

        if val_loss < best_loss:

            best_loss = val_loss

            best_state = model.state_dict()

            counter = 0

        else:

            counter += 1


        if counter >= patience:

            break


    # =====================================================
    # CARGAR EL MEJOR MODELO
    # =====================================================

    model.load_state_dict(best_state)


    # =====================================================
    # EVALUACIÓN FINAL
    # =====================================================

    preds = []

    labels = []

    model.eval()

    with torch.no_grad():

        for xb, y1, _, _ in val_loader:

            xb = xb.to(DEVICE)

            out = model(xb)

            pred = torch.argmax(
                out,
                dim=1).cpu().numpy()

            preds.extend(pred)

            labels.extend(y1.numpy())


    return (

	np.array(preds),

        np.array(labels),

        train_losses,

        val_losses)



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
        random_state=42)

    n_classes = len(np.unique(y))

    accuracies = []

    conf_total = np.zeros((n_classes, n_classes))

    all_train_losses = []

    all_val_losses = []


    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):

        print(f"\nFold {fold+1}")


        X_train = X[train_idx]

        X_val = X[val_idx]

        y_train = y[train_idx]

        y_val = y[val_idx]


        train_ds = data_augmentation(
            X_train,
            y_train,
            augment=True)

        val_ds = data_augmentation(
            X_val,
            y_val,
            augment=False)


        train_loader = DataLoader(
            train_ds,
            batch_size=32,
            shuffle=True)

        val_loader = DataLoader(
            val_ds,
            batch_size=32,
            shuffle=False)


        model = CNN(
            n_channels=64,
            n_samples=X.shape[-1],
            n_classes=n_classes).to(DEVICE)


        preds, labels, tr, val = train_model(
            model,
            train_loader,
            val_loader)

        # =================================================
        # ESTADÍSTICAS DE AUMENTO
        # =================================================

        print("\nAugmentation statistics")

        print(
            f"Total samples seen: "
            f"{train_ds.total_samples}")

        print(
            f"Frequency mixup: "
            f"{train_ds.mixup_count}")

        print(
            f"No augmentation: "
            f"{train_ds.no_aug_count}")


        # =================================================
        # RATIO DE AUMENTO EFECTIVO
        # =================================================

        aug_total = (train_ds.mixup_count)

        ratio = (aug_total / train_ds.total_samples)

        print(
            f"Effective augmentation ratio: "
            f"{ratio:.2f}x")


        acc = accuracy_score(
            labels,
            preds)

        cm = confusion_matrix(
            labels,
            preds)


        accuracies.append(acc)

        conf_total += cm

        all_train_losses.append(tr)

        all_val_losses.append(val)


        print(f"Fold Accuracy: {acc:.4f}")


        del model

        gc.collect()

        torch.cuda.empty_cache()


    return (

        np.mean(accuracies),

        conf_total,

        all_train_losses,

        all_val_losses)



# =========================================================
# CORRER TODOS LOS SUJETOS
# =========================================================

def run_all_subjects(
    folder,
    name):

    print(f"\n==================== {name} ====================")

    results = {}

    count = 1

    for file in sorted(os.listdir(folder)):

        if not file.endswith(".csv"):
            continue


        subj = file.split("_")[1].split(".")[0]

        print(f"\nSubject {subj} Nº{count}/20")

        count += 1

        X, y_raw = load_subject_csv(
            os.path.join(
                folder,
                file))


        le = LabelEncoder()

        y = le.fit_transform(y_raw)


        acc, cm, tr, val = run_cv(X,y)


        print(
            f"Subject {subj} "
            f"Accuracy {acc:.4f}")

        print(cm)


        results[subj] = {

            "accuracy": float(acc),

            "conf_matrix": cm.tolist(),

            "train_losses": tr,

            "val_losses": val}


        del X, y

        gc.collect()


    return results



# =========================================================
# CORRER
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
# RESUMEN
# =========================================================

def summarize_results(
    results,
    name):

    accs = [
        r["accuracy"]
        for r in results.values()]

    print(
        f"{name} Mean Accuracy: "
        f"{np.mean(accs):.4f}")


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
# GUARDAR JSON
# =========================================================

with open(
    os.path.join(
        RESULTS_DIR,
        "results_bsl.json"),
    "w") as f:

    json.dump(
        results_bsl,
        f)


with open(
    os.path.join(
        RESULTS_DIR,
        "results_sens.json"),
    "w") as f:

    json.dump(
        results_sens,
        f)


with open(
    os.path.join(
        RESULTS_DIR,
        "results_delay.json"),
    "w") as f:

    json.dump(
        results_delay,
        f)