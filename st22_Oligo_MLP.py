import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_curve, auc
from sklearn.model_selection import LeaveOneGroupOut
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# CONFIG
# ==========================================
train_csv_path = "/project/AIRC/NWang_lab/shared/xinyue_2/ST/lopo_selected_celltype/MRIdata_all_merged_Oligo_updated_updated.csv"
target_label_col = 22
batch_size = 256
lr = 1e-4
epochs = 1000
seed = 42
device = "cuda" if torch.cuda.is_available() else "cpu"
use_class_weight = True
output_dir = "/project/AIRC/NWang_lab/shared/xinyue_2/ST/lopo_selected_celltype/results_1_MLP/Oligo"
os.makedirs(output_dir, exist_ok=True)

# Parameter to choose the network model
# 你可以选择不用的网络试试，这个都是集中典型的简单网络: "MLP", "SimpleCNN", "SimpleResNet"
model_name = "MLP"

# ==========================================
# RANDOM SEED
# ==========================================
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# ==========================================
# DATA LOADING
# ==========================================
def load_one_file(csv_path, label_col_idx, sample_col_name="sampleID"):
    df = pd.read_csv(csv_path, header=0)
    feat_mri = df.iloc[:, 0:14].values.astype(np.float32)
    feat_xy = df.iloc[:, 16:18].values.astype(np.float32)
    X = np.concatenate([feat_mri, feat_xy], axis=1)
    y_raw = df.iloc[:, label_col_idx].astype(np.int64).to_numpy()
    groups = df[sample_col_name].to_numpy()
    return X, y_raw, groups, df


# ==========================================
# DATASET
# ==========================================
class TabDataset(Dataset):
    def __init__(self, X, y_enc):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y_enc, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ==========================================
# SIMPLE MODELS
# ==========================================

class MLP(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, num_classes=2):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class SimpleResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleResNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        residual = out
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return self.fc(out)


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def build_label_mapping(y_train_raw):
    classes_sorted = np.sort(np.unique(y_train_raw))
    class2idx = {int(c): i for i, c in enumerate(classes_sorted)}
    return class2idx, classes_sorted


def encode_labels(y_raw, class2idx):
    y_enc = np.full(y_raw.shape, -1, dtype=np.int64)
    for i, v in enumerate(y_raw):
        idx = class2idx.get(int(v), None)
        if idx is not None: y_enc[i] = idx
    return y_enc


def compute_class_weights(y_enc, num_classes, alpha=1, eps=1e-6):
    counts = np.bincount(y_enc, minlength=num_classes)
    N = len(y_enc)
    p = counts / (N + eps)
    weights = (1.0 / (p + eps)) ** alpha
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_fold(model, loader, class_weights):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
    model.train()
    epoch_losses = []
    for e in range(epochs):
        batch_losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        epoch_losses.append(np.mean(batch_losses))
        if e % 10 == 0 or e == 1:
            print(f"Epoch {e}: Loss={epoch_losses[-1]:.4f}")
    return epoch_losses


def infer(model, X):
    model.eval()
    probs_list, preds_list = [], []
    with torch.no_grad():
        n = X.shape[0]
        for i in range(0, n, batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32).unsqueeze(1).to(device)
            logits = model(xb)
            prob = F.softmax(logits, dim=1).cpu().numpy()
            preds = prob.argmax(axis=1)
            probs_list.append(prob)
            preds_list.append(preds)
    return np.concatenate(probs_list, axis=0), np.concatenate(preds_list, axis=0)


# ==========================================
# MAIN
# ==========================================
# ... keep your existing imports / functions ...

def main():
    X, y_raw, groups, df_all = load_one_file(train_csv_path, target_label_col, sample_col_name="sampleID")
    class2idx, idx2class = build_label_mapping(y_raw)
    y_enc = encode_labels(y_raw, class2idx)
    num_classes = len(idx2class)

    if use_class_weight:
        class_weights = compute_class_weights(y_enc, num_classes)
        print("class_weights:", class_weights.detach().cpu().numpy())
    else:
        class_weights = None
        print("class_weights: not used")

    logo = LeaveOneGroupOut()

    all_probs, all_preds, all_labels = [], [], []
    all_x, all_y, all_sampleID = [], [], []

    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y_enc, groups)):
        test_sample = np.unique(groups[test_idx])[0]
        print(f"\n=== Fold {fold + 1}: Test Sample = {test_sample} ===")

        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        train_ds = TabDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        if model_name == "MLP":
            model = MLP(input_dim=16, num_classes=num_classes).to(device)
            print("Using model: MLP")
        elif model_name == "SimpleCNN":
            model = SimpleCNN(num_classes=num_classes).to(device)
            print("Using model: SimpleCNN")
        elif model_name == "SimpleResNet":
            model = SimpleResNet(num_classes=num_classes).to(device)
            print("Using model: SimpleResNet")
        else:
            raise ValueError(f"Invalid model_name: {model_name}. Choose from 'MLP', 'SimpleCNN', 'SimpleResNet'")

        # ----------------------------------------------------------------------------------
        fold_loss = train_one_fold(model, train_loader, class_weights)

        fold_model_path = os.path.join(output_dir, f"col{target_label_col}_fold_{fold + 1}_{test_sample}_model.pth")
        torch.save(model.state_dict(), fold_model_path)

        plt.figure()
        plt.plot(fold_loss)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Fold {fold + 1} Loss")
        plt.savefig(os.path.join(output_dir, f"col{target_label_col}_fold_{fold + 1}_{test_sample}_loss_curve.png"))
        plt.close()

        probs, preds = infer(model, X_test)  # probs shape: (n_test, num_classes)

        all_x.append(df_all.iloc[test_idx]["x_coordinate"].to_numpy())
        all_y.append(df_all.iloc[test_idx]["y_coordinate"].to_numpy())
        all_sampleID.append(df_all.iloc[test_idx]["sampleID"].to_numpy())

        all_probs.append(probs)     # <-- keep probs for later concatenation
        all_preds.append(preds)
        all_labels.append(y_test)

        report_str = classification_report(
            y_test, preds, labels=list(range(num_classes)),
            target_names=[str(c) for c in idx2class],
            digits=4, zero_division=0
        )
        acc = accuracy_score(y_test, preds)
        metrics_txt = os.path.join(output_dir, f"col{target_label_col}_fold_{fold + 1}_{test_sample}_metrics.txt")
        with open(metrics_txt, "w", encoding="utf-8") as f:
            f.write(f"Test Sample: {test_sample}\n")
            f.write(report_str)
            f.write(f"\nOverall Accuracy: {acc:.6f}\n")
        print(f"col{target_label_col}_Fold {fold + 1} metrics saved to: {metrics_txt}")

        # --- fold metrics ---
        class1 = 1
        tp = np.sum((y_test == class1) & (preds == class1))
        fn = np.sum((y_test == class1) & (preds != class1))
        fp = np.sum((y_test != class1) & (preds == class1))

        recall_c1 = tp / (tp + fn + 1e-8)
        f1_c1 = 2 * tp / (2 * tp + fp + fn + 1e-8)

        macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        micro_f1 = f1_score(y_test, preds, average="micro", zero_division=0)

        fold_metrics.append({
            "fold": fold + 1,
            "test_sample": test_sample,
            "overall_accuracy": acc,
            "recall_class1": recall_c1,
            "f1score_class1": f1_c1,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1
        })

        plt.figure()
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve((y_test == i).astype(int), probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"Class {i} (AUC={roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"Fold {fold + 1} ROC")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"col{target_label_col}_fold_{fold + 1}_{test_sample}_roc.png"))
        plt.close()

    # =========================
    # Concatenate across folds
    # =========================
    all_labels_concat = np.concatenate(all_labels)
    all_preds_concat  = np.concatenate(all_preds)
    all_probs_concat  = np.concatenate(all_probs, axis=0)  # shape: (N, num_classes)

    print("\n=== Overall Metrics ===")
    print(classification_report(all_labels_concat, all_preds_concat, digits=4, zero_division=0))
    overall_acc = accuracy_score(all_labels_concat, all_preds_concat)
    print(f"Overall Accuracy: {overall_acc:.4f}")

    metrics_df = pd.DataFrame(fold_metrics)
    metrics_csv = os.path.join(output_dir, f"col{target_label_col}_all_folds_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"All-fold metrics table saved to: {metrics_csv}")

    all_x_concat = np.concatenate(all_x)
    all_y_concat = np.concatenate(all_y)
    all_sampleID_concat = np.concatenate(all_sampleID)

    # =========================
    # Add pred_prob to result_df
    # =========================
    # Option 1: probability of the predicted class (single column)
    pred_prob = all_probs_concat[np.arange(all_probs_concat.shape[0]), all_preds_concat]

    result_df = pd.DataFrame({
        "sampleID": all_sampleID_concat,
        "x_coordinate": all_x_concat,
        "y_coordinate": all_y_concat,
        "label_true_raw": all_labels_concat,
        "pred_class_raw": all_preds_concat,
        "pred_prob": pred_prob,  # <-- added
    })

    for sample_id, df_sample in result_df.groupby("sampleID"):
        out_csv = os.path.join(output_dir, f"col{target_label_col}_sample_{sample_id}_predictions.csv")
        df_sample.to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
