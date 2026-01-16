#!/usr/bin/env python3
"""
Modelo Híbrido Multimodal para Predição MCI → AD
Combina CNN 3D (imagens MRI) + Features Tabulares (clínicas/cognitivas)

TCC - Samuel Augusto Souza Alves Santana
Universidade Federal de Sergipe
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURAÇÕES
# ============================================

BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

PREPROCESSED_DIR = "./oasis3_preprocessed"
TABULAR_DATA_FILE = "mci_dataset_improved.csv"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================
# DATASET MULTIMODAL
# ============================================

class MultimodalDataset(Dataset):
    def __init__(self, subject_ids, labels, tabular_features, image_dir, augment=False):
        self.subject_ids = subject_ids
        self.labels = labels
        self.tabular_features = tabular_features
        self.image_dir = image_dir
        self.augment = augment
    
    def __len__(self):
        return len(self.subject_ids)
    
    def __getitem__(self, idx):
        subject_id = self.subject_ids[idx]
        label = self.labels[idx]
        tabular = self.tabular_features[idx]
        
        filepath = os.path.join(self.image_dir, f"{subject_id}.npy")
        image = np.load(filepath)
        image = np.expand_dims(image, axis=0)
        
        if self.augment:
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=3).copy()
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=1).copy()
            if np.random.rand() > 0.5:
                noise = np.random.normal(0, 0.01, image.shape)
                image = np.clip(image + noise, 0, 1)
        
        return torch.FloatTensor(image), torch.FloatTensor(tabular), torch.FloatTensor([label])


# ============================================
# ARQUITETURAS
# ============================================

class CNN3D_Encoder(nn.Module):
    def __init__(self):
        super(CNN3D_Encoder, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten()
        )
        
        self.embedding = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.embedding(x)
        return x


class TabularEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=64):
        super(TabularEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, embedding_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.encoder(x)


class MultimodalFusion_Early(nn.Module):
    """Early Fusion: Concatena embeddings antes do classificador"""
    
    def __init__(self, tabular_input_dim, dropout=0.5):
        super(MultimodalFusion_Early, self).__init__()
        
        self.cnn_encoder = CNN3D_Encoder()
        self.tabular_encoder = TabularEncoder(tabular_input_dim, embedding_dim=64)
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, image, tabular):
        cnn_features = self.cnn_encoder(image)
        tab_features = self.tabular_encoder(tabular)
        fused = torch.cat([cnn_features, tab_features], dim=1)
        return self.classifier(fused)


class MultimodalFusion_Intermediate(nn.Module):
    """Intermediate Fusion com Attention"""
    
    def __init__(self, tabular_input_dim, dropout=0.5):
        super(MultimodalFusion_Intermediate, self).__init__()
        
        self.cnn_encoder = CNN3D_Encoder()
        self.tabular_encoder = TabularEncoder(tabular_input_dim, embedding_dim=64)
        
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, image, tabular):
        cnn_features = self.cnn_encoder(image)
        tab_features = self.tabular_encoder(tabular)
        
        combined = torch.cat([cnn_features, tab_features], dim=1)
        attention_weights = self.attention(combined)
        
        fused = (attention_weights[:, 0:1] * cnn_features + 
                 attention_weights[:, 1:2] * tab_features)
        
        return self.classifier(fused), attention_weights


class MultimodalFusion_Late(nn.Module):
    """Late Fusion: Combina probabilidades"""
    
    def __init__(self, tabular_input_dim, dropout=0.5):
        super(MultimodalFusion_Late, self).__init__()
        
        self.cnn_encoder = CNN3D_Encoder()
        self.cnn_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.tabular_encoder = TabularEncoder(tabular_input_dim, embedding_dim=64)
        self.tabular_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.fusion_weight = nn.Parameter(torch.tensor([0.5]))
    
    def forward(self, image, tabular):
        cnn_features = self.cnn_encoder(image)
        cnn_pred = self.cnn_classifier(cnn_features)
        
        tab_features = self.tabular_encoder(tabular)
        tab_pred = self.tabular_classifier(tab_features)
        
        w = torch.sigmoid(self.fusion_weight)
        fused_pred = w * cnn_pred + (1 - w) * tab_pred
        
        return fused_pred, cnn_pred, tab_pred


# ============================================
# TREINAMENTO
# ============================================

def train_epoch(model, dataloader, criterion, optimizer, device, fusion_type='early'):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for images, tabular, labels in dataloader:
        images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if fusion_type == 'intermediate':
            outputs, _ = model(images, tabular)
        elif fusion_type == 'late':
            outputs, _, _ = model(images, tabular)
        else:
            outputs = model(images, tabular)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(dataloader), np.array(all_preds), np.array(all_labels)


def evaluate(model, dataloader, criterion, device, fusion_type='early'):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    attention_weights_all = []
    
    with torch.no_grad():
        for images, tabular, labels in dataloader:
            images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)
            
            if fusion_type == 'intermediate':
                outputs, att_weights = model(images, tabular)
                attention_weights_all.extend(att_weights.cpu().numpy())
            elif fusion_type == 'late':
                outputs, _, _ = model(images, tabular)
            else:
                outputs = model(images, tabular)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(dataloader), np.array(all_preds), np.array(all_labels), attention_weights_all


def train_model(model, train_loader, val_loader, num_epochs, device, 
                learning_rate=1e-4, patience=10, fusion_type='early'):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_auc = 0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        train_loss, _, _ = train_epoch(model, train_loader, criterion, optimizer, device, fusion_type)
        val_loss, val_preds, val_labels, _ = evaluate(model, val_loader, criterion, device, fusion_type)
        
        val_auc = roc_auc_score(val_labels.flatten(), val_preds.flatten())
        scheduler.step(val_loss)
        
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), f'best_multimodal_{fusion_type}.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping na época {epoch+1}")
                break
        
        if (epoch + 1) % 5 == 0:
            print(f"  Época {epoch+1}/{num_epochs}: Train Loss={train_loss:.4f}, Val AUC={val_auc:.4f}")
    
    return best_auc


# ============================================
# PREPARAR DADOS
# ============================================

def prepare_tabular_data(tabular_file, subject_ids):
    """Prepara features tabulares para os sujeitos com imagens"""
    
    df = pd.read_csv(tabular_file)
    
    # Features disponíveis
    feature_cols = [
        'baseline_age', 'sex_numeric', 'EDUC', 'has_apoe4',
        'baseline_mmse', 'baseline_cdrsum',
        'LOGIMEM', 'MEMUNITS', 'ANIMALS', 'VEG', 'digfor', 'digback',
        'tma', 'tmb', 'craftvrs', 'craftdvr', 'mocatots',
        'hippocampus_norm', 'entorhinal_norm', 'ventricles_norm',
        'TOTAL_HIPPOCAMPUS_VOLUME', 'Left-Hippocampus_volume', 'Right-Hippocampus_volume'
    ]
    
    available_cols = [c for c in feature_cols if c in df.columns]
    print(f"  Features tabulares disponíveis: {len(available_cols)}")
    
    # Filtrar sujeitos com imagens
    df_filtered = df[df['OASISID'].isin(subject_ids)].copy()
    df_filtered = df_filtered.set_index('OASISID')
    df_filtered = df_filtered.loc[subject_ids]
    
    X = df_filtered[available_cols].values
    y = df_filtered['label'].values
    
    # Imputar e normalizar
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, len(available_cols)


# ============================================
# VALIDAÇÃO CRUZADA
# ============================================

def run_cv(subject_ids, labels, tabular_features, image_dir, n_folds=5, fusion_type='early'):
    
    print(f"\n{'='*50}")
    print(f"FUSÃO: {fusion_type.upper()}")
    print(f"{'='*50}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = []
    tabular_dim = tabular_features.shape[1]
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(subject_ids, labels)):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        
        train_subjects = [subject_ids[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        train_tabular = tabular_features[train_idx]
        
        val_subjects = [subject_ids[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        val_tabular = tabular_features[val_idx]
        
        print(f"  Train: {len(train_subjects)} | Val: {len(val_subjects)}")
        
        train_dataset = MultimodalDataset(train_subjects, train_labels, train_tabular, image_dir, augment=True)
        val_dataset = MultimodalDataset(val_subjects, val_labels, val_tabular, image_dir, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        # Modelo
        if fusion_type == 'early':
            model = MultimodalFusion_Early(tabular_dim).to(DEVICE)
        elif fusion_type == 'intermediate':
            model = MultimodalFusion_Intermediate(tabular_dim).to(DEVICE)
        else:
            model = MultimodalFusion_Late(tabular_dim).to(DEVICE)
        
        # Treinar
        best_auc = train_model(model, train_loader, val_loader, NUM_EPOCHS, DEVICE,
                               LEARNING_RATE, EARLY_STOPPING_PATIENCE, fusion_type)
        
        # Avaliar
        model.load_state_dict(torch.load(f'best_multimodal_{fusion_type}.pt'))
        _, val_preds, val_labels_arr, att_weights = evaluate(model, val_loader, nn.BCELoss(), DEVICE, fusion_type)
        
        val_preds_binary = (val_preds.flatten() > 0.5).astype(int)
        val_labels_flat = val_labels_arr.flatten().astype(int)
        
        auc = roc_auc_score(val_labels_flat, val_preds.flatten())
        bal_acc = balanced_accuracy_score(val_labels_flat, val_preds_binary)
        
        tn, fp, fn, tp = confusion_matrix(val_labels_flat, val_preds_binary).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        results.append({'fold': fold+1, 'auc': auc, 'bal_acc': bal_acc, 'sens': sens, 'spec': spec})
        
        if fusion_type == 'intermediate' and att_weights:
            avg_att = np.mean(att_weights, axis=0)
            print(f"  Attention: CNN={avg_att[0]:.3f}, Tabular={avg_att[1]:.3f}")
        
        print(f"  AUC: {auc:.4f} | Bal.Acc: {bal_acc:.4f} | Sens: {sens:.4f} | Spec: {spec:.4f}")
    
    return results


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 60)
    print("MODELO HÍBRIDO MULTIMODAL: CNN 3D + FEATURES TABULARES")
    print("=" * 60)
    
    print(f"\nDevice: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Verificar dados
    if not os.path.exists(PREPROCESSED_DIR):
        print(f"\nERRO: Diretório {PREPROCESSED_DIR} não encontrado!")
        return
    
    if not os.path.exists(TABULAR_DATA_FILE):
        print(f"\nERRO: Arquivo {TABULAR_DATA_FILE} não encontrado!")
        return
    
    # Carregar sujeitos com imagens
    image_subjects = sorted([f.replace('.npy', '') for f in os.listdir(PREPROCESSED_DIR) if f.endswith('.npy')])
    print(f"\n[1] Sujeitos com imagens: {len(image_subjects)}")
    
    # Preparar dados tabulares
    print("\n[2] Preparando dados tabulares...")
    tabular_features, labels, tabular_dim = prepare_tabular_data(TABULAR_DATA_FILE, image_subjects)
    
    print(f"    Shape: {tabular_features.shape}")
    print(f"    pMCI: {sum(labels)} | sMCI: {len(labels) - sum(labels)}")
    
    # Testar estratégias de fusão
    all_results = {}
    
    for fusion_type in ['early', 'intermediate', 'late']:
        results = run_cv(image_subjects, labels, tabular_features, PREPROCESSED_DIR, n_folds=5, fusion_type=fusion_type)
        all_results[fusion_type] = results
    
    # Resumo
    print("\n" + "=" * 60)
    print("RESULTADOS FINAIS - MODELOS MULTIMODAIS")
    print("=" * 60)
    
    summary = []
    for fusion_type, results in all_results.items():
        aucs = [r['auc'] for r in results]
        bal_accs = [r['bal_acc'] for r in results]
        sens = [r['sens'] for r in results]
        specs = [r['spec'] for r in results]
        
        summary.append({
            'fusion_type': fusion_type,
            'auc_mean': np.mean(aucs),
            'auc_std': np.std(aucs),
            'bal_acc_mean': np.mean(bal_accs),
            'sens_mean': np.mean(sens),
            'spec_mean': np.mean(specs)
        })
        
        print(f"\n{fusion_type.upper()} FUSION:")
        print(f"  AUC:         {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        print(f"  Bal.Acc:     {np.mean(bal_accs):.4f}")
        print(f"  Sensitivity: {np.mean(sens):.4f}")
        print(f"  Specificity: {np.mean(specs):.4f}")
    
    print("\n" + "-" * 60)
    print("COMPARAÇÃO COM MODELOS UNIMODAIS:")
    print("-" * 60)
    print("  Regressão Logística (tabular): AUC = 0.837")
    print("  CNN 3D (imagem):               AUC = 0.828")
    
    best = max(summary, key=lambda x: x['auc_mean'])
    print(f"\n  MELHOR MULTIMODAL ({best['fusion_type']}): AUC = {best['auc_mean']:.4f} ± {best['auc_std']:.4f}")
    
    # Salvar
    with open('multimodal_results.json', 'w') as f:
        json.dump({'results': all_results, 'summary': summary}, f, indent=2, default=float)
    
    print("\n✓ Resultados salvos em multimodal_results.json")


if __name__ == "__main__":
    main()