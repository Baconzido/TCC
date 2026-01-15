#!/usr/bin/env python3
"""
CNN 3D para Predição de Conversão MCI → AD
TCC - Samuel Augusto Souza Alves Santana
Universidade Federal de Sergipe

Arquitetura baseada em:
- Spasov et al. (2019) - 3D CNN para predição de AD
- Bapat et al. (2023) - Deep learning para MCI

Dependências: pip install torch torchvision numpy scikit-learn tqdm matplotlib
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURAÇÕES
# ============================================

# Hiperparâmetros
BATCH_SIZE = 4  # Reduzido por causa da memória (imagens 3D são grandes)
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Diretórios
PREPROCESSED_DIR = "./oasis3_preprocessed"
LABELS_FILE = "mci_subjects_for_download.csv"

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================
# DATASET
# ============================================

class MRIDataset(Dataset):
    """Dataset para imagens MRI 3D"""
    
    def __init__(self, subject_ids, labels, data_dir, augment=False):
        self.subject_ids = subject_ids
        self.labels = labels
        self.data_dir = data_dir
        self.augment = augment
    
    def __len__(self):
        return len(self.subject_ids)
    
    def __getitem__(self, idx):
        subject_id = self.subject_ids[idx]
        label = self.labels[idx]
        
        # Carregar imagem pré-processada
        filepath = os.path.join(self.data_dir, f"{subject_id}.npy")
        image = np.load(filepath)
        
        # Adicionar canal (C, D, H, W)
        image = np.expand_dims(image, axis=0)
        
        # Data augmentation (se treino)
        if self.augment:
            image = self._augment(image)
        
        # Converter para tensor
        image = torch.FloatTensor(image)
        label = torch.FloatTensor([label])
        
        return image, label
    
    def _augment(self, image):
        """Augmentation simples para imagens 3D"""
        # Flip horizontal aleatório
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=3).copy()
        
        # Flip sagital aleatório
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=1).copy()
        
        # Pequeno ruído gaussiano
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.01, image.shape)
            image = image + noise
            image = np.clip(image, 0, 1)
        
        return image


# ============================================
# ARQUITETURA CNN 3D
# ============================================

class CNN3D_MCI(nn.Module):
    """
    CNN 3D para classificação MCI → AD
    
    Arquitetura inspirada em VGG-like para 3D
    Input: (batch, 1, 128, 128, 128)
    Output: (batch, 1) - probabilidade de conversão
    """
    
    def __init__(self, input_shape=(128, 128, 128), dropout=0.5):
        super(CNN3D_MCI, self).__init__()
        
        # Bloco 1
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)  # 128 -> 64
        )
        
        # Bloco 2
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)  # 64 -> 32
        )
        
        # Bloco 3
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)  # 32 -> 16
        )
        
        # Bloco 4
        self.conv4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)  # 16 -> 8
        )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classificador
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


class CNN3D_Light(nn.Module):
    """
    Versão mais leve da CNN 3D (para GPUs com menos memória)
    Input: (batch, 1, 96, 96, 96) ou (batch, 1, 64, 64, 64)
    """
    
    def __init__(self, dropout=0.5):
        super(CNN3D_Light, self).__init__()
        
        self.features = nn.Sequential(
            # Bloco 1: 96 -> 48
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            # Bloco 2: 48 -> 24
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            # Bloco 3: 24 -> 12
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            # Bloco 4: 12 -> 6
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            # Global pooling
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================
# TREINAMENTO
# ============================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Treina uma época"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_preds), np.array(all_labels)


def evaluate(model, dataloader, criterion, device):
    """Avalia o modelo"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_preds), np.array(all_labels)


def train_model(model, train_loader, val_loader, num_epochs, device, 
                learning_rate=1e-4, patience=10):
    """Pipeline completo de treinamento"""
    
    # Loss com peso para classes desbalanceadas
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_auc = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    
    for epoch in range(num_epochs):
        # Treino
        train_loss, _, _ = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validação
        val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        
        # Métricas
        val_auc = roc_auc_score(val_labels.flatten(), val_preds.flatten())
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Histórico
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        
        # Early stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            # Salvar melhor modelo
            torch.save(model.state_dict(), 'best_cnn3d_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping na época {epoch+1}")
                break
        
        if (epoch + 1) % 5 == 0:
            print(f"  Época {epoch+1}/{num_epochs}: "
                  f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val AUC={val_auc:.4f}")
    
    return history, best_auc


# ============================================
# VALIDAÇÃO CRUZADA
# ============================================

def run_cross_validation(subject_ids, labels, data_dir, n_folds=5, use_light_model=False):
    """Executa validação cruzada"""
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(subject_ids, labels)):
        print(f"\n{'='*50}")
        print(f"FOLD {fold+1}/{n_folds}")
        print(f"{'='*50}")
        
        # Split
        train_subjects = [subject_ids[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_subjects = [subject_ids[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        
        print(f"  Train: {len(train_subjects)} (pMCI: {sum(train_labels)})")
        print(f"  Val:   {len(val_subjects)} (pMCI: {sum(val_labels)})")
        
        # Datasets
        train_dataset = MRIDataset(train_subjects, train_labels, data_dir, augment=True)
        val_dataset = MRIDataset(val_subjects, val_labels, data_dir, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        # Modelo
        if use_light_model:
            model = CNN3D_Light(dropout=0.5).to(DEVICE)
        else:
            model = CNN3D_MCI(dropout=0.5).to(DEVICE)
        
        # Treinar
        history, best_auc = train_model(
            model, train_loader, val_loader,
            num_epochs=NUM_EPOCHS,
            device=DEVICE,
            learning_rate=LEARNING_RATE,
            patience=EARLY_STOPPING_PATIENCE
        )
        
        # Avaliar melhor modelo
        model.load_state_dict(torch.load('best_cnn3d_model.pt'))
        _, val_preds, val_labels_arr = evaluate(model, val_loader, nn.BCELoss(), DEVICE)
        
        val_preds_binary = (val_preds.flatten() > 0.5).astype(int)
        val_labels_flat = val_labels_arr.flatten().astype(int)
        
        accuracy = accuracy_score(val_labels_flat, val_preds_binary)
        auc = roc_auc_score(val_labels_flat, val_preds.flatten())
        
        tn, fp, fn, tp = confusion_matrix(val_labels_flat, val_preds_binary).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        results.append({
            'fold': fold + 1,
            'auc': auc,
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity
        })
        
        print(f"\n  Resultados Fold {fold+1}:")
        print(f"    AUC:         {auc:.4f}")
        print(f"    Accuracy:    {accuracy:.4f}")
        print(f"    Sensitivity: {sensitivity:.4f}")
        print(f"    Specificity: {specificity:.4f}")
    
    return results


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 60)
    print("CNN 3D PARA PREDIÇÃO MCI → AD")
    print("=" * 60)
    
    print(f"\nDevice: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memória: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Verificar dados
    if not os.path.exists(PREPROCESSED_DIR):
        print(f"\nERRO: Diretório {PREPROCESSED_DIR} não encontrado!")
        print("Execute primeiro: python preprocessar_mri.py")
        return
    
    # Carregar labels
    import csv
    subject_ids = []
    labels = []
    
    with open(LABELS_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject_id = row['OASISID']
            # Verificar se imagem existe
            if os.path.exists(os.path.join(PREPROCESSED_DIR, f"{subject_id}.npy")):
                subject_ids.append(subject_id)
                labels.append(int(row['label']))
    
    print(f"\n[1] Sujeitos com imagens: {len(subject_ids)}")
    print(f"    pMCI: {sum(labels)}")
    print(f"    sMCI: {len(labels) - sum(labels)}")
    
    if len(subject_ids) < 10:
        print("\nAVISO: Poucos sujeitos. Baixe mais imagens para resultados confiáveis.")
        return
    
    # Validação cruzada
    print("\n[2] Iniciando validação cruzada 5-fold...")
    
    # Usar modelo light se pouca memória GPU
    use_light = DEVICE.type == 'cpu' or (DEVICE.type == 'cuda' and 
               torch.cuda.get_device_properties(0).total_memory < 8e9)
    
    if use_light:
        print("    Usando modelo CNN3D_Light (menos memória)")
    
    results = run_cross_validation(
        subject_ids, labels, PREPROCESSED_DIR,
        n_folds=5, use_light_model=use_light
    )
    
    # Resumo
    print("\n" + "=" * 60)
    print("RESULTADOS FINAIS - CNN 3D")
    print("=" * 60)
    
    aucs = [r['auc'] for r in results]
    accs = [r['accuracy'] for r in results]
    sens = [r['sensitivity'] for r in results]
    specs = [r['specificity'] for r in results]
    
    print(f"\nMétricas (média ± std):")
    print(f"  AUC:         {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  Accuracy:    {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Sensitivity: {np.mean(sens):.4f} ± {np.std(sens):.4f}")
    print(f"  Specificity: {np.mean(specs):.4f} ± {np.std(specs):.4f}")
    
    # Salvar resultados
    import json
    with open('cnn3d_results.json', 'w') as f:
        json.dump({
            'results_per_fold': results,
            'summary': {
                'auc_mean': np.mean(aucs),
                'auc_std': np.std(aucs),
                'accuracy_mean': np.mean(accs),
                'sensitivity_mean': np.mean(sens),
                'specificity_mean': np.mean(specs)
            }
        }, f, indent=2)
    
    print("\n✓ Resultados salvos em cnn3d_results.json")


if __name__ == "__main__":
    main()
