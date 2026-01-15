#!/usr/bin/env python3
"""
Pré-processamento de Imagens MRI para CNN 3D
TCC - Samuel Augusto Souza Alves Santana

Este script:
1. Carrega imagens NIfTI
2. Normaliza intensidades
3. Redimensiona para tamanho padrão
4. Salva em formato pronto para treinamento

Dependências: pip install nibabel numpy scikit-image tqdm
"""

import os
import numpy as np
import nibabel as nib
from glob import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    from skimage.transform import resize
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("AVISO: scikit-image não instalado. Instale com: pip install scikit-image")

# ============================================
# CONFIGURAÇÕES
# ============================================

# Tamanho de saída padrão (comum na literatura)
TARGET_SHAPE = (128, 128, 128)  # Pode usar (96, 96, 96) para menos memória

# Diretórios
INPUT_DIR = "./oasis3_mri_data"
OUTPUT_DIR = "./oasis3_preprocessed"

# ============================================
# FUNÇÕES DE PRÉ-PROCESSAMENTO
# ============================================

def load_nifti(filepath):
    """Carrega imagem NIfTI"""
    img = nib.load(filepath)
    data = img.get_fdata()
    return data, img.affine


def normalize_intensity(image, percentile_low=1, percentile_high=99):
    """
    Normaliza intensidades da imagem para [0, 1]
    Usa percentis para robustez a outliers
    """
    # Remover valores de fundo (zeros)
    mask = image > 0
    
    if mask.sum() == 0:
        return image
    
    # Calcular percentis apenas nos voxels não-zero
    p_low = np.percentile(image[mask], percentile_low)
    p_high = np.percentile(image[mask], percentile_high)
    
    # Clip e normalizar
    image_clipped = np.clip(image, p_low, p_high)
    image_norm = (image_clipped - p_low) / (p_high - p_low + 1e-8)
    
    return image_norm


def resize_volume(image, target_shape):
    """Redimensiona volume 3D para tamanho alvo"""
    if not HAS_SKIMAGE:
        # Fallback simples usando numpy
        factors = np.array(target_shape) / np.array(image.shape)
        # Zoom simples (não ideal, mas funciona)
        from scipy.ndimage import zoom
        return zoom(image, factors, order=1)
    
    return resize(image, target_shape, mode='constant', anti_aliasing=True)


def preprocess_image(filepath, target_shape=TARGET_SHAPE):
    """Pipeline completo de pré-processamento"""
    
    # 1. Carregar
    image, affine = load_nifti(filepath)
    
    # 2. Garantir 3D (remover dimensão temporal se existir)
    if len(image.shape) == 4:
        image = image[:, :, :, 0]
    
    # 3. Normalizar intensidade
    image = normalize_intensity(image)
    
    # 4. Redimensionar
    image = resize_volume(image, target_shape)
    
    # 5. Garantir tipo float32
    image = image.astype(np.float32)
    
    return image


def find_nifti_files(subject_dir):
    """Encontra arquivos NIfTI em um diretório"""
    patterns = ['*.nii.gz', '*.nii', '**/*.nii.gz', '**/*.nii']
    
    for pattern in patterns:
        files = glob(os.path.join(subject_dir, pattern), recursive=True)
        # Filtrar apenas T1w
        t1_files = [f for f in files if 't1' in f.lower() or 'mprage' in f.lower() or 'anat' in f.lower()]
        if t1_files:
            return t1_files[0]
        if files:
            return files[0]
    
    return None


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 60)
    print("PRÉ-PROCESSAMENTO DE IMAGENS MRI")
    print("=" * 60)
    
    # Verificar dependências
    try:
        import scipy
    except ImportError:
        print("ERRO: scipy não instalado. Execute: pip install scipy")
        return
    
    # Criar diretório de saída
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Listar sujeitos
    subjects = [d for d in os.listdir(INPUT_DIR) 
                if os.path.isdir(os.path.join(INPUT_DIR, d)) and d.startswith('OAS')]
    
    if not subjects:
        print(f"\nNenhum sujeito encontrado em {INPUT_DIR}")
        print("Execute primeiro o download das imagens.")
        return
    
    print(f"\n[1] Encontrados {len(subjects)} sujeitos")
    print(f"[2] Tamanho alvo: {TARGET_SHAPE}")
    print(f"[3] Saída: {OUTPUT_DIR}")
    
    print("\n[4] Processando...")
    print("-" * 60)
    
    success = 0
    failed = 0
    
    for subject_id in tqdm(subjects, desc="Pré-processando"):
        subject_dir = os.path.join(INPUT_DIR, subject_id)
        output_file = os.path.join(OUTPUT_DIR, f"{subject_id}.npy")
        
        # Pular se já processado
        if os.path.exists(output_file):
            success += 1
            continue
        
        # Encontrar arquivo NIfTI
        nifti_file = find_nifti_files(subject_dir)
        
        if nifti_file is None:
            print(f"\n  {subject_id}: Nenhum arquivo NIfTI encontrado")
            failed += 1
            continue
        
        try:
            # Pré-processar
            image = preprocess_image(nifti_file, TARGET_SHAPE)
            
            # Salvar como numpy array
            np.save(output_file, image)
            success += 1
            
        except Exception as e:
            print(f"\n  {subject_id}: ERRO - {e}")
            failed += 1
    
    # Resumo
    print("\n" + "=" * 60)
    print("PRÉ-PROCESSAMENTO CONCLUÍDO")
    print("=" * 60)
    print(f"  ✓ Sucesso: {success}")
    print(f"  ✗ Falhas:  {failed}")
    print(f"  Arquivos salvos em: {OUTPUT_DIR}")
    
    # Verificar tamanho dos arquivos
    if success > 0:
        sample_file = glob(os.path.join(OUTPUT_DIR, "*.npy"))[0]
        sample = np.load(sample_file)
        print(f"\n  Shape das imagens: {sample.shape}")
        print(f"  Dtype: {sample.dtype}")
        print(f"  Tamanho por arquivo: {os.path.getsize(sample_file) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
