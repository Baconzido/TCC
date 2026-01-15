#!/usr/bin/env python3
"""
Script de Download de Imagens MRI - OASIS-3
Versão com SSL flexível e múltiplas opções

TCC - Samuel Augusto Souza Alves Santana
"""

import os
import sys
import requests
import argparse
import csv
import zipfile
import time
import urllib3
from tqdm import tqdm

# Desabilitar avisos de SSL (necessário para alguns servidores)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================
# CONFIGURAÇÕES
# ============================================

# Credenciais NITRC (mesmo do site)
NITRC_USERNAME = "baconzido"  # <-- EDITE co
NITRC_PASSWORD = "X68S!@#3H$V2eTg"     # <-- EDITE

# URLs alternativas
URLS = {
    'nitrc': 'https://www.nitrc.org/ir/',
    'oasis': 'https://oasis-brains.org/',
    'xnat': 'https://central.xnat.org'
}

OUTPUT_DIR = "./oasis3_mri_data"

# ============================================
# FUNÇÕES
# ============================================

def create_session(verify_ssl=False):
    """Cria sessão HTTP com opção de ignorar SSL"""
    session = requests.Session()
    session.verify = verify_ssl
    session.auth = (NITRC_USERNAME, NITRC_PASSWORD)
    return session


def test_connections():
    """Testa conexão com diferentes servidores"""
    print("\nTestando conexões...")
    
    results = {}
    for name, url in URLS.items():
        try:
            response = requests.get(url, timeout=10, verify=False)
            status = "✓ OK" if response.status_code == 200 else f"✗ {response.status_code}"
            results[name] = True if response.status_code == 200 else False
        except Exception as e:
            status = f"✗ Erro"
            results[name] = False
        print(f"  {name}: {status}")
    
    return results


def download_from_nitrc_manual_instructions():
    """Gera instruções para download manual"""
    
    print("\n" + "=" * 60)
    print("INSTRUÇÕES PARA DOWNLOAD MANUAL")
    print("=" * 60)
    
    print("""
Como o download automático não está funcionando, siga estes passos:

1. ACESSE O SITE:
   https://www.nitrc.org/projects/oasis
   
2. FAÇA LOGIN com suas credenciais NITRC

3. VÁ EM:
   "OASIS-3" → "MR Sessions" ou "Download Images"

4. PARA CADA SUJEITO, BAIXE:
   - Procure por scans "T1w" ou "MPRAGE"
   - Baixe o arquivo .nii.gz
   - Salve em: oasis3_mri_data/<SUBJECT_ID>/

5. SUJEITOS PRIORITÁRIOS (conversores - pMCI):
   OAS30007, OAS30013, OAS30019, OAS30026, OAS30027
   OAS30029, OAS30033, OAS30035, OAS30040, OAS30041

6. SUJEITOS CONTROLE (estáveis - sMCI):
   OAS30001, OAS30002, OAS30003, OAS30004, OAS30005
   OAS30006, OAS30008, OAS30009, OAS30010, OAS30011

TOTAL RECOMENDADO: Pelo menos 20-30 sujeitos para começar.
""")


def check_existing_downloads(output_dir):
    """Verifica imagens já baixadas"""
    if not os.path.exists(output_dir):
        return []
    
    downloaded = []
    for folder in os.listdir(output_dir):
        folder_path = os.path.join(output_dir, folder)
        if os.path.isdir(folder_path) and folder.startswith('OAS'):
            # Verificar se tem arquivo NIfTI
            files = os.listdir(folder_path)
            nifti_files = [f for f in files if f.endswith('.nii') or f.endswith('.nii.gz')]
            if nifti_files:
                downloaded.append(folder)
    
    return downloaded


def organize_downloads(input_dir, output_dir):
    """
    Organiza arquivos baixados manualmente.
    Se você baixou vários arquivos em uma pasta, este script organiza.
    """
    print("\nOrganizando arquivos...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Procurar por arquivos NIfTI
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                # Tentar extrair ID do sujeito do nome do arquivo
                for pattern in ['OAS3', 'oas3']:
                    if pattern in file.lower():
                        # Extrair OAS3XXXX
                        start = file.lower().find('oas3')
                        subject_id = file[start:start+8].upper()
                        
                        # Criar pasta e mover
                        subject_dir = os.path.join(output_dir, subject_id)
                        os.makedirs(subject_dir, exist_ok=True)
                        
                        src = os.path.join(root, file)
                        dst = os.path.join(subject_dir, file)
                        
                        if not os.path.exists(dst):
                            import shutil
                            shutil.copy2(src, dst)
                            print(f"  {subject_id}: {file}")
                        break


def generate_download_list(subjects_file, output_file="download_list.txt"):
    """Gera lista formatada para download manual"""
    
    subjects = []
    with open(subjects_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            subjects.append(row)
    
    pmci = [s for s in subjects if s['conversion_status'] == 'pMCI']
    smci = [s for s in subjects if s['conversion_status'] == 'sMCI']
    
    with open(output_file, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("LISTA DE SUJEITOS PARA DOWNLOAD\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("CONVERSORES (pMCI) - PRIORIDADE ALTA\n")
        f.write("-" * 40 + "\n")
        for s in pmci:
            f.write(f"{s['OASISID']}\n")
        
        f.write(f"\nTotal pMCI: {len(pmci)}\n\n")
        
        f.write("ESTÁVEIS (sMCI)\n")
        f.write("-" * 40 + "\n")
        for s in smci:
            f.write(f"{s['OASISID']}\n")
        
        f.write(f"\nTotal sMCI: {len(smci)}\n")
        f.write(f"\nTOTAL GERAL: {len(subjects)}\n")
    
    print(f"\n✓ Lista salva em: {output_file}")
    return output_file


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Download OASIS-3 MRI - Opções alternativas')
    parser.add_argument('--test', action='store_true', help='Testar conexões')
    parser.add_argument('--organize', type=str, help='Organizar arquivos de uma pasta')
    parser.add_argument('--list', action='store_true', help='Gerar lista para download manual')
    parser.add_argument('--check', action='store_true', help='Verificar downloads existentes')
    parser.add_argument('--instructions', action='store_true', help='Mostrar instruções de download')
    args = parser.parse_args()
    
    print("=" * 60)
    print("DOWNLOAD DE IMAGENS MRI - OASIS-3")
    print("=" * 60)
    
    if args.test:
        test_connections()
        return
    
    if args.instructions:
        download_from_nitrc_manual_instructions()
        return
    
    if args.list:
        if os.path.exists('mci_subjects_for_download.csv'):
            generate_download_list('mci_subjects_for_download.csv')
        else:
            print("ERRO: Arquivo mci_subjects_for_download.csv não encontrado")
        return
    
    if args.organize:
        organize_downloads(args.organize, OUTPUT_DIR)
        return
    
    if args.check:
        downloaded = check_existing_downloads(OUTPUT_DIR)
        print(f"\nImagens já baixadas: {len(downloaded)}")
        if downloaded:
            print("Sujeitos:")
            for s in sorted(downloaded):
                print(f"  - {s}")
        return
    
    # Se nenhum argumento, mostrar ajuda
    print("\nUso:")
    print("  python download_oasis_mri_v2.py --instructions  # Ver instruções")
    print("  python download_oasis_mri_v2.py --list          # Gerar lista de sujeitos")
    print("  python download_oasis_mri_v2.py --check         # Verificar downloads")
    print("  python download_oasis_mri_v2.py --organize <pasta>  # Organizar arquivos")
    print("  python download_oasis_mri_v2.py --test          # Testar conexões")
    
    print("\n" + "-" * 60)
    print("RECOMENDAÇÃO: Use --instructions para ver como baixar manualmente")
    print("-" * 60)


if __name__ == "__main__":
    main()