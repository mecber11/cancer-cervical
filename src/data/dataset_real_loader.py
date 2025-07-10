"""
Dataset Real Loader - Cargar dataset SIPaKMeD real desde Google Drive
"""

import os
import shutil
import random
import numpy as np
import pandas as pd
from pathlib import Path
from config import *

def setup_real_sipakmed_dataset():
    """
    Configurar dataset real de SIPaKMeD desde Google Drive
    """
    print("🔗 Conectando al dataset real de SIPaKMeD...")
    
    # Rutas típicas donde puede estar el dataset en Drive
    possible_drive_paths = [
        Path("/content/drive/MyDrive/SIPaKMeD"),
        Path("/content/drive/MyDrive/cervical-cancer-sipakmed"),
        Path("/content/drive/MyDrive/sipakmed"),
        Path("C:/Users/David/Google Drive/SIPaKMeD"),  # Windows
        Path("C:/Users/David/OneDrive/SIPaKMeD"),      # OneDrive
        Path("D:/SIPaKMeD"),                           # Otro disco
    ]
    
    # Buscar el dataset en las rutas posibles
    dataset_source = None
    for path in possible_drive_paths:
        if path.exists():
            print(f"✅ Dataset encontrado en: {path}")
            dataset_source = path
            break
    
    if not dataset_source:
        print("❌ Dataset no encontrado en rutas típicas")
        print("📂 Rutas buscadas:")
        for path in possible_drive_paths:
            print(f"   - {path}")
        
        # Solicitar ruta manual
        manual_path = input("\n📁 Ingresa la ruta completa a tu dataset SIPaKMeD: ")
        if manual_path and Path(manual_path).exists():
            dataset_source = Path(manual_path)
        else:
            print("❌ Ruta no válida. Usando dataset sintético...")
            return False
    
    # Copiar/organizar dataset real
    return organize_real_dataset(dataset_source)

def organize_real_dataset(source_path):
    """
    Organizar dataset real en la estructura esperada
    
    Args:
        source_path: Ruta al dataset original
    
    Returns:
        bool: True si se organizó exitosamente
    """
    print(f"📁 Organizando dataset desde: {source_path}")
    
    # Buscar estructura del dataset
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(list(source_path.rglob(ext)))
    
    if not image_files:
        print("❌ No se encontraron imágenes en el dataset")
        return False
    
    print(f"📊 Encontradas {len(image_files)} imágenes")
    
    # Mapeo de nombres de carpetas/archivos a clases
    class_mappings = {
        'dyskeratotic': ['dyskeratotic', 'dys', 'class1'],
        'koilocytotic': ['koilocytotic', 'koilo', 'class2'], 
        'metaplastic': ['metaplastic', 'meta', 'class3'],
        'parabasal': ['parabasal', 'para', 'class4'],
        'superficial_intermediate': ['superficial', 'intermediate', 'super', 'class5']
    }
    
    # Clasificar imágenes por clase
    classified_images = {class_name: [] for class_name in REAL_CLASSES.keys()}
    
    for img_path in image_files:
        # Determinar clase basada en la ruta/nombre
        img_str = str(img_path).lower()
        assigned = False
        
        for class_name, keywords in class_mappings.items():
            if any(keyword in img_str for keyword in keywords):
                classified_images[class_name].append(img_path)
                assigned = True
                break
        
        if not assigned:
            print(f"⚠️ Imagen no clasificada: {img_path.name}")
    
    # Mostrar estadísticas
    print("\n📊 DISTRIBUCIÓN DEL DATASET:")
    total_images = 0
    for class_name, images in classified_images.items():
        count = len(images)
        total_images += count
        friendly_name = CLASS_NAMES_FRIENDLY[class_name]
        print(f"   {friendly_name}: {count} imágenes")
    
    if total_images == 0:
        print("❌ No se pudo clasificar ninguna imagen")
        return False
    
    # Crear split train/val
    return create_train_val_split(classified_images)

def create_train_val_split(classified_images, train_ratio=0.8):
    """
    Crear división train/validation y copiar imágenes
    
    Args:
        classified_images: Diccionario {clase: [rutas_imagenes]}
        train_ratio: Proporción para entrenamiento
    
    Returns:
        bool: True si se creó exitosamente
    """
    print(f"\n🔄 Creando división train/val ({train_ratio:.0%}/{1-train_ratio:.0%})...")
    
    # Limpiar directorios existentes
    for split in ['train', 'val']:
        split_dir = PROCESSED_DATA_DIR / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True)
    
    # Dividir y copiar imágenes
    train_total = 0
    val_total = 0
    
    for class_name, image_paths in classified_images.items():
        if not image_paths:
            continue
            
        # Mezclar imágenes
        random.shuffle(image_paths)
        
        # Calcular división
        n_train = int(len(image_paths) * train_ratio)
        train_images = image_paths[:n_train]
        val_images = image_paths[n_train:]
        
        # Crear directorios de clase
        train_class_dir = PROCESSED_DATA_DIR / 'train' / class_name
        val_class_dir = PROCESSED_DATA_DIR / 'val' / class_name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Copiar imágenes de entrenamiento
        for i, img_path in enumerate(train_images):
            dest_path = train_class_dir / f"{class_name}_{i:04d}{img_path.suffix}"
            shutil.copy2(img_path, dest_path)
            train_total += 1
        
        # Copiar imágenes de validación
        for i, img_path in enumerate(val_images):
            dest_path = val_class_dir / f"{class_name}_val_{i:04d}{img_path.suffix}"
            shutil.copy2(img_path, dest_path)
            val_total += 1
        
        friendly_name = CLASS_NAMES_FRIENDLY[class_name]
        print(f"   {friendly_name}: {len(train_images)} train, {len(val_images)} val")
    
    print(f"\n✅ Dataset organizado:")
    print(f"   🎯 Train: {train_total} imágenes")
    print(f"   🎯 Val: {val_total} imágenes")
    print(f"   🎯 Total: {train_total + val_total} imágenes")
    
    return True

def download_sipakmed_from_api():
    """
    Descargar dataset SIPaKMeD usando credenciales API (si disponible)
    """
    print("🌐 Intentando descargar dataset desde API SIPaKMeD...")
    
    try:
        import requests
        import zipfile
        
        # URL hipotética de la API (ajustar según API real)
        api_url = "https://api.sipakmed.com/dataset/download"
        
        headers = {
            'Authorization': f'Bearer {SIPAKMED_TOKEN}',
            'User': SIPAKMED_USERNAME
        }
        
        print("📡 Conectando a la API...")
        response = requests.get(api_url, headers=headers, stream=True)
        
        if response.status_code == 200:
            zip_path = RAW_DATA_DIR / "sipakmed_dataset.zip"
            
            print("📥 Descargando dataset...")
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print("📂 Extrayendo archivos...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(RAW_DATA_DIR)
            
            print("✅ Dataset descargado exitosamente")
            return RAW_DATA_DIR
        else:
            print(f"❌ Error en API: {response.status_code}")
            return None
            
    except ImportError:
        print("❌ requests no disponible para descarga")
        return None
    except Exception as e:
        print(f"❌ Error descargando: {e}")
        return None

def load_real_dataset():
    """
    Función principal para cargar dataset real
    
    Returns:
        bool: True si se cargó exitosamente
    """
    print("🚀 CARGANDO DATASET REAL DE SIPAKMED")
    print("=" * 60)
    
    # Opción 1: Buscar dataset local
    if setup_real_sipakmed_dataset():
        return True
    
    # Opción 2: Descargar desde API
    download_path = download_sipakmed_from_api()
    if download_path and setup_real_sipakmed_dataset():
        return True
    
    # Opción 3: Fallback a dataset sintético
    print("\n⚠️ No se pudo cargar dataset real")
    print("🔄 ¿Quieres continuar con dataset sintético? (y/n): ")
    
    choice = input().lower().strip()
    if choice in ['y', 'yes', 'sí', 'si', '']:
        print("📝 Usando dataset sintético...")
        from src.data.dataset_creator import create_sample_dataset
        create_sample_dataset()
        return True
    else:
        print("❌ Deteniendo ejecución")
        return False

def verify_dataset():
    """
    Verificar que el dataset esté correctamente organizado
    """
    print("\n🔍 Verificando dataset...")
    
    train_dir = PROCESSED_DATA_DIR / 'train'
    val_dir = PROCESSED_DATA_DIR / 'val'
    
    if not train_dir.exists() or not val_dir.exists():
        print("❌ Directorios train/val no encontrados")
        return False
    
    total_train = 0
    total_val = 0
    
    for class_name in REAL_CLASSES.keys():
        train_class_dir = train_dir / class_name
        val_class_dir = val_dir / class_name
        
        train_count = len(list(train_class_dir.glob('*.*'))) if train_class_dir.exists() else 0
        val_count = len(list(val_class_dir.glob('*.*'))) if val_class_dir.exists() else 0
        
        total_train += train_count
        total_val += val_count
        
        friendly_name = CLASS_NAMES_FRIENDLY[class_name]
        print(f"   {friendly_name}: {train_count} train, {val_count} val")
    
    print(f"\n📊 TOTAL: {total_train} train, {total_val} val")
    
    if total_train == 0 or total_val == 0:
        print("❌ Dataset vacío o incompleto")
        return False
    
    print("✅ Dataset verificado correctamente")
    return True

if __name__ == "__main__":
    # Cargar dataset real
    if load_real_dataset():
        verify_dataset()
    else:
        print("❌ Error cargando dataset")
        