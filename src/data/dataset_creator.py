# Copiar del artifact dataset_creator_py"""


import os
import random
import numpy as np
import cv2
from config import *

def create_sample_dataset():
    """
    Crear dataset sintético de ejemplo con células cervicales
    """
    print("🔄 Creando dataset de ejemplo...")
    
    # Crear directorios
    train_dir = PROCESSED_DATA_DIR / 'train'
    val_dir = PROCESSED_DATA_DIR / 'val'
    
    for split_dir in [train_dir, val_dir]:
        for class_name in REAL_CLASSES.keys():
            class_dir = split_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
    
    # Generar imágenes para cada split
    splits_config = {
        'train': 80,  # 80 imágenes por clase
        'val': 20     # 20 imágenes por clase
    }
    
    for split, num_samples in splits_config.items():
        split_dir = PROCESSED_DATA_DIR / split
        
        for class_name in REAL_CLASSES.keys():
            class_dir = split_dir / class_name
            
            print(f"  📝 Generando {num_samples} imágenes para {class_name} ({split})")
            
            for i in range(num_samples):
                # Crear imagen base
                img = _create_cell_image(class_name)
                
                # Guardar imagen
                img_path = class_dir / f"{class_name}_{i:03d}.jpg"
                cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    # Mostrar estadísticas
    _show_dataset_stats()
    print("✅ Dataset de ejemplo creado")

def _create_cell_image(class_name):
    """
    Crear una imagen sintética de célula según el tipo
    """
    # Imagen base con ruido
    img = np.random.randint(180, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    center_x, center_y = IMG_SIZE // 2, IMG_SIZE // 2
    
    if class_name == "dyskeratotic":
        # Células displásicas: múltiples círculos irregulares
        cv2.circle(img, (center_x, center_y), random.randint(40, 70), (120, 80, 120), -1)
        for _ in range(5):
            x = random.randint(center_x-30, center_x+30)
            y = random.randint(center_y-30, center_y+30)
            cv2.circle(img, (x, y), random.randint(5, 15), (80, 50, 80), -1)
            
    elif class_name == "koilocytotic":
        # Células koilocitóticas: halo característico
        cv2.circle(img, (center_x, center_y), 50, (200, 180, 200), -1)
        cv2.circle(img, (center_x, center_y), 25, (140, 100, 140), -1)
        
    elif class_name == "metaplastic":
        # Células metaplásicas: forma elíptica
        cv2.ellipse(img, (center_x, center_y), (60, 30), 0, 0, 360, (160, 130, 160), -1)
        
    elif class_name == "parabasal":
        # Células parabasales: núcleo prominente
        cv2.circle(img, (center_x, center_y), 35, (180, 150, 180), -1)
        cv2.circle(img, (center_x, center_y), 20, (100, 80, 100), -1)
        
    else:  # superficial_intermediate
        # Células superficiales-intermedias: citoplasma amplio
        cv2.circle(img, (center_x, center_y), 80, (200, 180, 200), -1)
        cv2.circle(img, (center_x, center_y), 15, (120, 100, 120), -1)
    
    # Agregar ruido realista
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    return img

def _show_dataset_stats():
    """
    Mostrar estadísticas del dataset creado
    """
    print(f"\n📊 ESTADÍSTICAS DEL DATASET:")
    
    total_images = 0
    for split in ['train', 'val']:
        split_dir = PROCESSED_DATA_DIR / split
        print(f"\n📁 {split.upper()}:")
        
        split_total = 0
        for class_name in REAL_CLASSES.keys():
            class_dir = split_dir / class_name
            if class_dir.exists():
                count = len([f for f in class_dir.iterdir() 
                           if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
                friendly_name = CLASS_NAMES_FRIENDLY[class_name]
                print(f"   {friendly_name}: {count} imágenes")
                split_total += count
        
        print(f"   TOTAL {split}: {split_total} imágenes")
        total_images += split_total
    
    print(f"\n📈 TOTAL DATASET: {total_images} imágenes")
    print(f"🎯 {len(REAL_CLASSES)} clases de células cervicales")

def get_dataset_info():
    """
    Obtener información del dataset existente
    """
    info = {
        'train': {},
        'val': {},
        'total': 0
    }
    
    for split in ['train', 'val']:
        split_dir = PROCESSED_DATA_DIR / split
        split_total = 0
        
        for class_name in REAL_CLASSES.keys():
            class_dir = split_dir / class_name
            if class_dir.exists():
                count = len([f for f in class_dir.iterdir() 
                           if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
                info[split][class_name] = count
                split_total += count
        
        info[split]['total'] = split_total
        info['total'] += split_total
    
    return info

if __name__ == "__main__":
    # Ejecutar directamente para crear solo el dataset
    create_sample_dataset()