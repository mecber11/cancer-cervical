#!/usr/bin/env python3
"""
Script para entrenar un modelo especifico
Uso: python scripts/train_single_model.py --model mobilenet
"""

import sys
import argparse
from pathlib import Path

# Agregar el directorio raiz al path
sys.path.append(str(Path(__file__).parent.parent))

from config import *
from src.models.trainer import train_cervical_model
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

def main():
    parser = argparse.ArgumentParser(description='Entrenar modelo individual')
    parser.add_argument('--model', choices=['mobilenet', 'resnet', 'efficientnet'], 
                       default='mobilenet', help='Modelo a entrenar')
    
    args = parser.parse_args()
    
    models_map = {
        'mobilenet': (MobileNetV2, mobilenet_preprocess, "MobileNetV2"),
        'resnet': (ResNet50, resnet_preprocess, "ResNet50"),
        'efficientnet': (EfficientNetB0, efficientnet_preprocess, "EfficientNetB0")
    }
    
    model_fn, preprocess_fn, name = models_map[args.model]
    
    print(f"Entrenando {name}...")
    result = train_cervical_model(model_fn, preprocess_fn, name)
    
    if result:
        print(f"✓ Entrenamiento exitoso: {result['accuracy']:.4f}")
    else:
        print("Error en el entrenamiento")

if __name__ == "__main__":
    main()
