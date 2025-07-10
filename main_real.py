"""
MAIN - Sistema Completo SIPaKMeD con Dataset Real
Ejecuta el entrenamiento completo usando el dataset real de SIPaKMeD
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

# Importar módulos del proyecto
from config import *
from src.data.dataset_real_loader import load_real_dataset, verify_dataset
from src.models.trainer import train_cervical_model
from src.utils.visualization import generate_final_reports
from src.evaluation.evaluator import evaluate_all_models

def setup_tensorflow():
    """Configurar TensorFlow para el entorno"""
    print("Configurando TensorFlow...")
    
    # Configurar GPU si está disponible
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, GPU_MEMORY_GROWTH)
            print(f"GPU configurada: {len(gpus)} dispositivo(s)")
        except RuntimeError as e:
            print(f"Error configurando GPU: {e}")
    else:
        print("Usando CPU")
    
    # Configurar precisión mixta si está habilitada
    if MIXED_PRECISION:
        from tensorflow.keras.mixed_precision import Policy
        policy = Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Precision mixta habilitada")

def check_dataset_location():
    """
    Ayudar al usuario a localizar su dataset
    """
    print("\n📁 UBICACION DEL DATASET SIPAKMED")
    print("=" * 50)
    print("Para usar tu dataset real, debe estar en una de estas ubicaciones:")
    print()
    print("WINDOWS:")
    print("  - C:/Users/David/Google Drive/SIPaKMeD/")
    print("  - C:/Users/David/OneDrive/SIPaKMeD/") 
    print("  - D:/SIPaKMeD/")
    print("  - C:/Users/David/Downloads/SIPaKMeD/")
    print()
    print("COLAB/LINUX:")
    print("  - /content/drive/MyDrive/SIPaKMeD/")
    print("  - /content/drive/MyDrive/cervical-cancer-sipakmed/")
    print()
    print("ESTRUCTURA ESPERADA:")
    print("  SIPaKMeD/")
    print("  ├── dyskeratotic/ (o class1/)")
    print("  ├── koilocytotic/ (o class2/)")
    print("  ├── metaplastic/ (o class3/)")
    print("  ├── parabasal/ (o class4/)")
    print("  └── superficial_intermediate/ (o class5/)")
    print()

def main():
    """Función principal"""
    print("INICIANDO SISTEMA SIPAKMED CON DATASET REAL")
    print("=" * 80)
    
    # Configurar TensorFlow
    setup_tensorflow()
    
    # Mostrar información sobre ubicación del dataset
    check_dataset_location()
    
    # Cargar dataset real
    print("\nCARGANDO DATASET REAL...")
    if not load_real_dataset():
        print("❌ No se pudo cargar el dataset real")
        print("🔄 ¿Quieres usar el dataset sintético en su lugar? (y/n): ")
        choice = input().lower().strip()
        
        if choice in ['y', 'yes', 'sí', 'si', '']:
            print("📝 Usando dataset sintético...")
            from src.data.dataset_creator import create_sample_dataset
            create_sample_dataset()
        else:
            print("❌ Deteniendo ejecución")
            return 1
    
    # Verificar dataset
    if not verify_dataset():
        print("❌ Dataset no válido")
        return 1
    
    # Definir modelos a entrenar
    modelos_sipakmed = [
        (MobileNetV2, mobilenet_preprocess, "MobileNetV2"),
        (ResNet50, resnet_preprocess, "ResNet50"),
        (EfficientNetB0, efficientnet_preprocess, "EfficientNetB0")
    ]
    
    # Entrenar todos los modelos
    print("\nINICIANDO ENTRENAMIENTO CON DATASET REAL...")
    resultados = []
    
    for i, (model_fn, preprocess_fn, name) in enumerate(modelos_sipakmed, 1):
        try:
            print(f"\n{'='*60}")
            print(f"ENTRENANDO [{i}/{len(modelos_sipakmed)}]: {name}")
            print(f"{'='*60}")
            
            start_time = time.time()
            resultado = train_cervical_model(model_fn, preprocess_fn, name)
            
            if resultado is not None:
                resultado['total_time'] = time.time() - start_time
                resultados.append(resultado)
                print(f"✓ {name} COMPLETADO")
                print(f"   Precision: {resultado['accuracy']:.4f}")
                print(f"   Tiempo: {resultado['total_time']:.1f}s")
            else:
                print(f"✗ {name} FALLO")
                
        except Exception as e:
            print(f"✗ Error critico con {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generar reportes finales
    if resultados:
        print(f"\nGENERANDO REPORTES FINALES...")
        best_model = generate_final_reports(resultados)
        
        print(f"\nENTRENAMIENTO COMPLETADO!")
        print(f"✓ Modelos exitosos: {len(resultados)}/{len(modelos_sipakmed)}")
        print(f"Mejor modelo: {best_model['name']} ({best_model['accuracy']:.4f})")
        
        # Mostrar resumen
        print(f"\nRESUMEN DE RESULTADOS:")
        for resultado in resultados:
            print(f"   {resultado['name']:15} | "
                  f"Acc: {resultado['accuracy']:.4f} | "
                  f"Tiempo: {resultado['total_time']:.1f}s")
        
        print(f"\nARCHIVOS GENERADOS:")
        print(f"   Modelos: {MODELS_DIR}")
        print(f"   Reportes: {REPORTS_DIR}")
        print(f"   Figuras: {FIGURES_DIR}")
        
        # Información adicional sobre el dataset usado
        print(f"\nDATASET UTILIZADO:")
        train_total = sum(len(list((PROCESSED_DATA_DIR / 'train' / class_name).glob('*.*'))) 
                         for class_name in REAL_CLASSES.keys() 
                         if (PROCESSED_DATA_DIR / 'train' / class_name).exists())
        val_total = sum(len(list((PROCESSED_DATA_DIR / 'val' / class_name).glob('*.*'))) 
                       for class_name in REAL_CLASSES.keys() 
                       if (PROCESSED_DATA_DIR / 'val' / class_name).exists())
        
        print(f"   Total imágenes: {train_total + val_total}")
        print(f"   Train: {train_total}, Val: {val_total}")
        print(f"   Fuente: Dataset real SIPaKMeD")
        
    else:
        print("✗ No se completo ningun entrenamiento exitosamente")
        return 1
    
    print(f"\nSISTEMA SIPAKMED LISTO PARA USAR!")
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nEntrenamiento interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\nError critico en main: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)