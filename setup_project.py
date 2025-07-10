"""
Setup Script - Configuracion automatica del proyecto SIPaKMeD (Compatible Windows)
"""

import os
import sys
from pathlib import Path

def create_project_structure():
    """Crear estructura completa del proyecto"""
    
    structure = {
        "src": {
            "__init__.py": '"""Modulo del proyecto SIPaKMeD"""',
            "data": {
                "__init__.py": '"""Modulo de datos"""',
                "dataset_creator.py": "# Copiar del artifact dataset_creator_py",
                "data_generators.py": "# Copiar del artifact data_generators_py"
            },
            "models": {
                "__init__.py": '"""Modulo de modelos"""',
                "model_builder.py": "# Copiar del artifact model_builder_py", 
                "trainer.py": "# Copiar del artifact trainer_py"
            },
            "utils": {
                "__init__.py": '"""Modulo de utilidades"""',
                "preprocessing.py": "# Copiar del artifact preprocessing_py",
                "visualization.py": "# Se creara automaticamente"
            },
            "evaluation": {
                "__init__.py": '"""Modulo de evaluacion"""',
                "evaluator.py": "# Se creara automaticamente"
            }
        },
        "data": {
            "raw": {},
            "processed": {},
            "models": {}
        },
        "reports": {
            "figures": {},
            "metrics": {}
        },
        "scripts": {
            "train_single_model.py": "# Script para entrenar un modelo",
            "train_all_models.py": "# Script para entrenar todos",
            "evaluate_models.py": "# Script para evaluacion"
        }
    }
    
    def create_dirs(base_path, structure):
        for name, content in structure.items():
            path = base_path / name
            
            if isinstance(content, dict):
                path.mkdir(exist_ok=True)
                create_dirs(path, content)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(content)
    
    project_root = Path.cwd()
    create_dirs(project_root, structure)
    
    print("✓ Estructura del proyecto creada")

def create_visualization_module():
    """Crear modulo de visualizacion basico"""
    
    visualization_code = '''"""
Visualization - Funciones de visualizacion y reportes
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from config import *

def plot_confusion_matrix(conf_matrix, class_names, model_name):
    """Graficar matriz de confusion"""
    
    plt.figure(figsize=(10, 8))
    
    # Nombres amigables para las clases
    friendly_names = [CLASS_NAMES_FRIENDLY[name] for name in class_names]
    
    sns.heatmap(conf_matrix, annot=True, fmt="d",
                xticklabels=friendly_names,
                yticklabels=friendly_names,
                cmap="Blues")
    
    plt.title(f"Matriz de Confusion - {model_name}", fontsize=16, fontweight='bold')
    plt.ylabel("Real", fontsize=14)
    plt.xlabel("Predicho", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Guardar figura
    save_path = get_figure_path(f"confusion_matrix_{model_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"Matriz de confusion guardada: {save_path}")

def plot_training_history(history, model_name):
    """Graficar historial de entrenamiento"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title(f'Accuracy - {model_name}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title(f'Loss - {model_name}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar figura
    save_path = get_figure_path(f"training_history_{model_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"Historial guardado: {save_path}")

def generate_final_reports(resultados):
    """Generar reportes finales comparativos"""
    
    if not resultados:
        return None
    
    # Extraer metricas
    accuracies = [r['accuracy'] for r in resultados]
    losses = [r['loss'] for r in resultados]
    times = [r['training_time'] for r in resultados]
    names = [r['name'] for r in resultados]
    
    # Grafico comparativo
    plt.figure(figsize=(15, 5))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'][:len(names)]
    
    # Accuracy
    plt.subplot(1, 3, 1)
    bars = plt.bar(names, accuracies, color=colors)
    plt.title('Precision por Modelo', fontweight='bold')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Loss
    plt.subplot(1, 3, 2)
    bars = plt.bar(names, losses, color=colors)
    plt.title('Perdida de Validacion', fontweight='bold')
    plt.ylabel('Loss')
    plt.xticks(rotation=45)
    for bar, loss in zip(bars, losses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{loss:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Tiempo
    plt.subplot(1, 3, 3)
    bars = plt.bar(names, times, color=colors)
    plt.title('Tiempo de Entrenamiento', fontweight='bold')
    plt.ylabel('Segundos')
    plt.xticks(rotation=45)
    for bar, time_val in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f'{time_val:.0f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Guardar comparacion
    save_path = get_figure_path("models_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Encontrar mejor modelo
    best_idx = np.argmax(accuracies)
    best_model = resultados[best_idx]
    
    # Guardar mejor modelo
    best_path = MODELS_DIR / "best_model.h5"
    best_model['model'].save(str(best_path))
    
    print(f"Mejor modelo guardado: {best_path}")
    print(f"Comparacion guardada: {save_path}")
    
    return best_model
'''
    
    viz_path = Path("src/utils/visualization.py")
    viz_path.parent.mkdir(parents=True, exist_ok=True)
    with open(viz_path, 'w', encoding='utf-8') as f:
        f.write(visualization_code)
    
    print("✓ Modulo de visualizacion creado")

def create_evaluator_module():
    """Crear modulo evaluador basico"""
    
    evaluator_code = '''"""
Evaluator - Funciones de evaluacion de modelos
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from config import *

def evaluate_all_models():
    """Evaluar todos los modelos entrenados"""
    
    print("Evaluando todos los modelos...")
    
    # Buscar modelos entrenados
    models = list(MODELS_DIR.glob("sipakmed_*.h5"))
    
    if not models:
        print("No se encontraron modelos entrenados")
        return
    
    for model_path in models:
        print(f"Evaluando {model_path.name}...")
        # Implementar evaluacion especifica segun necesidades
    
    print("✓ Evaluacion completada")

def load_and_evaluate_model(model_path, test_data=None):
    """Cargar y evaluar un modelo especifico"""
    
    try:
        model = tf.keras.models.load_model(str(model_path))
        print(f"✓ Modelo cargado: {model_path}")
        
        if test_data:
            predictions = model.predict(test_data)
            # Procesar predicciones
            
        return model
        
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return None
'''
    
    eval_path = Path("src/evaluation/evaluator.py")
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, 'w', encoding='utf-8') as f:
        f.write(evaluator_code)
    
    print("✓ Modulo evaluador creado")

def create_additional_scripts():
    """Crear scripts adicionales utiles"""
    
    # Script de entrenamiento individual
    train_single = '''#!/usr/bin/env python3
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
'''
    
    # Script de evaluacion
    evaluate_script = '''#!/usr/bin/env python3
"""
Script para evaluar modelos entrenados
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import *
from src.evaluation.evaluator import evaluate_all_models

def main():
    print("Iniciando evaluacion de modelos...")
    evaluate_all_models()

if __name__ == "__main__":
    main()
'''
    
    # Guardar scripts
    scripts_dir = Path("scripts")
    scripts_dir.mkdir(exist_ok=True)
    
    with open(scripts_dir / "train_single_model.py", 'w', encoding='utf-8') as f:
        f.write(train_single)
    
    with open(scripts_dir / "evaluate_models.py", 'w', encoding='utf-8') as f:
        f.write(evaluate_script)
    
    print("✓ Scripts adicionales creados")

def main():
    """Configuracion completa del proyecto"""
    
    print("CONFIGURANDO PROYECTO SIPAKMED")
    print("=" * 50)
    
    try:
        create_project_structure()
        create_visualization_module()
        create_evaluator_module()
        create_additional_scripts()
        
        print("\n✓ PROYECTO CONFIGURADO EXITOSAMENTE!")
        print("\nPROXIMOS PASOS:")
        print("1. Copiar el contenido de los artifacts en los archivos correspondientes")
        print("2. pip install -r requirements.txt")
        print("3. python main.py")
        
        print("\nESTRUCTURA CREADA:")
        print("   src/          - Codigo fuente")
        print("   data/         - Datos del proyecto")
        print("   reports/      - Reportes y figuras")
        print("   scripts/      - Scripts de utilidad")
        
        print("\nIMPORTANTE:")
        print("Debes copiar manualmente el contenido de los artifacts:")
        print("- dataset_creator_py -> src/data/dataset_creator.py")
        print("- data_generators_py -> src/data/data_generators.py")
        print("- preprocessing_py -> src/utils/preprocessing.py")
        print("- model_builder_py -> src/models/model_builder.py")
        print("- trainer_py -> src/models/trainer.py")
        
    except Exception as e:
        print(f"Error configurando proyecto: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)