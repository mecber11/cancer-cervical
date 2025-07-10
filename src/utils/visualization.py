"""
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
