"""
Trainer - Entrenamiento de modelos para clasificacion de celulas cervicales (CORREGIDO)
"""

import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

from config import *
from src.models.model_builder import build_cervical_model
from src.data.data_generators import get_data_generators
from src.utils.visualization import plot_confusion_matrix, plot_training_history
from src.evaluation.evaluator import save_model_predictions, calculate_matthews_correlation_coefficient

def train_cervical_model(model_fn, preprocess_fn, model_name):
    """
    Entrenar un modelo para clasificacion de celulas cervicales
    
    Args:
        model_fn: Funcion constructora del modelo base (MobileNetV2, ResNet50, etc.)
        preprocess_fn: Funcion de preprocesamiento especifica del modelo
        model_name: Nombre del modelo para logging y guardado
        
    Returns:
        dict: Resultados del entrenamiento incluyendo modelo, metricas, etc.
    """
    print(f"\nEntrenando {model_name}...")
    
    try:
        # 1. Construir modelo
        print("Construyendo modelo...")
        model = build_cervical_model(model_fn, preprocess_fn, model_name)
        
        # 2. Compilar modelo
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Arquitectura del modelo:")
        print(f"   - Parametros totales: {model.count_params():,}")
        print(f"   - Parametros entrenables: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
        
        # 3. Obtener generadores de datos
        print("Preparando datos...")
        train_data, val_data = get_data_generators(preprocess_fn, model_name)
        
        # 4. Configurar callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=REDUCE_LR_PATIENCE,
                min_lr=MIN_LR,
                verbose=1,
                mode='min'
            )
        ]
        
        # 5. Entrenar modelo (SIN use_multiprocessing y workers)
        print(f"Iniciando entrenamiento ({EPOCHS} epochs maximo)...")
        start_time = time.time()
        
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=VERBOSE
        )
        
        training_time = time.time() - start_time
        print(f"Entrenamiento completado en {training_time:.1f} segundos")
        
        # 6. Guardar modelo
        model_path = get_model_path(model_name)
        model.save(str(model_path))
        print(f"Modelo guardado: {model_path}")
        
          # 7. Evaluacion final
        print("Evaluando modelo...")
        val_data.reset()
        
        # Predicciones
        predictions = model.predict(val_data, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        y_true = val_data.classes
        
        # NUEVO: Guardar predicciones para análisis estadístico
        save_model_predictions(model_name, y_true, y_pred, predictions)
        
        # NUEVO: Calcular MCC
        mcc_score = calculate_matthews_correlation_coefficient(y_true, y_pred)
        print(f"Matthews Correlation Coefficient: {mcc_score:.4f}")
        
        # Nombres de clases en el orden correcto
        class_names = list(train_data.class_indices.keys())
        
        # Reporte de clasificacion
        report = classification_report(
            y_true, y_pred, 
            target_names=class_names, 
            output_dict=True,
            zero_division=0
        )
        
        # Agregar MCC al reporte
        report['mcc'] = mcc_score
        
        # Predicciones
        predictions = model.predict(val_data, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        y_true = val_data.classes
        
        # Nombres de clases en el orden correcto
        class_names = list(train_data.class_indices.keys())
        
        # Reporte de clasificacion
        report = classification_report(
            y_true, y_pred, 
            target_names=class_names, 
            output_dict=True,
            zero_division=0
        )
        
        # Matriz de confusion
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # 8. Guardar reportes
        _save_training_report(model_name, report, training_time, history)
        
        # 9. Generar visualizaciones
        plot_confusion_matrix(conf_matrix, class_names, model_name)
        plot_training_history(history, model_name)
        
        # 10. Preparar resultados
        final_accuracy = report['accuracy']
        final_loss = history.history['val_loss'][-1]
        
        result = {
            'name': model_name,
            'model': model,
            'history': history,
            'accuracy': final_accuracy,
            'loss': final_loss,
            'report': report,
            'confusion_matrix': conf_matrix,
            'training_time': training_time,
            'class_names': class_names,
            'model_path': str(model_path)
        }
        
        print(f"✓ {model_name} completado - Precision: {final_accuracy:.4f}")
        return result
        
    except Exception as e:
        print(f"✗ Error entrenando {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def _save_training_report(model_name, report, training_time, history):
    """
    Guardar reporte detallado del entrenamiento
    """
    report_path = get_report_path(model_name, "txt")
    
    # Calcular metricas adicionales
    best_epoch = len(history.history['loss'])
    best_val_acc = max(history.history['val_accuracy'])
    best_val_loss = min(history.history['val_loss'])
    
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(f"=== REPORTE DE ENTRENAMIENTO - {model_name} ===\n\n")
        f.write(f"Configuracion:\n")
        f.write(f"  - Epochs totales: {best_epoch}\n")
        f.write(f"  - Batch size: {BATCH_SIZE}\n")
        f.write(f"  - Learning rate: {LEARNING_RATE}\n")
        f.write(f"  - Imagen size: {IMG_SIZE}x{IMG_SIZE}\n\n")
        
        f.write(f"Resultados:\n")
        f.write(f"  - Tiempo entrenamiento: {training_time:.1f}s\n")
        f.write(f"  - Mejor accuracy validacion: {best_val_acc:.4f}\n")
        f.write(f"  - Mejor loss validacion: {best_val_loss:.4f}\n")
        f.write(f"  - Accuracy final: {report['accuracy']:.4f}\n\n")
        
        f.write(f"Reporte por clase:\n")
        for class_name in get_class_names():
            if class_name in report:
                metrics = report[class_name]
                friendly_name = CLASS_NAMES_FRIENDLY[class_name]
                f.write(f"  {friendly_name}:\n")
                f.write(f"    - Precision: {metrics['precision']:.4f}\n")
                f.write(f"    - Recall: {metrics['recall']:.4f}\n")
                f.write(f"    - F1-score: {metrics['f1-score']:.4f}\n")
                f.write(f"    - Soporte: {metrics['support']}\n\n")
        
        f.write(f"Metricas generales:\n")
        f.write(f"  - Macro avg precision: {report['macro avg']['precision']:.4f}\n")
        f.write(f"  - Macro avg recall: {report['macro avg']['recall']:.4f}\n")
        f.write(f"  - Macro avg f1-score: {report['macro avg']['f1-score']:.4f}\n")
        f.write(f"  - Weighted avg precision: {report['weighted avg']['precision']:.4f}\n")
        f.write(f"  - Weighted avg recall: {report['weighted avg']['recall']:.4f}\n")
        f.write(f"  - Weighted avg f1-score: {report['weighted avg']['f1-score']:.4f}\n")
    
    print(f"Reporte guardado: {report_path}")

def fine_tune_model(model_path, train_data, val_data, epochs=5):
    """
    Fine-tuning de un modelo pre-entrenado
    
    Args:
        model_path: Ruta al modelo guardado
        train_data: Generador de datos de entrenamiento
        val_data: Generador de datos de validacion
        epochs: Numero de epochs para fine-tuning
    """
    print(f"Iniciando fine-tuning...")
    
    # Cargar modelo
    model = tf.keras.models.load_model(str(model_path))
    
    # Descongelar algunas capas del modelo base
    base_model = model.layers[1]  # Asumiendo que es la segunda capa
    base_model.trainable = True
    
    # Congelar las primeras capas, entrenar solo las ultimas
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # Recompilar con learning rate mas bajo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Entrenar (SIN argumentos problematicos)
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        verbose=VERBOSE
    )
    
    # Guardar modelo fine-tuned
    fine_tuned_path = model_path.replace('.h5', '_finetuned.h5')
    model.save(fine_tuned_path)
    
    print(f"✓ Fine-tuning completado: {fine_tuned_path}")
    return model, history

if __name__ == "__main__":
    # Ejemplo de entrenamiento individual
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    
    print("Entrenamiento de prueba - MobileNetV2")
    result = train_cervical_model(MobileNetV2, preprocess_input, "MobileNetV2_test")
    
    if result:
        print(f"✓ Prueba exitosa: {result['accuracy']:.4f}")