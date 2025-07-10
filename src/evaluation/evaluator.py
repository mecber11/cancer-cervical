"""
Evaluator - Funciones de evaluación de modelos con pruebas estadísticas inferenciales
Incluye Matthews Correlation Coefficient (MCC) y prueba de McNemar
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
from scipy.stats import chi2
import pandas as pd
import json
from pathlib import Path
from config import *

def calculate_matthews_correlation_coefficient(y_true, y_pred):
    """
    Calcula el Matthews Correlation Coefficient (MCC) para clasificación multiclase
    
    MCC toma en cuenta verdaderos y falsos positivos y negativos
    Rango: [-1, 1] donde 1 es predicción perfecta, 0 es aleatoria, -1 es desacuerdo total
    """
    return matthews_corrcoef(y_true, y_pred)

def perform_mcnemar_test(y_true, pred_model1, pred_model2):
    """
    Realiza la prueba de McNemar para comparar dos modelos
    
    Args:
        y_true: Etiquetas verdaderas
        pred_model1: Predicciones del modelo 1
        pred_model2: Predicciones del modelo 2
        
    Returns:
        dict: Estadístico de prueba, p-valor e interpretación
    """
    # Crear tabla de contingencia 2x2
    # a: ambos modelos correctos
    # b: modelo 1 correcto, modelo 2 incorrecto
    # c: modelo 1 incorrecto, modelo 2 correcto
    # d: ambos modelos incorrectos
    
    correct_m1 = (pred_model1 == y_true)
    correct_m2 = (pred_model2 == y_true)
    
    a = np.sum(correct_m1 & correct_m2)
    b = np.sum(correct_m1 & ~correct_m2)
    c = np.sum(~correct_m1 & correct_m2)
    d = np.sum(~correct_m1 & ~correct_m2)
    
    # Prueba de McNemar con corrección de continuidad
    if b + c == 0:
        # Si no hay discordancias, los modelos son idénticos
        return {
            'statistic': 0,
            'p_value': 1.0,
            'b': b,
            'c': c,
            'significant': False,
            'interpretation': 'Los modelos tienen exactamente las mismas predicciones'
        }
    
    # Estadístico de McNemar con corrección de continuidad de Yates
    statistic = (abs(b - c) - 1) ** 2 / (b + c) if b + c > 0 else 0
    
    # P-valor usando distribución chi-cuadrado con 1 grado de libertad
    p_value = 1 - chi2.cdf(statistic, df=1)
    
    # Interpretación
    alpha = 0.05
    significant = p_value < alpha
    
    if significant:
        if b > c:
            interpretation = f"Modelo 1 es significativamente mejor (p={p_value:.4f})"
        else:
            interpretation = f"Modelo 2 es significativamente mejor (p={p_value:.4f})"
    else:
        interpretation = f"No hay diferencia significativa entre modelos (p={p_value:.4f})"
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'b': int(b),  # Modelo 1 correcto, Modelo 2 incorrecto
        'c': int(c),  # Modelo 1 incorrecto, Modelo 2 correcto
        'significant': significant,
        'interpretation': interpretation
    }

def save_model_predictions(model_name, y_true, y_pred, predictions_proba):
    """
    Guarda las predicciones del modelo para análisis posterior
    """
    predictions_dir = DATA_DIR / 'predictions'
    predictions_dir.mkdir(exist_ok=True)
    
    # Guardar como numpy arrays
    np.save(predictions_dir / f'{model_name}_y_true.npy', y_true)
    np.save(predictions_dir / f'{model_name}_y_pred.npy', y_pred)
    np.save(predictions_dir / f'{model_name}_proba.npy', predictions_proba)
    
    print(f"✓ Predicciones guardadas para {model_name}")

def load_model_predictions(model_name):
    """
    Carga las predicciones guardadas de un modelo
    """
    predictions_dir = DATA_DIR / 'predictions'
    
    try:
        y_true = np.load(predictions_dir / f'{model_name}_y_true.npy')
        y_pred = np.load(predictions_dir / f'{model_name}_y_pred.npy')
        predictions_proba = np.load(predictions_dir / f'{model_name}_proba.npy')
        
        return y_true, y_pred, predictions_proba
    except FileNotFoundError:
        print(f"⚠️ No se encontraron predicciones para {model_name}")
        return None, None, None

def calculate_all_mcc_scores():
    """
    Calcula MCC para todos los modelos entrenados
    """
    models = ['MobileNetV2', 'ResNet50', 'EfficientNetB0']
    mcc_scores = {}
    
    for model_name in models:
        y_true, y_pred, _ = load_model_predictions(model_name)
        
        if y_true is not None:
            mcc = calculate_matthews_correlation_coefficient(y_true, y_pred)
            mcc_scores[model_name] = mcc
            print(f"{model_name}: MCC = {mcc:.4f}")
        else:
            print(f"⚠️ No se pudo calcular MCC para {model_name}")
    
    return mcc_scores

def perform_all_mcnemar_tests():
    """
    Realiza pruebas de McNemar entre todos los pares de modelos
    """
    models = ['MobileNetV2', 'ResNet50', 'EfficientNetB0']
    results = {}
    
    # Cargar predicciones de todos los modelos
    predictions = {}
    for model_name in models:
        y_true, y_pred, _ = load_model_predictions(model_name)
        if y_true is not None:
            predictions[model_name] = (y_true, y_pred)
    
    # Realizar pruebas entre pares
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i < j and model1 in predictions and model2 in predictions:
                y_true1, y_pred1 = predictions[model1]
                y_true2, y_pred2 = predictions[model2]
                
                # Asegurar que las etiquetas verdaderas son las mismas
                assert np.array_equal(y_true1, y_true2), "Las etiquetas verdaderas deben ser idénticas"
                
                result = perform_mcnemar_test(y_true1, y_pred1, y_pred2)
                key = f"{model1}_vs_{model2}"
                results[key] = result
                
                print(f"\n{model1} vs {model2}:")
                print(f"  - Estadístico: {result['statistic']:.4f}")
                print(f"  - P-valor: {result['p_value']:.4f}")
                print(f"  - {result['interpretation']}")
    
    return results

def generate_statistical_report():
    """
    Genera un reporte completo con análisis estadístico inferencial
    """
    print("\n=== ANÁLISIS ESTADÍSTICO INFERENCIAL ===\n")
    
    # 1. Calcular MCC para todos los modelos
    print("1. Matthews Correlation Coefficient (MCC):")
    print("-" * 40)
    mcc_scores = calculate_all_mcc_scores()
    
    # 2. Realizar pruebas de McNemar
    print("\n2. Pruebas de McNemar entre modelos:")
    print("-" * 40)
    mcnemar_results = perform_all_mcnemar_tests()
    
    # 3. Convertir resultados a tipos nativos de Python para JSON
    # Convertir MCC scores
    mcc_scores_json = {k: float(v) for k, v in mcc_scores.items()}
    
    # Convertir resultados de McNemar
    mcnemar_results_json = {}
    for key, result in mcnemar_results.items():
        mcnemar_results_json[key] = {
            'statistic': float(result['statistic']),
            'p_value': float(result['p_value']),
            'b': int(result['b']),
            'c': int(result['c']),
            'significant': bool(result['significant']),  # Convertir numpy.bool_ a bool
            'interpretation': str(result['interpretation'])
        }
    
    # 3. Guardar resultados
    results = {
        'mcc_scores': mcc_scores_json,
        'mcnemar_tests': mcnemar_results_json
    }
    
    # Guardar como JSON
    report_path = REPORTS_DIR / 'statistical_analysis.json'
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✓ Reporte estadístico guardado en: {report_path}")
    
    # 4. Crear tabla resumen
    create_summary_tables(mcc_scores, mcnemar_results)
    
    return results

def create_summary_tables(mcc_scores, mcnemar_results):
    """
    Crea tablas resumen para visualización
    """
    # Tabla de MCC
    print("\n📊 RESUMEN - Matthews Correlation Coefficient:")
    print("-" * 50)
    mcc_df = pd.DataFrame(list(mcc_scores.items()), columns=['Modelo', 'MCC'])
    mcc_df = mcc_df.sort_values('MCC', ascending=False)
    print(mcc_df.to_string(index=False))
    
    # Interpretación de MCC
    print("\nInterpretación MCC:")
    print("  • MCC > 0.5: Muy buena concordancia")
    print("  • MCC 0.3-0.5: Buena concordancia")
    print("  • MCC 0.0-0.3: Concordancia débil")
    print("  • MCC = 0: No mejor que aleatorio")
    
    # Tabla de McNemar
    print("\n📊 RESUMEN - Pruebas de McNemar (α = 0.05):")
    print("-" * 80)
    
    for comparison, result in mcnemar_results.items():
        models = comparison.split('_vs_')
        print(f"\n{models[0]} vs {models[1]}:")
        print(f"  • Casos donde solo {models[0]} acierta: {result['b']}")
        print(f"  • Casos donde solo {models[1]} acierta: {result['c']}")
        print(f"  • P-valor: {result['p_value']:.4f}")
        print(f"  • Conclusión: {result['interpretation']}")

def evaluate_single_model(model_path, test_data=None):
    """
    Evalúa un modelo específico incluyendo MCC
    """
    try:
        model = tf.keras.models.load_model(str(model_path))
        model_name = Path(model_path).stem.replace('sipakmed_', '')
        print(f"✓ Modelo cargado: {model_name}")
        
        if test_data:
            # Realizar predicciones
            predictions_proba = model.predict(test_data)
            y_pred = np.argmax(predictions_proba, axis=1)
            y_true = test_data.classes
            
            # Calcular métricas
            accuracy = np.mean(y_pred == y_true)
            mcc = calculate_matthews_correlation_coefficient(y_true, y_pred)
            
            # Guardar predicciones
            save_model_predictions(model_name, y_true, y_pred, predictions_proba)
            
            print(f"  • Accuracy: {accuracy:.4f}")
            print(f"  • MCC: {mcc:.4f}")
            
            return {
                'model': model,
                'accuracy': accuracy,
                'mcc': mcc,
                'y_true': y_true,
                'y_pred': y_pred,
                'predictions_proba': predictions_proba
            }
            
    except Exception as e:
        print(f"Error evaluando modelo: {e}")
        return None

def evaluate_all_models():
    """
    Evalúa todos los modelos entrenados y realiza análisis estadístico
    """
    print("🔍 Evaluando todos los modelos con análisis estadístico...")
    
    # Buscar modelos entrenados
    models = list(MODELS_DIR.glob("sipakmed_*.h5"))
    
    if not models:
        print("❌ No se encontraron modelos entrenados")
        return
    
    # Si ya existen predicciones guardadas, generar reporte estadístico
    predictions_exist = all(
        (DATA_DIR / 'predictions' / f'{name}_y_pred.npy').exists() 
        for name in ['MobileNetV2', 'ResNet50', 'EfficientNetB0']
    )
    
    if predictions_exist:
        print("✓ Predicciones encontradas. Generando análisis estadístico...")
        generate_statistical_report()
    else:
        print("⚠️ No se encontraron predicciones guardadas.")
        print("   Ejecuta el entrenamiento con trainer.py actualizado para guardar predicciones.")
    
    print("\n✓ Evaluación completada")

if __name__ == "__main__":
    # Ejecutar evaluación completa
    evaluate_all_models()