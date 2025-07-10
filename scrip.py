"""
Script para ejecutar el análisis estadístico completo
Incluye cálculo de MCC y pruebas de McNemar
"""

import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent))

from src.evaluation.evaluator import (
    evaluate_all_models,
    generate_statistical_report,
    calculate_all_mcc_scores,
    perform_all_mcnemar_tests
)

def main():
    """Ejecuta el análisis estadístico completo"""
    
    print("=" * 80)
    print("🔬 ANÁLISIS ESTADÍSTICO DE MODELOS - CÉLULAS CERVICALES")
    print("=" * 80)
    
    print("\nEste script realizará:")
    print("1. Cálculo del Matthews Correlation Coefficient (MCC)")
    print("2. Pruebas de McNemar entre todos los pares de modelos")
    print("3. Generación de reporte estadístico completo")
    
    print("\n" + "=" * 80)
    
    # Verificar si existen las predicciones
    predictions_dir = Path("data/predictions")
    if not predictions_dir.exists():
        print("❌ ERROR: No se encontró la carpeta de predicciones.")
        print("   Primero debes entrenar los modelos con el trainer actualizado.")
        print("\n📝 PASOS A SEGUIR:")
        print("1. Actualiza trainer.py con las modificaciones proporcionadas")
        print("2. Ejecuta: python main_real.py")
        print("3. Luego ejecuta este script nuevamente")
        return
    
    # Verificar predicciones individuales
    models = ['MobileNetV2', 'ResNet50', 'EfficientNetB0']
    missing_predictions = []
    
    for model in models:
        pred_file = predictions_dir / f"{model}_y_pred.npy"
        if not pred_file.exists():
            missing_predictions.append(model)
    
    if missing_predictions:
        print(f"⚠️ ADVERTENCIA: Faltan predicciones para: {', '.join(missing_predictions)}")
        print("   Necesitas reentrenar estos modelos con el trainer actualizado.")
        
        # Preguntar si continuar con los modelos disponibles
        response = input("\n¿Continuar con los modelos disponibles? (s/n): ")
        if response.lower() != 's':
            print("❌ Análisis cancelado.")
            return
    
    try:
        # Ejecutar análisis estadístico
        print("\n🔄 Ejecutando análisis estadístico...")
        results = generate_statistical_report()
        
        if results:
            print("\n✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
            
            # Mostrar resumen de resultados
            print("\n📊 RESUMEN DE RESULTADOS:")
            print("-" * 50)
            
            # MCC scores
            mcc_scores = results.get('mcc_scores', {})
            if mcc_scores:
                print("\nMatthews Correlation Coefficient:")
                for model, mcc in sorted(mcc_scores.items(), key=lambda x: x[1], reverse=True):
                    print(f"  • {model}: {mcc:.4f}")
            
            # McNemar más significativo
            mcnemar_tests = results.get('mcnemar_tests', {})
            if mcnemar_tests:
                print("\nPruebas de McNemar más significativas:")
                significant_tests = [
                    (k, v) for k, v in mcnemar_tests.items() 
                    if v['significant']
                ]
                
                if significant_tests:
                    for comparison, result in significant_tests:
                        models = comparison.replace('_vs_', ' vs ')
                        print(f"  • {models}: p={result['p_value']:.4f} - {result['interpretation']}")
                else:
                    print("  • No se encontraron diferencias significativas entre modelos")
            
            print("\n📁 ARCHIVOS GENERADOS:")
            print(f"  • reports/statistical_analysis.json")
            print(f"  • Resultados integrados en app.py")
            
            print("\n🚀 PRÓXIMOS PASOS:")
            print("1. Ejecuta la aplicación: streamlit run app.py")
            print("2. Ve a la pestaña 'Análisis Estadístico' en los resultados")
            print("3. El PDF incluirá automáticamente las estadísticas")
            
        else:
            print("❌ Error durante el análisis estadístico")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()