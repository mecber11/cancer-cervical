"""
Verificador de archivos para la aplicación Streamlit
Verifica que todos los archivos necesarios estén presentes
"""

import os
from pathlib import Path

def verificar_estructura_aplicacion():
    """Verifica que todos los archivos necesarios estén presentes"""
    
    print("🔍 VERIFICANDO ESTRUCTURA DE LA APLICACIÓN")
    print("=" * 60)
    
    # Archivos principales
    archivos_principales = [
        "app.py",
        "requirements_streamlit.txt"
    ]
    
    # Modelos entrenados
    modelos_requeridos = [
        "data/models/sipakmed_MobileNetV2.h5",
        "data/models/sipakmed_ResNet50.h5", 
        "data/models/sipakmed_EfficientNetB0.h5"
    ]
    
    # Imágenes de entrenamiento
    imagenes_entrenamiento = [
        "reports/figures/confusion_matrix_MobileNetV2.png",
        "reports/figures/confusion_matrix_ResNet50.png",
        "reports/figures/confusion_matrix_EfficientNetB0.png",
        "reports/figures/training_history_MobileNetV2.png",
        "reports/figures/training_history_ResNet50.png", 
        "reports/figures/training_history_EfficientNetB0.png",
        "reports/figures/models_comparison.png"
    ]
    
    todos_ok = True
    
    # Verificar archivos principales
    print("\n📄 ARCHIVOS PRINCIPALES:")
    for archivo in archivos_principales:
        if Path(archivo).exists():
            print(f"   ✅ {archivo}")
        else:
            print(f"   ❌ {archivo} - FALTANTE")
            todos_ok = False
    
    # Verificar modelos
    print("\n🤖 MODELOS ENTRENADOS:")
    modelos_encontrados = 0
    for modelo in modelos_requeridos:
        if Path(modelo).exists():
            size_mb = Path(modelo).stat().st_size / (1024*1024)
            print(f"   ✅ {modelo} ({size_mb:.1f} MB)")
            modelos_encontrados += 1
        else:
            print(f"   ❌ {modelo} - FALTANTE")
            todos_ok = False
    
    # Verificar imágenes de entrenamiento
    print("\n📊 IMÁGENES DE ENTRENAMIENTO:")
    imagenes_encontradas = 0
    for imagen in imagenes_entrenamiento:
        if Path(imagen).exists():
            print(f"   ✅ {imagen}")
            imagenes_encontradas += 1
        else:
            print(f"   ❌ {imagen} - FALTANTE")
    
    # Resumen
    print(f"\n📋 RESUMEN:")
    print(f"   📄 Archivos principales: {len(archivos_principales)}/{len(archivos_principales)}")
    print(f"   🤖 Modelos: {modelos_encontrados}/{len(modelos_requeridos)}")
    print(f"   📊 Imágenes: {imagenes_encontradas}/{len(imagenes_entrenamiento)}")
    
    if todos_ok and modelos_encontrados == len(modelos_requeridos):
        print(f"\n🎉 ¡PERFECTO! Todos los archivos necesarios están presentes.")
        print(f"✅ La aplicación está lista para ejecutarse.")
        print(f"\n🚀 Para ejecutar: streamlit run app.py")
        return True
    else:
        print(f"\n⚠️ ARCHIVOS FALTANTES DETECTADOS")
        
        if modelos_encontrados < len(modelos_requeridos):
            print(f"\n❌ MODELOS FALTANTES:")
            print(f"   Los modelos deben generarse ejecutando:")
            print(f"   python main_real.py")
        
        if imagenes_encontradas < len(imagenes_entrenamiento):
            print(f"\n⚠️ IMÁGENES DE ENTRENAMIENTO FALTANTES:")
            print(f"   Las imágenes se generan automáticamente durante el entrenamiento.")
            print(f"   Algunas funcionalidades de visualización no estarán disponibles.")
            print(f"   Para generar todas las imágenes, ejecuta: python main_real.py")
        
        if modelos_encontrados == len(modelos_requeridos):
            print(f"\n✅ Los modelos están completos. Puedes ejecutar la aplicación:")
            print(f"   streamlit run app.py")
            print(f"   (Las imágenes faltantes no impiden el funcionamiento)")
        
        return False

def verificar_dependencias():
    """Verifica que las dependencias estén instaladas"""
    print(f"\n🔧 VERIFICANDO DEPENDENCIAS:")
    
    dependencias = [
        "streamlit",
        "plotly", 
        "opencv-python",
        "reportlab",
        "tensorflow",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "PIL"
    ]
    
    faltantes = []
    
    for dep in dependencias:
        try:
            if dep == "opencv-python":
                import cv2
            elif dep == "PIL":
                from PIL import Image
            else:
                __import__(dep)
            print(f"   ✅ {dep}")
        except ImportError:
            print(f"   ❌ {dep} - NO INSTALADO")
            faltantes.append(dep)
    
    if faltantes:
        print(f"\n❌ DEPENDENCIAS FALTANTES:")
        print(f"   Instala con: pip install {' '.join(faltantes)}")
        print(f"   O ejecuta: pip install -r requirements_streamlit.txt")
        return False
    else:
        print(f"\n✅ Todas las dependencias están instaladas.")
        return True

def main():
    print("🔬 VERIFICADOR DE APLICACIÓN STREAMLIT - CÉLULAS CERVICALES")
    print("=" * 80)
    
    # Verificar estructura
    estructura_ok = verificar_estructura_aplicacion()
    
    # Verificar dependencias
    deps_ok = verificar_dependencias()
    
    # Resultado final
    print(f"\n" + "=" * 80)
    
    if estructura_ok and deps_ok:
        print(f"🎉 ¡APLICACIÓN COMPLETAMENTE LISTA!")
        print(f"\n🚀 EJECUTAR APLICACIÓN:")
        print(f"   streamlit run app.py")
        print(f"\n🌐 Se abrirá en: http://localhost:8501")
        
    elif estructura_ok and not deps_ok:
        print(f"⚠️ APLICACIÓN PARCIALMENTE LISTA")
        print(f"✅ Archivos correctos")
        print(f"❌ Instalar dependencias faltantes")
        
    elif not estructura_ok and deps_ok:
        print(f"⚠️ APLICACIÓN PARCIALMENTE LISTA") 
        print(f"❌ Archivos faltantes")
        print(f"✅ Dependencias correctas")
        
    else:
        print(f"❌ APLICACIÓN NO LISTA")
        print(f"❌ Archivos y dependencias faltantes")
    
    print(f"\n📋 PRÓXIMOS PASOS:")
    if not deps_ok:
        print(f"1. pip install -r requirements_streamlit.txt")
    if not estructura_ok:
        print(f"2. python main_real.py  # Para generar modelos")
    if estructura_ok and deps_ok:
        print(f"1. streamlit run app.py")
        print(f"2. ¡Disfruta de tu aplicación médica!")

if __name__ == "__main__":
    main()
