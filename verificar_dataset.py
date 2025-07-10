"""
Verificador rápido de dataset SIPaKMeD
"""

import os
from pathlib import Path

def verificar_dataset_sipakmed():
    """Verificar si el dataset SIPaKMeD está disponible"""
    
    print("🔍 BUSCANDO DATASET SIPAKMED...")
    print("=" * 50)
    
    # Rutas típicas donde puede estar el dataset
    rutas_posibles = [
        "C:/Users/David/Google Drive/SIPaKMeD",
        "C:/Users/David/OneDrive/SIPaKMeD",
        "D:/SIPaKMeD",
        "C:/Users/David/Downloads/SIPaKMeD",
        "C:/Users/David/Documents/SIPaKMeD",
        "C:/Users/David/Desktop/SIPaKMeD",
        "C:/SIPaKMeD",
        # Variaciones de nombres
        "C:/Users/David/Google Drive/sipakmed",
        "C:/Users/David/Google Drive/cervical-cancer-sipakmed",
        "C:/Users/David/Google Drive/cervical_cancer",
        "C:/Users/David/OneDrive/sipakmed",
    ]
    
    dataset_encontrado = False
    
    for ruta in rutas_posibles:
        path = Path(ruta)
        if path.exists():
            print(f"✅ ENCONTRADO: {ruta}")
            
            # Buscar imágenes
            imagenes = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                imagenes.extend(list(path.rglob(ext)))
            
            print(f"   📊 Total imágenes: {len(imagenes)}")
            
            # Buscar carpetas de clases
            clases_esperadas = ['dyskeratotic', 'koilocytotic', 'metaplastic', 'parabasal', 'superficial']
            clases_alternativas = ['class1', 'class2', 'class3', 'class4', 'class5']
            
            carpetas = [d.name.lower() for d in path.iterdir() if d.is_dir()]
            
            print(f"   📁 Carpetas encontradas: {carpetas}")
            
            clases_encontradas = 0
            for clase in clases_esperadas + clases_alternativas:
                if any(clase in carpeta for carpeta in carpetas):
                    clases_encontradas += 1
            
            print(f"   🎯 Clases detectadas: {clases_encontradas}/5")
            
            if len(imagenes) > 100 and clases_encontradas >= 3:
                print(f"   ✅ DATASET VÁLIDO")
                dataset_encontrado = True
            else:
                print(f"   ⚠️ Dataset incompleto")
            
            print()
    
    if not dataset_encontrado:
        print("❌ DATASET NO ENCONTRADO")
        print("\nRutas buscadas:")
        for ruta in rutas_posibles:
            print(f"   - {ruta}")
        
        print("\n🔍 BUSCA MANUALMENTE:")
        print("1. Abre el Explorador de Windows")
        print("2. Busca carpetas con nombres como:")
        print("   - dyskeratotic")
        print("   - koilocytotic") 
        print("   - metaplastic")
        print("   - parabasal")
        print("   - superficial")
        print("3. O busca archivos .jpg con 'sipakmed' en el nombre")
        
        print("\n📝 INGRESA RUTA MANUAL:")
        ruta_manual = input("Ruta completa a tu dataset (Enter para omitir): ").strip()
        
        if ruta_manual and Path(ruta_manual).exists():
            print(f"✅ Ruta válida: {ruta_manual}")
            return ruta_manual
        else:
            print("❌ Continuando con dataset sintético...")
            return None
    
    return dataset_encontrado

def main():
    print("🧬 VERIFICADOR DE DATASET SIPAKMED")
    print("=" * 60)
    
    resultado = verificar_dataset_sipakmed()
    
    if resultado:
        print("\n🎉 ¡PERFECTO!")
        print("Tu dataset SIPaKMeD está listo para usar.")
        print("\nPara entrenar con datos reales:")
        print("1. Crea el archivo src/data/dataset_real_loader.py")
        print("2. Crea el archivo main_real.py") 
        print("3. Ejecuta: python main_real.py")
        
    else:
        print("\n⚠️ DATASET NO ENCONTRADO")
        print("Opciones:")
        print("1. Buscar manualmente tu dataset")
        print("2. Descargar dataset SIPaKMeD oficial")
        print("3. Continuar con dataset sintético (python main.py)")

if __name__ == "__main__":
    main()
