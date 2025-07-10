#!/usr/bin/env python3
"""
Script para verificar todas las versiones instaladas de las dependencias del proyecto
"""
import subprocess
import sys
import importlib
from datetime import datetime

print("=" * 60)
print("VERIFICADOR DE VERSIONES DEL PROYECTO")
print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)
print()

# Diccionario de paquetes con sus nombres de importación
packages = {
    'PHP': {'type': 'command', 'command': ['php', '-v']},
    'Python': {'type': 'system'},
    'streamlit': {'type': 'module', 'import_name': 'streamlit'},
    'plotly': {'type': 'module', 'import_name': 'plotly'},
    'opencv-python': {'type': 'module', 'import_name': 'cv2'},
    'reportlab': {'type': 'module', 'import_name': 'reportlab'},
    'Pillow': {'type': 'module', 'import_name': 'PIL'},
    'pandas': {'type': 'module', 'import_name': 'pandas'},
    'numpy': {'type': 'module', 'import_name': 'numpy'},
    'matplotlib': {'type': 'module', 'import_name': 'matplotlib'},
    'seaborn': {'type': 'module', 'import_name': 'seaborn'},
    'tensorflow': {'type': 'module', 'import_name': 'tensorflow'}
}

# Función para obtener versión de módulos Python
def get_module_version(module_name, import_name=None):
    try:
        if import_name:
            module = importlib.import_module(import_name)
        else:
            module = importlib.import_module(module_name)
        
        # Intentar diferentes atributos de versión
        for attr in ['__version__', 'VERSION', 'version']:
            if hasattr(module, attr):
                version = getattr(module, attr)
                if isinstance(version, str):
                    return version
                elif hasattr(version, '__str__'):
                    return str(version)
        return "Versión no disponible"
    except ImportError:
        return "No instalado"
    except Exception as e:
        return f"Error: {str(e)}"

# Función para obtener versión de comandos del sistema
def get_command_version(command):
    try:
        result = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
        # Para PHP, obtener solo la primera línea
        if command[0] == 'php':
            return result.split('\n')[0].replace('PHP ', '')
        return result.strip()
    except subprocess.CalledProcessError:
        return "No instalado o no en PATH"
    except FileNotFoundError:
        return "No instalado o no en PATH"
    except Exception as e:
        return f"Error: {str(e)}"

# Verificar todas las dependencias
print("VERSIONES INSTALADAS:")
print("-" * 60)

max_name_length = max(len(name) for name in packages.keys())

for package_name, info in packages.items():
    if info['type'] == 'command':
        version = get_command_version(info['command'])
    elif info['type'] == 'system':
        version = f"{sys.version.split()[0]} ({sys.version.split('(')[1].split(')')[0]})"
    else:  # module
        import_name = info.get('import_name', package_name)
        version = get_module_version(package_name, import_name)
    
    # Formatear salida
    print(f"{package_name:<{max_name_length}} : {version}")

print("-" * 60)

# Verificar si hay paquetes no instalados
print("\nRESUMEN:")
not_installed = []
installed = []

for package_name, info in packages.items():
    if info['type'] == 'module':
        import_name = info.get('import_name', package_name)
        version = get_module_version(package_name, import_name)
        if "No instalado" in version:
            not_installed.append(package_name)
        else:
            installed.append(package_name)

print(f"✓ Paquetes instalados: {len(installed)}")
print(f"✗ Paquetes no instalados: {len(not_installed)}")

if not_installed:
    print(f"\nPaquetes faltantes: {', '.join(not_installed)}")
    print("\nPara instalar los paquetes faltantes, ejecuta:")
    print(f"pip install {' '.join(not_installed)}")

# Información adicional del sistema
print("\n" + "=" * 60)
print("INFORMACIÓN ADICIONAL DEL SISTEMA:")
print("-" * 60)
print(f"Sistema Operativo: {sys.platform}")
print(f"Versión de pip: ", end="")
try:
    pip_version = subprocess.check_output([sys.executable, '-m', 'pip', '--version'], text=True).split()[1]
    print(pip_version)
except:
    print("No disponible")

# Verificar GPU para TensorFlow
print("\nGPU para TensorFlow: ", end="")
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ {len(gpus)} GPU(s) disponible(s)")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
    else:
        print("✗ No se detectaron GPUs")
except:
    print("TensorFlow no instalado o error al verificar")

print("=" * 60)