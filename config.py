"""
Configuracion global del proyecto SIPaKMeD (Compatible Windows)
"""
import os
from pathlib import Path

# ============================================================================
# RUTAS DEL PROYECTO
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

# Crear directorios si no existen
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                  MODELS_DIR, REPORTS_DIR, FIGURES_DIR, METRICS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURACION SIPAKMED
# ============================================================================
SIPAKMED_USERNAME = "juliusdta"
SIPAKMED_TOKEN = "3611de84f34951034eef4a9ab943c79e"

# ============================================================================
# CLASES DEL DATASET
# ============================================================================
REAL_CLASSES = {
    "dyskeratotic": 0,
    "koilocytotic": 1,
    "metaplastic": 2,
    "parabasal": 3,
    "superficial_intermediate": 4
}

CLASS_NAMES_FRIENDLY = {
    "dyskeratotic": "Celulas Displasicas",
    "koilocytotic": "Celulas Koilocitoticas",
    "metaplastic": "Celulas Metaplasicas", 
    "parabasal": "Celulas Parabasales",
    "superficial_intermediate": "Celulas Superficiales-Intermedias"
}

# ============================================================================
# HIPERPARAMETROS DE ENTRENAMIENTO
# ============================================================================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

# Data augmentation
ROTATION_RANGE = 180
ZOOM_RANGE = 0.3
SHEAR_RANGE = 0.2
BRIGHTNESS_RANGE = (0.7, 1.3)
SHIFT_RANGE = 0.2

# Early stopping
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
MIN_LR = 1e-7

# ============================================================================
# CONFIGURACION DE HARDWARE
# ============================================================================
# Configurar GPU si esta disponible
GPU_MEMORY_GROWTH = True
MIXED_PRECISION = False  # Cambiar a True para entrenar mas rapido en GPU

# ============================================================================
# CONFIGURACION DE LOGGING
# ============================================================================
VERBOSE = 1  # 0: silencioso, 1: informacion basica, 2: informacion detallada

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================
def get_class_names():
    """Retorna lista de nombres de clases"""
    return list(REAL_CLASSES.keys())

def get_friendly_names():
    """Retorna lista de nombres amigables"""
    return list(CLASS_NAMES_FRIENDLY.values())

def get_model_path(model_name):
    """Retorna path completo para guardar modelo"""
    return MODELS_DIR / f"sipakmed_{model_name}.h5"

def get_report_path(model_name, extension="txt"):
    """Retorna path completo para guardar reporte"""
    return METRICS_DIR / f"sipakmed_{model_name}_report.{extension}"

def get_figure_path(filename):
    """Retorna path completo para guardar figura"""
    return FIGURES_DIR / filename