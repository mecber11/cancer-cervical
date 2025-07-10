"""
Preprocessing - Funciones de preprocesamiento de imagenes medicas (CORREGIDO)
"""

import cv2
import numpy as np
from config import *

def preprocess_cervical_cell(image):
    """
    Preprocesamiento especifico para celulas cervicales (CORREGIDO)
    
    Aplica:
    - Mejora de contraste CLAHE en espacio LAB
    - Filtro bilateral para reducir ruido
    - Normalizacion de iluminacion
    
    Args:
        image: Imagen en formato RGB (numpy array)
        
    Returns:
        Imagen preprocesada en formato RGB
    """
    try:
        # Asegurar que la imagen este en el formato correcto
        if len(image.shape) != 3 or image.shape[2] != 3:
            return image
        
        # Convertir a uint8 si no lo esta
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        # Convertir a LAB para mejor control de luminancia
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # CORRECCION: Asegurar que el canal L sea uint8
        l = l.astype(np.uint8)
        
        # Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Reconstruir imagen
        limg = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        
        # Aplicar filtro bilateral para suavizar preservando bordes
        enhanced_img = cv2.bilateralFilter(enhanced_img, 9, 75, 75)
        
        return enhanced_img
        
    except Exception as e:
        # Si hay error, retornar imagen original sin procesamiento
        return image

def enhance_nuclei_visibility(image):
    """
    Realzar la visibilidad de nucleos celulares
    
    Args:
        image: Imagen en formato RGB
        
    Returns:
        Imagen con nucleos realzados
    """
    try:
        # Asegurar formato correcto
        if len(image.shape) != 3 or image.shape[2] != 3:
            return image
            
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        # Convertir a HSV para trabajar con saturacion
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # Realzar saturacion para destacar nucleos
        s = cv2.multiply(s, 1.2)
        s = np.clip(s, 0, 255).astype(np.uint8)
        
        # Reconstruir imagen
        enhanced_hsv = cv2.merge((h, s, v))
        enhanced_img = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2RGB)
        
        return enhanced_img
        
    except Exception as e:
        return image

def normalize_staining(image):
    """
    Normalizar tincion de imagenes citologicas
    
    Args:
        image: Imagen en formato RGB
        
    Returns:
        Imagen con tincion normalizada
    """
    try:
        # Asegurar formato correcto
        if len(image.shape) != 3 or image.shape[2] != 3:
            return image
            
        # Normalización por percentiles para tincion estable
        image_float = image.astype(np.float32)
        
        for channel in range(3):
            channel_data = image_float[:, :, channel]
            
            # Calcular percentiles
            p_low = np.percentile(channel_data, 1)
            p_high = np.percentile(channel_data, 99)
            
            # Normalizar entre percentiles
            if p_high > p_low:
                channel_data = (channel_data - p_low) / (p_high - p_low) * 255
                channel_data = np.clip(channel_data, 0, 255)
                image_float[:, :, channel] = channel_data
        
        return image_float.astype(np.uint8)
        
    except Exception as e:
        return image

def remove_artifacts(image):
    """
    Remover artefactos comunes en imagenes citologicas
    
    Args:
        image: Imagen en formato RGB
        
    Returns:
        Imagen limpia
    """
    try:
        # Asegurar formato correcto
        if len(image.shape) != 3 or image.shape[2] != 3:
            return image
            
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        # Convertir a escala de grises para deteccion de artefactos
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detectar y rellenar regiones muy oscuras (posibles artefactos)
        _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        
        # Dilatar mascara para incluir bordes
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Aplicar inpainting para rellenar artefactos
        cleaned = cv2.inpaint(image, 255 - mask, 3, cv2.INPAINT_TELEA)
        
        return cleaned
        
    except Exception as e:
        return image

def preprocess_comprehensive(image):
    """
    Preprocesamiento completo combinando todas las tecnicas
    
    Args:
        image: Imagen en formato RGB
        
    Returns:
        Imagen completamente preprocesada
    """
    try:
        # Aplicar pipeline completo
        processed = remove_artifacts(image)
        processed = normalize_staining(processed)
        processed = enhance_nuclei_visibility(processed)
        processed = preprocess_cervical_cell(processed)
        
        return processed
        
    except Exception as e:
        return image

def simple_preprocess(image):
    """
    Preprocesamiento simple sin CLAHE (para evitar errores)
    
    Args:
        image: Imagen en formato RGB
        
    Returns:
        Imagen preprocesada
    """
    try:
        # Asegurar formato correcto
        if len(image.shape) != 3 or image.shape[2] != 3:
            return image
            
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        # Solo aplicar filtro bilateral (sin CLAHE)
        enhanced_img = cv2.bilateralFilter(image, 9, 75, 75)
        
        return enhanced_img
        
    except Exception as e:
        return image

def preview_preprocessing_steps(image_path, save_figures=True):
    """
    Mostrar los pasos del preprocesamiento
    
    Args:
        image_path: Ruta a la imagen
        save_figures: Si guardar las figuras
    """
    import matplotlib.pyplot as plt
    
    # Cargar imagen
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Aplicar cada paso
    steps = [
        ("Original", image),
        ("Sin Artefactos", remove_artifacts(image)),
        ("Tincion Normalizada", normalize_staining(image)),
        ("Nucleos Realzados", enhance_nuclei_visibility(image)),
        ("CLAHE + Bilateral", preprocess_cervical_cell(image)),
        ("Simple (Sin CLAHE)", simple_preprocess(image))
    ]
    
    # Crear figura
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, (title, img) in enumerate(steps):
        axes[i].imshow(img)
        axes[i].set_title(title, fontsize=12, fontweight='bold')
        axes[i].axis('off')
    
    plt.suptitle('Pasos del Preprocesamiento de Celulas Cervicales', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_figures:
        plt.savefig(FIGURES_DIR / 'preprocessing_steps.png', dpi=300, bbox_inches='tight')
        print(f"Figura guardada en {FIGURES_DIR / 'preprocessing_steps.png'}")
    
    plt.show()

if __name__ == "__main__":
    # Ejemplo de uso
    from pathlib import Path
    
    # Buscar imagen de ejemplo
    train_dir = PROCESSED_DATA_DIR / 'train'
    example_img = None
    
    for class_dir in train_dir.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob('*.jpg'))
            if images:
                example_img = images[0]
                break
    
    if example_img:
        print(f"Previsualizando preprocesamiento con: {example_img}")
        preview_preprocessing_steps(example_img)
    else:
        print("No se encontraron imagenes de ejemplo")