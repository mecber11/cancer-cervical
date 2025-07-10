# Copiar del artifact data_generators_py"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import *
from src.utils.preprocessing import preprocess_cervical_cell

def get_data_generators(preprocess_fn, model_name):
    """
    Crear generadores de datos para entrenamiento y validación
    
    Args:
        preprocess_fn: Función de preprocesamiento específica del modelo
        model_name: Nombre del modelo para logging
    
    Returns:
        tuple: (train_generator, val_generator)
    """
    train_dir = PROCESSED_DATA_DIR / 'train'
    val_dir = PROCESSED_DATA_DIR / 'val'
    
    def combined_preprocess(image):
        """Combinar preprocesamiento de células + modelo específico"""
        try:
            # Preprocesamiento específico para células cervicales
            enhanced = preprocess_cervical_cell(image)
            # Preprocesamiento específico del modelo (MobileNet, ResNet, etc.)
            return preprocess_fn(enhanced)
        except Exception as e:
            print(f"⚠️ Error en preprocesamiento: {e}")
            return preprocess_fn(image)
    
    # Generador de entrenamiento con augmentación
    train_generator = ImageDataGenerator(
        preprocessing_function=combined_preprocess,
        rotation_range=ROTATION_RANGE,
        zoom_range=ZOOM_RANGE,
        shear_range=SHEAR_RANGE,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=BRIGHTNESS_RANGE,
        width_shift_range=SHIFT_RANGE,
        height_shift_range=SHIFT_RANGE,
        fill_mode='nearest'
    )
    
    # Generador de validación sin augmentación
    val_generator = ImageDataGenerator(
        preprocessing_function=combined_preprocess
    )
    
    # Crear los flujos de datos
    train_data = train_generator.flow_from_directory(
        str(train_dir),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    val_data = val_generator.flow_from_directory(
        str(val_dir),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )
    
    print(f"📊 {model_name} - Train: {train_data.samples}, Val: {val_data.samples}")
    print(f"📋 Clases detectadas: {list(train_data.class_indices.keys())}")
    
    return train_data, val_data

def get_simple_generators(preprocess_fn):
    """
    Crear generadores simples sin augmentación para evaluación
    
    Args:
        preprocess_fn: Función de preprocesamiento del modelo
    
    Returns:
        tuple: (train_generator, val_generator)
    """
    train_dir = PROCESSED_DATA_DIR / 'train'
    val_dir = PROCESSED_DATA_DIR / 'val'
    
    def combined_preprocess(image):
        try:
            enhanced = preprocess_cervical_cell(image)
            return preprocess_fn(enhanced)
        except Exception as e:
            return preprocess_fn(image)
    
    # Solo preprocesamiento, sin augmentación
    generator = ImageDataGenerator(
        preprocessing_function=combined_preprocess
    )
    
    train_data = generator.flow_from_directory(
        str(train_dir),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    val_data = generator.flow_from_directory(
        str(val_dir),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_data, val_data

def preview_augmentation(class_name="dyskeratotic", num_samples=4):
    """
    Mostrar ejemplos de augmentación de datos
    
    Args:
        class_name: Clase a mostrar
        num_samples: Número de ejemplos
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    
    # Buscar una imagen de ejemplo
    class_dir = PROCESSED_DATA_DIR / 'train' / class_name
    images = list(class_dir.glob('*.jpg'))
    
    if not images:
        print(f"❌ No se encontraron imágenes en {class_dir}")
        return
    
    # Cargar imagen original
    img_path = images[0]
    original_img = load_img(str(img_path), target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(original_img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Crear generador con augmentación
    datagen = ImageDataGenerator(
        rotation_range=ROTATION_RANGE,
        zoom_range=ZOOM_RANGE,
        shear_range=SHEAR_RANGE,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=BRIGHTNESS_RANGE,
        width_shift_range=SHIFT_RANGE,
        height_shift_range=SHIFT_RANGE,
        fill_mode='nearest'
    )
    
    # Generar ejemplos
    plt.figure(figsize=(15, 4))
    
    # Imagen original
    plt.subplot(1, num_samples + 1, 1)
    plt.imshow(original_img)
    plt.title('Original')
    plt.axis('off')
    
    # Imágenes augmentadas
    i = 0
    for batch in datagen.flow(img_array, batch_size=1):
        plt.subplot(1, num_samples + 1, i + 2)
        augmented_img = batch[0].astype('uint8')
        plt.imshow(augmented_img)
        plt.title(f'Augmentada {i + 1}')
        plt.axis('off')
        
        i += 1
        if i >= num_samples:
            break
    
    plt.suptitle(f'Ejemplos de Augmentación - {CLASS_NAMES_FRIENDLY[class_name]}')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'augmentation_preview_{class_name}.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    # Previsualizar augmentación
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    
    print("🔍 Previsualizando augmentación...")
    preview_augmentation("dyskeratotic")
    preview_augmentation("koilocytotic")