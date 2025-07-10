import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization,
    Input, Concatenate, Conv2D, MaxPooling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from config import *

def build_cervical_model(base_model_fn, preprocess_fn, name):
    """
    Construir modelo completo para clasificación de células cervicales
    
    Args:
        base_model_fn: Función constructora del modelo base
        preprocess_fn: Función de preprocesamiento 
        name: Nombre del modelo
        
    Returns:
        Modelo compilado de Keras
    """
    # Crear modelo base pre-entrenado
    base_model = base_model_fn(
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        weights='imagenet'
    )
    
    # Congelar el modelo base inicialmente
    base_model.trainable = False
    
    # Construir cabeza de clasificación
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='input_image')
    
    # Aplicar modelo base
    x = base_model(inputs, training=False)
    
    # Pooling global
    x = GlobalAveragePooling2D(name='global_average_pooling')(x)
    
    # Capas densas con regularización
    x = Dense(512, activation='relu', 
              kernel_regularizer=l2(0.001),
              name='dense_512')(x)
    x = BatchNormalization(name='bn_1')(x)
    x = Dropout(0.5, name='dropout_1')(x)
    
    x = Dense(256, activation='relu',
              kernel_regularizer=l2(0.001), 
              name='dense_256')(x)
    x = BatchNormalization(name='bn_2')(x)
    x = Dropout(0.3, name='dropout_2')(x)
    
    # Capa de salida
    outputs = Dense(len(REAL_CLASSES), 
                   activation='softmax',
                   name='predictions')(x)
    
    # Crear modelo
    model = Model(inputs, outputs, name=f'CervicalNet_{name}')
    
    return model

def build_custom_cnn():
    """
    Construir CNN personalizada para células cervicales
    
    Returns:
        Modelo CNN personalizado
    """
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='input')
    
    # Bloque 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Bloque 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Bloque 3
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Bloque 4
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    
    # Clasificador
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(len(REAL_CLASSES), activation='softmax')(x)
    
    model = Model(inputs, outputs, name='CustomCervicalCNN')
    return model

def build_ensemble_model(model_paths):
    """
    Construir modelo ensemble a partir de modelos entrenados
    
    Args:
        model_paths: Lista de rutas a modelos entrenados
        
    Returns:
        Modelo ensemble
    """
    # Cargar modelos base
    models = []
    for path in model_paths:
        model = keras.models.load_model(str(path))
        # Remover la última capa softmax
        model = Model(inputs=model.input, 
                     outputs=model.layers[-2].output)
        models.append(model)
    
    # Input común
    input_layer = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Obtener outputs de todos los modelos
    outputs = []
    for i, model in enumerate(models):
        model._name = f'model_{i}'
        output = model(input_layer)
        outputs.append(output)
    
    # Concatenar features
    if len(outputs) > 1:
        combined = Concatenate(name='ensemble_concat')(outputs)
    else:
        combined = outputs[0]
    
    # Capa de fusión
    x = Dense(512, activation='relu', name='fusion_dense')(combined)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Salida final
    final_output = Dense(len(REAL_CLASSES), 
                        activation='softmax', 
                        name='ensemble_output')(x)
    
    ensemble_model = Model(inputs=input_layer, 
                          outputs=final_output, 
                          name='CervicalEnsemble')
    
    return ensemble_model

def build_attention_model(base_model_fn, name):
    """
    Construir modelo con mecanismo de atención
    
    Args:
        base_model_fn: Función del modelo base
        name: Nombre del modelo
        
    Returns:
        Modelo con atención
    """
    # Modelo base
    base_model = base_model_fn(
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        weights='imagenet'
    )
    base_model.trainable = False
    
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Features del modelo base
    features = base_model(inputs, training=False)
    
    # Mecanismo de atención espacial
    attention = Conv2D(1, (1, 1), activation='sigmoid', 
                      name='attention_map')(features)
    
    # Aplicar atención
    attended_features = tf.multiply(features, attention, 
                                   name='attended_features')
    
    # Pooling global
    pooled = GlobalAveragePooling2D()(attended_features)
    
    # Clasificador
    x = Dense(512, activation='relu', 
              kernel_regularizer=l2(0.001))(pooled)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, activation='relu',
              kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(len(REAL_CLASSES), activation='softmax')(x)
    
    model = Model(inputs, outputs, name=f'AttentionNet_{name}')
    
    return model

def get_model_summary(model):
    """
    Obtener resumen detallado del modelo
    
    Args:
        model: Modelo de Keras
        
    Returns:
        dict: Información del modelo
    """
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) 
                           for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    # Calcular memoria aproximada (MB)
    memory_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes por parámetro
    
    return {
        'name': model.name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'memory_mb': memory_mb,
        'layers': len(model.layers),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape
    }

def compare_architectures():
    """
    Comparar diferentes arquitecturas de modelos
    """
    from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
    
    models_to_compare = [
        (MobileNetV2, "MobileNetV2"),
        (ResNet50, "ResNet50"), 
        (EfficientNetB0, "EfficientNetB0"),
        (build_custom_cnn, "CustomCNN")
    ]
    
    print("📊 COMPARACIÓN DE ARQUITECTURAS")
    print("=" * 80)
    
    results = []
    for model_fn, name in models_to_compare:
        try:
            if name == "CustomCNN":
                model = model_fn()
            else:
                model = build_cervical_model(model_fn, lambda x: x, name)
            
            summary = get_model_summary(model)
            results.append(summary)
            
            print(f"\n🏗️ {name}:")
            print(f"   Parámetros totales: {summary['total_params']:,}")
            print(f"   Parámetros entrenables: {summary['trainable_params']:,}")
            print(f"   Memoria aprox: {summary['memory_mb']:.1f} MB")
            print(f"   Capas: {summary['layers']}")
            
        except Exception as e:
            print(f"❌ Error con {name}: {e}")
    
    return results

if __name__ == "__main__":
    print("🧪 Probando construcción de modelos...")
    compare_architectures()