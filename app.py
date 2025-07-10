import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import io
import base64
from datetime import datetime
import json
from pathlib import Path

# Configuración de la página
st.set_page_config(
    page_title="🔬 Clasificador de Células Cervicales - SIPaKMeD",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados mejorados y más profesionales
st.markdown("""
<style>
    /* Importar fuentes profesionales */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Variables de color */
    :root {
        --primary-color: #0066CC;
        --secondary-color: #6C63FF;
        --success-color: #00D25B;
        --warning-color: #FFAB00;
        --danger-color: #FC424A;
        --dark-color: #191C24;
        --light-bg: #F5F7FA;
        --card-bg: #FFFFFF;
        --text-primary: #2D3748;
        --text-secondary: #718096;
    }
    
    /* Estilos generales */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        padding: 2.5rem 0;
        text-align: center;
        margin: -3rem -3rem 2rem -3rem;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        text-align: center;
        color: var(--text-secondary);
        font-size: 1.1rem;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    /* Cards profesionales */
    .professional-card {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        margin-bottom: 1.5rem;
    }
    
    .professional-card:hover {
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    /* Métricas estilizadas */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-container:hover {
        transform: scale(1.02);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sección de resultados mejorada */
    .results-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 20px;
        padding: 3rem;
        margin: 2rem 0;
    }
    
    /* Badges de estado */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 30px;
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-normal {
        background: rgba(0, 210, 91, 0.1);
        color: var(--success-color);
        border: 2px solid var(--success-color);
    }
    
    .status-warning {
        background: rgba(255, 171, 0, 0.1);
        color: var(--warning-color);
        border: 2px solid var(--warning-color);
    }
    
    .status-danger {
        background: rgba(252, 66, 74, 0.1);
        color: var(--danger-color);
        border: 2px solid var(--danger-color);
    }
    
    /* Botones personalizados */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    /* Sidebar personalizado */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        font-weight: 700;
        font-size: 1.2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .sidebar-section {
        background: rgba(255,255,255,0.05);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .sidebar-section h3 {
        color: white;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Info boxes mejorados */
    .info-box-professional {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin: 1.5rem 0;
    }
    
    .warning-box-professional {
        background: linear-gradient(135deg, #FA8BFF 0%, #2BD2FF 90%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        margin: 1.5rem 0;
    }
    
    /* Tablas mejoradas */
    .dataframe {
        border: none !important;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    /* Progress bars */
    .progress-container {
        background: #E0E0E0;
        border-radius: 10px;
        height: 10px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    /* Animaciones */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fadeIn {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
            padding: 1.5rem 0;
        }
        
        .professional-card {
            padding: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Configuración de rutas de modelos
MODEL_PATH = "data/models"
IMG_SIZE = 224

# Definición de clases de células cervicales
CLASS_NAMES = [
    "dyskeratotic",
    "koilocytotic", 
    "metaplastic",
    "parabasal",
    "superficial_intermediate"
]

CLASS_NAMES_FRIENDLY = {
    "dyskeratotic": "Células Displásicas",
    "koilocytotic": "Células Koilocitóticas",
    "metaplastic": "Células Metaplásicas", 
    "parabasal": "Células Parabasales",
    "superficial_intermediate": "Células Superficiales-Intermedias"
}

# Información clínica de cada tipo de célula
CLINICAL_INFO = {
    "dyskeratotic": {
        "descripcion": "Células con alteraciones displásicas que pueden indicar cambios precancerosos.",
        "significado": "Requiere seguimiento médico y posibles estudios adicionales.",
        "color": "#FC424A",
        "riesgo": "Alto",
        "icon": "🔴"
    },
    "koilocytotic": {
        "descripcion": "Células con cambios citopáticos característicos de infección por VPH.",
        "significado": "Indica presencia de virus del papiloma humano (VPH).",
        "color": "#FFAB00",
        "riesgo": "Moderado",
        "icon": "🟠"
    },
    "metaplastic": {
        "descripcion": "Células de la zona de transformación cervical en proceso de cambio.",
        "significado": "Proceso normal de reparación, generalmente benigno.",
        "color": "#0066CC",
        "riesgo": "Bajo",
        "icon": "🟡"
    },
    "parabasal": {
        "descripcion": "Células de las capas profundas del epitelio cervical.",
        "significado": "Parte normal del epitelio cervical estratificado.",
        "color": "#00D25B",
        "riesgo": "Normal",
        "icon": "🟢"
    },
    "superficial_intermediate": {
        "descripcion": "Células de las capas superficiales e intermedias del epitelio.",
        "significado": "Células maduras normales del epitelio cervical.",
        "color": "#00D25B", 
        "riesgo": "Normal",
        "icon": "🟢"
    }
}

@st.cache_resource
def load_models():
    """Carga los modelos entrenados de SIPaKMeD"""
    models = {}
    model_files = {
        "MobileNetV2": "sipakmed_MobileNetV2.h5",
        "ResNet50": "sipakmed_ResNet50.h5", 
        "EfficientNetB0": "sipakmed_EfficientNetB0.h5"
    }

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (name, filename) in enumerate(model_files.items()):
        model_path = os.path.join(MODEL_PATH, filename)
        status_text.text(f'Cargando modelo {name}...')
        
        if os.path.exists(model_path):
            try:
                models[name] = load_model(model_path)
                progress_bar.progress((i + 1) / len(model_files))
            except Exception as e:
                st.error(f"❌ Error cargando {name}: {str(e)}")
        else:
            st.warning(f"⚠️ No se encontró el archivo: {model_path}")
    
    progress_bar.empty()
    status_text.empty()
    
    return models

def load_training_images():
    """Carga las imágenes generadas durante el entrenamiento"""
    figures_path = "reports/figures"
    training_images = {}
    
    # Definir las imágenes esperadas
    image_files = {
        "confusion_matrices": [
            "confusion_matrix_MobileNetV2.png",
            "confusion_matrix_ResNet50.png", 
            "confusion_matrix_EfficientNetB0.png"
        ],
        "training_histories": [
            "training_history_MobileNetV2.png",
            "training_history_ResNet50.png",
            "training_history_EfficientNetB0.png"
        ],
        "model_comparison": "models_comparison.png"
    }
    
    # Verificar y cargar imágenes
    for category, files in image_files.items():
        if isinstance(files, list):
            training_images[category] = []
            for file in files:
                file_path = os.path.join(figures_path, file)
                if os.path.exists(file_path):
                    training_images[category].append({
                        'path': file_path,
                        'name': file.replace('.png', '').replace('_', ' ').title(),
                        'model': file.split('_')[-1].replace('.png', '')
                    })
        else:
            file_path = os.path.join(figures_path, files)
            if os.path.exists(file_path):
                training_images[category] = {
                    'path': file_path,
                    'name': files.replace('.png', '').replace('_', ' ').title()
                }
    
    return training_images

def load_statistical_results():
    """Carga los resultados del análisis estadístico si existen"""
    try:
        stats_path = Path("reports/statistical_analysis.json")
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error cargando estadísticas: {e}")
    return None

def display_statistical_analysis(statistical_results):
    """Muestra el análisis estadístico inferencial (MCC y McNemar)"""
    st.markdown("""
    <div class="results-section animate-fadeIn">
        <h2 style="text-align: center; color: var(--text-primary); margin-bottom: 1rem; font-weight: 700;">
            📊 Análisis Estadístico Inferencial
        </h2>
        <p style="text-align: center; color: var(--text-secondary); font-size: 1.1rem; margin-bottom: 2rem;">
            Evaluación rigurosa de los modelos mediante pruebas estadísticas
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if not statistical_results:
        st.warning("⚠️ No se encontraron resultados estadísticos. Ejecuta el análisis completo primero.")
        return
    
    # Tabs para organizar los resultados
    tab1, tab2 = st.tabs(["📈 Matthews Correlation Coefficient", "🔬 Prueba de McNemar"])
    
    with tab1:
        st.markdown("### 📊 Matthews Correlation Coefficient (MCC)")
        st.markdown("""
        <div class="info-box-professional">
            <h4 style="margin-bottom: 1rem;">ℹ️ ¿Qué es el MCC?</h4>
            <p style="margin: 0;">
                El MCC es una medida de calidad para clasificaciones que considera verdaderos y falsos positivos/negativos.
                Es especialmente útil para datasets desbalanceados. Rango: [-1, 1]
            </p>
            <ul style="margin-top: 0.5rem; padding-left: 1.5rem;">
                <li><strong>MCC = 1</strong>: Predicción perfecta</li>
                <li><strong>MCC > 0.5</strong>: Muy buena concordancia</li>
                <li><strong>MCC = 0</strong>: No mejor que aleatorio</li>
                <li><strong>MCC = -1</strong>: Desacuerdo total</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Mostrar MCC scores
        mcc_scores = statistical_results.get('mcc_scores', {})
        if mcc_scores:
            # Crear DataFrame y ordenar por MCC
            mcc_df = pd.DataFrame(list(mcc_scores.items()), columns=['Modelo', 'MCC'])
            mcc_df = mcc_df.sort_values('MCC', ascending=False)
            
            # Visualización con barras horizontales
            fig_mcc = go.Figure()
            
            # Colores según el valor de MCC
            colors_mcc = []
            for mcc in mcc_df['MCC']:
                if mcc > 0.5:
                    colors_mcc.append('#00D25B')  # Verde
                elif mcc > 0.3:
                    colors_mcc.append('#FFAB00')  # Amarillo
                else:
                    colors_mcc.append('#FC424A')  # Rojo
            
            fig_mcc.add_trace(go.Bar(
                y=mcc_df['Modelo'],
                x=mcc_df['MCC'],
                orientation='h',
                marker_color=colors_mcc,
                text=[f'{mcc:.4f}' for mcc in mcc_df['MCC']],
                textposition='outside'
            ))
            
            fig_mcc.update_layout(
                title='Matthews Correlation Coefficient por Modelo',
                xaxis_title='MCC',
                yaxis_title='Modelo',
                xaxis_range=[-0.1, 1.1],
                height=400,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            
            fig_mcc.add_vline(x=0.5, line_dash="dash", line_color="gray", 
                             annotation_text="Muy buena concordancia", annotation_position="top")
            
            st.plotly_chart(fig_mcc, use_container_width=True)
            
            # Tabla de resultados
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🏆 Ranking de Modelos por MCC")
                mcc_display = mcc_df.copy()
                mcc_display['MCC'] = mcc_display['MCC'].apply(lambda x: f"{x:.4f}")
                mcc_display['Interpretación'] = mcc_df['MCC'].apply(
                    lambda x: '⭐ Excelente' if x > 0.5 else '✅ Bueno' if x > 0.3 else '⚠️ Regular'
                )
                st.dataframe(mcc_display, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("### 🔬 Prueba de McNemar")
        st.markdown("""
        <div class="info-box-professional">
            <h4 style="margin-bottom: 1rem;">ℹ️ ¿Qué es la prueba de McNemar?</h4>
            <p style="margin: 0;">
                La prueba de McNemar compara el rendimiento de dos modelos evaluando las predicciones discordantes.
                Es útil para determinar si un modelo es significativamente mejor que otro.
            </p>
            <ul style="margin-top: 0.5rem; padding-left: 1.5rem;">
                <li><strong>p < 0.05</strong>: Diferencia estadísticamente significativa</li>
                <li><strong>p ≥ 0.05</strong>: No hay diferencia significativa</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Mostrar resultados de McNemar
        mcnemar_results = statistical_results.get('mcnemar_tests', {})
        if mcnemar_results:
            # Crear matriz de p-valores
            models = ['MobileNetV2', 'ResNet50', 'EfficientNetB0']
            p_matrix = np.ones((3, 3))
            
            # Mapeo de resultados a matriz
            model_indices = {model: i for i, model in enumerate(models)}
            
            for comparison, result in mcnemar_results.items():
                model1, model2 = comparison.replace('_vs_', ' ').split()
                if model1 in model_indices and model2 in model_indices:
                    i, j = model_indices[model1], model_indices[model2]
                    p_matrix[i, j] = result['p_value']
                    p_matrix[j, i] = result['p_value']
            
            # Crear heatmap
            fig_mcnemar = go.Figure(data=go.Heatmap(
                z=p_matrix,
                x=models,
                y=models,
                text=[[f'{p:.4f}' if i != j else '-' for j, p in enumerate(row)] for i, row in enumerate(p_matrix)],
                texttemplate='%{text}',
                colorscale='RdYlGn_r',
                zmin=0,
                zmax=0.1,
                colorbar=dict(title="p-valor")
            ))
            
            fig_mcnemar.update_layout(
                title='Matriz de p-valores (Prueba de McNemar)',
                xaxis_title='Modelo',
                yaxis_title='Modelo',
                height=500
            )
            
            st.plotly_chart(fig_mcnemar, use_container_width=True)
            
            # Detalles de cada comparación
            st.markdown("#### 📋 Comparaciones Detalladas")
            
            for comparison, result in mcnemar_results.items():
                models_compared = comparison.replace('_vs_', ' vs ')
                
                # Determinar el color del contenedor según significancia
                if result['significant']:
                    container_style = "background: rgba(252, 66, 74, 0.1); border-left: 4px solid #FC424A;"
                else:
                    container_style = "background: rgba(0, 210, 91, 0.1); border-left: 4px solid #00D25B;"
                
                st.markdown(f"""
                <div class="professional-card" style="{container_style} margin-bottom: 1rem;">
                    <h5 style="color: var(--text-primary); margin-bottom: 0.5rem;">{models_compared}</h5>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                        <div>
                            <p style="margin: 0.2rem 0;"><strong>Estadístico χ²:</strong> {result['statistic']:.4f}</p>
                            <p style="margin: 0.2rem 0;"><strong>p-valor:</strong> {result['p_value']:.4f}</p>
                        </div>
                        <div>
                            <p style="margin: 0.2rem 0;"><strong>Solo Modelo 1 acierta:</strong> {result['b']} casos</p>
                            <p style="margin: 0.2rem 0;"><strong>Solo Modelo 2 acierta:</strong> {result['c']} casos</p>
                        </div>
                    </div>
                    <p style="margin-top: 0.5rem; font-weight: 600; color: {'#FC424A' if result['significant'] else '#00D25B'};">
                        📊 {result['interpretation']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
    """Carga las imágenes generadas durante el entrenamiento"""
    figures_path = "reports/figures"
    training_images = {}
    
    image_files = {
        "confusion_matrices": [
            "confusion_matrix_MobileNetV2.png",
            "confusion_matrix_ResNet50.png", 
            "confusion_matrix_EfficientNetB0.png"
        ],
        "training_histories": [
            "training_history_MobileNetV2.png",
            "training_history_ResNet50.png",
            "training_history_EfficientNetB0.png"
        ],
        "model_comparison": "models_comparison.png"
    }
    
    for category, files in image_files.items():
        if isinstance(files, list):
            training_images[category] = []
            for file in files:
                file_path = os.path.join(figures_path, file)
                if os.path.exists(file_path):
                    training_images[category].append({
                        'path': file_path,
                        'name': file.replace('.png', '').replace('_', ' ').title(),
                        'model': file.split('_')[-1].replace('.png', '')
                    })
        else:
            file_path = os.path.join(figures_path, files)
            if os.path.exists(file_path):
                training_images[category] = {
                    'path': file_path,
                    'name': files.replace('.png', '').replace('_', ' ').title()
                }
    
    return training_images

def display_training_results():
    """Muestra los resultados del entrenamiento realizado"""
    st.markdown("""
    <div class="results-section animate-fadeIn">
        <h2 style="text-align: center; color: var(--text-primary); margin-bottom: 1rem; font-weight: 700;">
            📊 Resultados del Entrenamiento
        </h2>
        <p style="text-align: center; color: var(--text-secondary); font-size: 1.1rem; margin-bottom: 2rem;">
            Análisis completo del entrenamiento con <strong>5,015 imágenes</strong> del dataset SIPaKMeD
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Métricas de entrenamiento en cards profesionales
    col1, col2, col3 = st.columns(3)
    
    training_metrics = {
        "MobileNetV2": {"accuracy": "84.18%", "time": "17.8 min", "params": "3.0M", "color": "#667eea"},
        "ResNet50": {"accuracy": "89.95%", "time": "27.9 min", "params": "24.8M", "color": "#00D25B"},
        "EfficientNetB0": {"accuracy": "86.07%", "time": "15.6 min", "params": "4.8M", "color": "#6C63FF"}
    }
    
    for i, (model_name, metrics) in enumerate(training_metrics.items()):
        with [col1, col2, col3][i]:
            st.markdown(f"""
            <div class="professional-card">
                <div style="background: linear-gradient(135deg, {metrics['color']} 0%, {metrics['color']}dd 100%); 
                            color: white; padding: 1.5rem; border-radius: 12px; margin: -1rem -1rem 1rem -1rem;">
                    <h3 style="margin: 0; font-weight: 700; font-size: 1.3rem;">{model_name}</h3>
                </div>
                <div style="text-align: center; padding: 1rem 0;">
                    <div class="metric-value" style="color: {metrics['color']}; font-size: 2.5rem;">
                        {metrics['accuracy']}
                    </div>
                    <div style="color: var(--text-secondary); margin-top: 1rem;">
                        <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                            <span>⏱️ Tiempo:</span>
                            <strong>{metrics['time']}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                            <span>🔧 Parámetros:</span>
                            <strong>{metrics['params']}</strong>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Cargar y mostrar imágenes de entrenamiento
    training_images = load_training_images()
    
    # Cargar resultados estadísticos si existen
    statistical_results = load_statistical_results()
    
    if training_images:
        # Tabs para organizar mejor las imágenes
        tab_list = ["📈 Comparación General", "🎯 Matrices de Confusión", "📉 Historiales de Entrenamiento", "📊 Dataset Info"]
        if statistical_results:
            tab_list.append("📊 Análisis Estadístico")
        
        tabs = st.tabs(tab_list)
        
        with tabs[0]:
            if 'model_comparison' in training_images:
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    try:
                        comparison_img = Image.open(training_images['model_comparison']['path'])
                        st.image(comparison_img, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error cargando imagen: {e}")
        
        with tabs[1]:
            if 'confusion_matrices' in training_images and training_images['confusion_matrices']:
                # Mostrar matrices de confusión una por una para mejor visualización
                st.markdown("##### 🎯 Matrices de Confusión por Modelo")
                
                for i, img_info in enumerate(training_images['confusion_matrices']):
                    with st.expander(f"📊 {img_info['model']}", expanded=(i==0)):
                        try:
                            conf_img = Image.open(img_info['path'])
                            col1, col2, col3 = st.columns([1, 3, 1])
                            with col2:
                                st.image(conf_img, caption=f"Matriz de Confusión - {img_info['model']}", use_container_width=True)
                        except Exception as e:
                            st.error(f"Error cargando matriz de confusión: {e}")
        
        with tabs[2]:
            if 'training_histories' in training_images and training_images['training_histories']:
                # Mostrar historiales de entrenamiento con mejor layout
                st.markdown("##### 📉 Evolución del Entrenamiento")
                
                # Crear dos columnas para mejor distribución
                for i in range(0, len(training_images['training_histories']), 2):
                    cols = st.columns(2)
                    
                    for j, col in enumerate(cols):
                        if i + j < len(training_images['training_histories']):
                            img_info = training_images['training_histories'][i + j]
                            with col:
                                try:
                                    hist_img = Image.open(img_info['path'])
                                    st.image(hist_img, caption=f"{img_info['model']}", use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error: {e}")
        
        with tabs[3]:
            # Información del dataset de manera visual
            st.markdown("##### 📊 Dataset SIPaKMeD - Estadísticas")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="metric-container" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                    <div class="metric-label">TOTAL IMÁGENES</div>
                    <div class="metric-value">5,015</div>
                    <div style="font-size: 0.9rem; opacity: 0.9;">Dataset completo</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-container" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                    <div class="metric-label">ENTRENAMIENTO</div>
                    <div class="metric-value">4,010</div>
                    <div style="font-size: 0.9rem; opacity: 0.9;">80% del dataset</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-container" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                    <div class="metric-label">VALIDACIÓN</div>
                    <div class="metric-value">1,005</div>
                    <div style="font-size: 0.9rem; opacity: 0.9;">20% del dataset</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Distribución por clases
            st.markdown("##### 📊 Distribución por Tipo de Célula")
            
            class_distribution = pd.DataFrame({
                'Tipo de Célula': list(CLASS_NAMES_FRIENDLY.values()),
                'Cantidad': [1003, 1003, 1003, 1003, 1003],  # Ajusta estos valores según tu dataset real
                'Porcentaje': ['20%', '20%', '20%', '20%', '20%']
            })
            
            st.dataframe(class_distribution, use_container_width=True, hide_index=True)
        
        # Tab de análisis estadístico si existe
        if statistical_results and len(tabs) > 4:
            with tabs[4]:
                display_statistical_analysis(statistical_results)

def enhance_cervical_cell_image(image):
    """Mejora específica para imágenes de células cervicales"""
    try:
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
            
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        if l.dtype == np.uint8:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
        
        enhanced = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return enhanced
        
    except Exception as e:
        st.warning(f"Error en mejora de imagen: {e}")
        return np.array(image) if isinstance(image, Image.Image) else image

def preprocess_image(image, model_name):
    """Preprocesa la imagen según el modelo específico"""
    if isinstance(image, Image.Image):
        image_array = np.array(image.convert('RGB'))
    else:
        image_array = image

    enhanced_image = enhance_cervical_cell_image(image_array)
    image_resized = cv2.resize(enhanced_image, (IMG_SIZE, IMG_SIZE))
    image_final = np.array(image_resized, dtype=np.float32)
    image_expanded = np.expand_dims(image_final, axis=0)

    if model_name == "MobileNetV2":
        return mobilenet_preprocess(image_expanded)
    elif model_name == "ResNet50":
        return resnet_preprocess(image_expanded)
    elif model_name == "EfficientNetB0":
        return efficientnet_preprocess(image_expanded)
    else:
        return image_expanded / 255.0

def predict_cervical_cells(image, models):
    """Realiza predicciones con todos los modelos disponibles"""
    predictions = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (model_name, model) in enumerate(models.items()):
        try:
            status_text.text(f'Analizando con {model_name}...')
            processed_image = preprocess_image(image, model_name)
            pred = model.predict(processed_image, verbose=0)
            pred_class_idx = np.argmax(pred[0])
            pred_class = CLASS_NAMES[pred_class_idx]
            confidence = float(pred[0][pred_class_idx])

            predictions[model_name] = {
                'class': pred_class,
                'class_friendly': CLASS_NAMES_FRIENDLY[pred_class],
                'confidence': confidence,
                'probabilities': pred[0],
                'clinical_info': CLINICAL_INFO[pred_class]
            }
            
            progress_bar.progress((i + 1) / len(models))
            
        except Exception as e:
            st.error(f"Error en predicción con {model_name}: {e}")
    
    progress_bar.empty()
    status_text.empty()
    
    return predictions

def create_interactive_plots(predictions):
    """Crea gráficos interactivos con Plotly"""
    models = list(predictions.keys())
    n_models = len(models)
    
    fig = make_subplots(
        rows=1, cols=n_models,
        subplot_titles=[f'{model}' for model in models],
        specs=[[{"type": "bar"} for _ in range(n_models)]]
    )
    
    colors_plot = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe']
    
    for idx, (model_name, pred) in enumerate(predictions.items()):
        friendly_names = [CLASS_NAMES_FRIENDLY[class_name] for class_name in CLASS_NAMES]
        
        fig.add_trace(
            go.Bar(
                x=friendly_names,
                y=pred['probabilities'],
                name=model_name,
                marker_color=colors_plot,
                text=[f'{p:.1%}' for p in pred['probabilities']],
                textposition='outside',
                showlegend=False,
                marker=dict(
                    line=dict(width=2, color='white'),
                    opacity=0.9
                )
            ),
            row=1, col=idx+1
        )
    
    fig.update_layout(
        title={
            'text': "Distribución de Probabilidades por Modelo",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'family': 'Inter, sans-serif'}
        },
        height=500,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
    fig.update_yaxes(range=[0, 1], title_text="Probabilidad", gridcolor='rgba(0,0,0,0.1)')
    
    return fig

def create_consensus_chart(predictions):
    """Crea gráfico de consenso entre modelos"""
    prediction_counts = {}
    for pred in predictions.values():
        class_name = pred['class_friendly']
        prediction_counts[class_name] = prediction_counts.get(class_name, 0) + 1
    
    fig = go.Figure(data=[go.Pie(
        labels=list(prediction_counts.keys()),
        values=list(prediction_counts.values()),
        hole=0.6,
        textinfo='label+percent',
        textfont_size=14,
        marker=dict(
            colors=['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe'],
            line=dict(color='white', width=2)
        ),
        textposition='outside'
    )])
    
    fig.update_layout(
        title={
            'text': "Consenso entre Modelos",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'family': 'Inter, sans-serif'}
        },
        height=400,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        annotations=[
            dict(
                text=f'{len(predictions)}<br>Modelos',
                x=0.5, y=0.5,
                font=dict(size=20, family='Inter, sans-serif', weight=700),
                showarrow=False
            )
        ]
    )
    
    return fig

def generate_pdf_report(predictions, image_info, patient_info=None, statistical_results=None):
    """Genera un reporte en PDF con los resultados del análisis e imágenes del entrenamiento"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []
    
    # Estilos
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.darkblue
    )
    
    # Título del reporte
    story.append(Paragraph("REPORTE DE ANÁLISIS DE CÉLULAS CERVICALES", title_style))
    story.append(Paragraph("Sistema de Clasificación SIPaKMeD", styles['Normal']))
    story.append(Spacer(1, 30))
    
    # Información del análisis
    fecha_analisis = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    info_data = [
        ['Fecha de Análisis:', fecha_analisis],
        ['Sistema:', 'Clasificador de Células Cervicales - SIPaKMeD'],
        ['Imagen Analizada:', image_info.get('filename', 'N/A')],
        ['Dimensiones:', f"{image_info.get('size', 'N/A')}"],
        ['Formato:', image_info.get('format', 'N/A')]
    ]
    
    if patient_info:
        info_data.extend([
            ['Paciente:', patient_info.get('nombre', 'N/A')],
            ['ID:', patient_info.get('id', 'N/A')]
        ])
    
    info_table = Table(info_data, colWidths=[2.5*inch, 3.5*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (1, 0), (1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(info_table)
    story.append(Spacer(1, 30))
    
    # Resultados por modelo
    story.append(Paragraph("RESULTADOS POR MODELO", heading_style))
    
    results_data = [['Modelo', 'Tipo Celular', 'Confianza', 'Nivel de Riesgo']]
    for model_name, pred in predictions.items():
        results_data.append([
            model_name,
            pred['class_friendly'],
            f"{pred['confidence']:.2%}",
            pred['clinical_info']['riesgo']
        ])
    
    results_table = Table(results_data, colWidths=[1.5*inch, 2.5*inch, 1.2*inch, 1.2*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(results_table)
    story.append(Spacer(1, 30))
    
    # Interpretación clínica
    story.append(Paragraph("INTERPRETACIÓN CLÍNICA", heading_style))
    
    # Obtener predicción más común
    prediction_counts = {}
    for pred in predictions.values():
        class_name = pred['class']
        prediction_counts[class_name] = prediction_counts.get(class_name, 0) + 1
    
    most_common = max(prediction_counts.items(), key=lambda x: x[1])
    consensus_class = most_common[0]
    consensus_count = most_common[1]
    
    clinical_info = CLINICAL_INFO[consensus_class]
    
    # Determinar resultado
    if consensus_class in ['parabasal', 'superficial_intermediate']:
        status = "NORMAL"
        color = colors.green
    elif consensus_class in ['metaplastic']:
        status = "BENIGNO"
        color = colors.orange
    else:
        status = "REQUIERE ATENCIÓN"
        color = colors.red
    
    # Resultado principal
    resultado_style = ParagraphStyle(
        'Resultado',
        parent=styles['Normal'],
        fontSize=16,
        alignment=TA_CENTER,
        textColor=color,
        fontName='Helvetica-Bold'
    )
    
    story.append(Paragraph(f"<b>RESULTADO: {status}</b>", resultado_style))
    story.append(Paragraph(f"Tipo celular predominante: {CLASS_NAMES_FRIENDLY[consensus_class]}", styles['Normal']))
    story.append(Paragraph(f"Consenso: {consensus_count}/{len(predictions)} modelos", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Descripción clínica
    story.append(Paragraph(f"<b>Descripción:</b> {clinical_info['descripcion']}", styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Significado Clínico:</b> {clinical_info['significado']}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Recomendaciones
    story.append(Paragraph("RECOMENDACIONES", heading_style))
    
    if status == "REQUIERE ATENCIÓN":
        recomendaciones = [
            "• Consulte con un especialista en ginecología",
            "• Considere estudios adicionales (colposcopía, biopsia)",
            "• Mantenga seguimiento médico regular",
            "• Este resultado requiere interpretación por un patólogo certificado"
        ]
    else:
        recomendaciones = [
            "• Mantenga controles ginecológicos rutinarios",
            "• Continúe con el programa de tamizaje regular",
            "• Consulte con su médico para interpretación final"
        ]
    
    for rec in recomendaciones:
        story.append(Paragraph(rec, styles['Normal']))
    
    # SECCIÓN DE ENTRENAMIENTO - Sin PageBreak innecesario
    story.append(Spacer(1, 40))
    story.append(Paragraph("INFORMACIÓN DE ENTRENAMIENTO DE MODELOS", heading_style))
    
    # Métricas de entrenamiento
    story.append(Paragraph("Métricas de Rendimiento", heading_style))
    
    training_metrics_data = [
        ['Modelo', 'Accuracy', 'Tiempo', 'Parámetros'],
        ['MobileNetV2', '84.18%', '17.8 min', '3.0M'],
        ['ResNet50', '89.95%', '27.9 min', '24.8M'],
        ['EfficientNetB0', '86.07%', '15.6 min', '4.8M']
    ]
    
    metrics_table = Table(training_metrics_data, colWidths=[1.8*inch, 1.2*inch, 1.2*inch, 1.2*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(metrics_table)
    story.append(Spacer(1, 30))
    
    # Dataset información
    story.append(Paragraph("Dataset SIPaKMeD", heading_style))
    
    dataset_info_text = """
    • Total de imágenes: 5,015<br/>
    • Imágenes de entrenamiento: 4,010 (80%)<br/>
    • Imágenes de validación: 1,005 (20%)<br/>
    • Número de clases: 5 tipos de células cervicales<br/>
    • Formato: Imágenes JPG de alta resolución<br/>
    • Fuente: Dataset médico real de citología cervical
    """
    
    story.append(Paragraph(dataset_info_text, styles['Normal']))
    
    # SECCIÓN DE ANÁLISIS ESTADÍSTICO
    if statistical_results:
        story.append(Spacer(1, 30))
        story.append(Paragraph("ANÁLISIS ESTADÍSTICO INFERENCIAL", heading_style))
        
        # Matthews Correlation Coefficient
        story.append(Paragraph("Matthews Correlation Coefficient (MCC)", heading_style))
        story.append(Paragraph(
            "El MCC es una medida de calidad para clasificaciones que considera verdaderos y falsos positivos/negativos. "
            "Rango: [-1, 1] donde 1 indica predicción perfecta y 0 indica no mejor que aleatorio.",
            styles['Normal']
        ))
        story.append(Spacer(1, 12))
        
        mcc_scores = statistical_results.get('mcc_scores', {})
        if mcc_scores:
            # Crear gráfico visual de MCC
            mcc_data = [['Modelo', 'MCC', 'Interpretación', '']]
            for model, mcc in sorted(mcc_scores.items(), key=lambda x: x[1], reverse=True):
                interpretation = 'Excelente' if mcc > 0.5 else 'Bueno' if mcc > 0.3 else 'Regular'
                # Calcular ancho de barra visual
                bar_width = int(mcc * 50)  # Escala de 0-50 caracteres
                bar_visual = '█' * bar_width
                mcc_data.append([model, f'{mcc:.4f}', interpretation, bar_visual])
            
            mcc_table = Table(mcc_data, colWidths=[1.5*inch, 1*inch, 1.5*inch, 2.5*inch])
            
            # Determinar colores para cada fila según el valor de MCC
            table_style = [
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]
            
            # Colorear las barras según el valor
            for i, (model, mcc) in enumerate(sorted(mcc_scores.items(), key=lambda x: x[1], reverse=True), 1):
                if mcc > 0.5:
                    bar_color = colors.green
                elif mcc > 0.3:
                    bar_color = colors.orange
                else:
                    bar_color = colors.red
                table_style.append(('TEXTCOLOR', (3, i), (3, i), bar_color))
            
            mcc_table.setStyle(TableStyle(table_style))
            story.append(mcc_table)
        
        # Prueba de McNemar
        story.append(Spacer(1, 20))
        story.append(Paragraph("Prueba de McNemar entre Modelos", heading_style))
        story.append(Paragraph(
            "La prueba de McNemar evalúa si las diferencias entre modelos son estadísticamente significativas (α = 0.05).",
            styles['Normal']
        ))
        story.append(Spacer(1, 12))
        
        mcnemar_results = statistical_results.get('mcnemar_tests', {})
        if mcnemar_results:
            # Crear matriz de comparación
            models = ['MobileNetV2', 'ResNet50', 'EfficientNetB0']
            
            # Encabezados
            mcnemar_matrix_data = [[''] + models]
            
            # Llenar matriz
            for i, model1 in enumerate(models):
                row = [model1]
                for j, model2 in enumerate(models):
                    if i == j:
                        row.append('-')
                    else:
                        # Buscar el resultado correspondiente
                        key1 = f"{model1}_vs_{model2}"
                        key2 = f"{model2}_vs_{model1}"
                        
                        if key1 in mcnemar_results:
                            p_val = mcnemar_results[key1]['p_value']
                            if p_val < 0.001:
                                cell_text = "p<0.001 ***"
                            elif p_val < 0.01:
                                cell_text = f"p={p_val:.3f} **"
                            elif p_val < 0.05:
                                cell_text = f"p={p_val:.3f} *"
                            else:
                                cell_text = f"p={p_val:.3f} NS"
                            row.append(cell_text)
                        elif key2 in mcnemar_results:
                            p_val = mcnemar_results[key2]['p_value']
                            if p_val < 0.001:
                                cell_text = "p<0.001 ***"
                            elif p_val < 0.01:
                                cell_text = f"p={p_val:.3f} **"
                            elif p_val < 0.05:
                                cell_text = f"p={p_val:.3f} *"
                            else:
                                cell_text = f"p={p_val:.3f} NS"
                            row.append(cell_text)
                        else:
                            row.append('-')
                
                mcnemar_matrix_data.append(row)
            
            mcnemar_matrix = Table(mcnemar_matrix_data, colWidths=[1.5*inch] * 4)
            mcnemar_matrix.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('BACKGROUND', (0, 0), (0, -1), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (1, 1), (-1, -1), colors.beige)
            ]))
            
            story.append(mcnemar_matrix)
            story.append(Spacer(1, 10))
            story.append(Paragraph(
                "<i>NS: No significativo, *: p<0.05, **: p<0.01, ***: p<0.001</i>", 
                styles['Normal']
            ))
            
            # Detalles de comparaciones significativas
            story.append(Spacer(1, 15))
            story.append(Paragraph("Comparaciones Detalladas:", heading_style))
            
            for comparison, result in mcnemar_results.items():
                models_compared = comparison.replace('_vs_', ' vs ')
                if result['significant']:
                    story.append(Paragraph(
                        f"<b>{models_compared}</b>: {result['interpretation']} "
                        f"(χ²={result['statistic']:.2f}, p={result['p_value']:.4f})",
                        styles['Normal']
                    ))
    
    # Cargar imágenes del entrenamiento
    training_images = load_training_images()
    
    if training_images:
        # Comparación de modelos
        if 'model_comparison' in training_images and training_images['model_comparison']:
            story.append(PageBreak())
            story.append(Paragraph("COMPARACIÓN DE MODELOS", heading_style))
            story.append(Spacer(1, 20))
            
            try:
                img_path = training_images['model_comparison']['path']
                img = RLImage(img_path, width=5.5*inch, height=3.5*inch)
                story.append(img)
                story.append(Spacer(1, 12))
                story.append(Paragraph(
                    "<i>Comparación de precisión, pérdida y tiempo de entrenamiento</i>", 
                    styles['Normal']
                ))
            except:
                pass
        
        # Matrices de confusión
        if 'confusion_matrices' in training_images and training_images['confusion_matrices']:
            story.append(PageBreak())
            story.append(Paragraph("MATRICES DE CONFUSIÓN", heading_style))
            story.append(Spacer(1, 20))
            
            for img_info in training_images['confusion_matrices']:
                try:
                    story.append(Paragraph(f"<b>{img_info['model']}</b>", styles['Normal']))
                    story.append(Spacer(1, 12))
                    
                    img = RLImage(img_info['path'], width=4*inch, height=3*inch)
                    story.append(img)
                    story.append(Spacer(1, 20))
                except:
                    pass
        
        # Historiales de entrenamiento
        if 'training_histories' in training_images and training_images['training_histories']:
            story.append(PageBreak())
            story.append(Paragraph("HISTORIALES DE ENTRENAMIENTO", heading_style))
            story.append(Spacer(1, 20))
            
            for img_info in training_images['training_histories']:
                try:
                    story.append(Paragraph(f"<b>{img_info['model']}</b>", styles['Normal']))
                    story.append(Spacer(1, 12))
                    
                    img = RLImage(img_info['path'], width=4.5*inch, height=3*inch)
                    story.append(img)
                    story.append(Spacer(1, 20))
                except:
                    pass
    
    # Disclaimer final
    story.append(PageBreak())
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.red,
        alignment=TA_CENTER,
        spaceAfter=30
    )
    
    story.append(Paragraph(
        "<b>IMPORTANTE:</b> Este reporte es generado por un sistema de inteligencia artificial "
        "y tiene fines educativos y de investigación únicamente. NO reemplaza el diagnóstico "
        "médico profesional. Siempre consulte con un especialista calificado.",
        disclaimer_style
    ))
    
    # Construir PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def display_download_section(predictions, image_info):
    """Muestra la sección de descarga de reportes"""
    st.markdown("### 📥 Descargar Reporte Completo")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.expander("📋 Información del Paciente (Opcional)", expanded=False):
            patient_name = st.text_input("Nombre del Paciente", placeholder="Ej: María García")
            patient_id = st.text_input("ID/Historia Clínica", placeholder="Ej: HC-001234")
            
            patient_info = None
            if patient_name or patient_id:
                patient_info = {
                    'nombre': patient_name if patient_name else 'N/A',
                    'id': patient_id if patient_id else 'N/A'
                }
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🔽 Generar Reporte PDF", use_container_width=True, type="primary"):
            try:
                with st.spinner("Generando reporte profesional..."):
                    # Cargar resultados estadísticos si existen
                    statistical_results = load_statistical_results()
                    
                    # Generar PDF con resultados estadísticos
                    pdf_buffer = generate_pdf_report(predictions, image_info, patient_info, statistical_results)
                    fecha = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"reporte_celulas_cervicales_{fecha}.pdf"
                    
                    st.download_button(
                        label="📄 Descargar PDF",
                        data=pdf_buffer.getvalue(),
                        file_name=filename,
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                    st.success("✅ Reporte generado exitosamente")
                    
            except Exception as e:
                st.error(f"Error generando PDF: {str(e)}")

def display_clinical_interpretation(predictions):
    """Muestra interpretación clínica de los resultados"""
    # Obtener predicción más común
    prediction_counts = {}
    for pred in predictions.values():
        class_name = pred['class']
        prediction_counts[class_name] = prediction_counts.get(class_name, 0) + 1
    
    most_common = max(prediction_counts.items(), key=lambda x: x[1])
    consensus_class = most_common[0]
    consensus_count = most_common[1]
    
    clinical_info = CLINICAL_INFO[consensus_class]
    
    # Determinar el tipo de resultado
    if consensus_class in ['parabasal', 'superficial_intermediate']:
        status = "NORMAL"
        status_class = "status-normal"
        recommendation_type = "info"
    elif consensus_class in ['metaplastic']:
        status = "BENIGNO"
        status_class = "status-warning"
        recommendation_type = "info"
    else:
        status = "REQUIERE ATENCIÓN"
        status_class = "status-danger"
        recommendation_type = "warning"
    
    # Card de resultado principal
    st.markdown(f"""
    <div class="professional-card" style="text-align: center;">
        <h2 style="color: var(--text-primary); margin-bottom: 1rem;">🏥 Interpretación Clínica</h2>
        <div class="status-badge {status_class}" style="font-size: 1.2rem; margin: 1rem 0;">
            {clinical_info['icon']} {status}
        </div>
        <p style="font-size: 1.1rem; color: var(--text-primary); margin: 1rem 0;">
            <strong>{CLASS_NAMES_FRIENDLY[consensus_class]}</strong>
        </p>
        <p style="color: var(--text-secondary);">
            Consenso: {consensus_count} de {len(predictions)} modelos
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Información clínica
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="professional-card">
            <h4 style="color: var(--primary-color); margin-bottom: 1rem;">📋 Descripción</h4>
            <p style="color: var(--text-secondary);">{clinical_info['descripcion']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="professional-card">
            <h4 style="color: var(--primary-color); margin-bottom: 1rem;">🎯 Significado Clínico</h4>
            <p style="color: var(--text-secondary);">{clinical_info['significado']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recomendaciones
    if recommendation_type == "warning":
        st.markdown(f"""
        <div class="warning-box-professional">
            <h4 style="margin-bottom: 1rem;">⚠️ Recomendaciones Importantes</h4>
            <ul style="margin: 0; padding-left: 1.5rem;">
                <li>Consulte con un especialista en ginecología inmediatamente</li>
                <li>Considere estudios adicionales (colposcopía, biopsia)</li>
                <li>Mantenga seguimiento médico regular</li>
                <li>Este resultado requiere interpretación por un patólogo certificado</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="info-box-professional">
            <h4 style="margin-bottom: 1rem;">✅ Recomendaciones</h4>
            <ul style="margin: 0; padding-left: 1.5rem;">
                <li>Mantenga controles ginecológicos rutinarios</li>
                <li>Continúe con el programa de tamizaje regular</li>
                <li>Consulte con su médico para interpretación final</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Header principal mejorado
    st.markdown("""
    <div class="main-header">
        🔬 Clasificador de Células Cervicales
    </div>
    <p class="subtitle">
        Sistema de análisis automatizado basado en Deep Learning • Dataset SIPaKMeD
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar personalizado
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            🔬 SIPAKMED AI
        </div>
        """, unsafe_allow_html=True)
        
        # Sección de tipos de células
        st.markdown("""
        <div class="sidebar-section">
            <h3>📊 TIPOS DE CÉLULAS</h3>
        </div>
        """, unsafe_allow_html=True)
        
        for class_key, info in CLINICAL_INFO.items():
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); padding: 0.8rem; 
                        border-radius: 8px; margin-bottom: 0.8rem; 
                        border-left: 3px solid {info['color']};">
                <div style="color: white; font-weight: 600;">
                    {info['icon']} {CLASS_NAMES_FRIENDLY[class_key]}
                </div>
                <div style="color: rgba(255,255,255,0.7); font-size: 0.85rem; margin-top: 0.3rem;">
                    Riesgo: {info['riesgo']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Configuración
        st.markdown("""
        <div class="sidebar-section">
            <h3>⚙️ CONFIGURACIÓN</h3>
        </div>
        """, unsafe_allow_html=True)
        
        enhance_image = st.checkbox(
            "🖼️ Mejora de imagen CLAHE",
            value=True,
            help="Aplica mejora de contraste adaptativa"
        )
        
        # Información del sistema
        st.markdown("""
        <div class="sidebar-section">
            <h3>📊 INFORMACIÓN DEL SISTEMA</h3>
            <div style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                <p><strong>Modelos:</strong> 3 CNNs pre-entrenadas</p>
                <p><strong>Dataset:</strong> 5,015 imágenes</p>
                <p><strong>Accuracy:</strong> 84-90%</p>
                <p><strong>Validación:</strong> 20% holdout</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Advertencia
        st.markdown("""
        <div class="sidebar-section" style="background: rgba(252, 66, 74, 0.1); 
                                           border: 1px solid rgba(252, 66, 74, 0.3);">
            <h3>⚠️ AVISO LEGAL</h3>
            <div style="color: rgba(255,255,255,0.9); font-size: 0.85rem;">
                Esta herramienta es solo para investigación y educación. 
                NO reemplaza el diagnóstico médico profesional.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Contenido principal
    # Sección de introducción mejorada
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="professional-card" style="text-align: center;">
            <h3 style="color: var(--primary-color); margin-bottom: 1rem;">
                🎯 Sistema de Análisis Automatizado
            </h3>
            <p style="color: var(--text-secondary); line-height: 1.8;">
                Utiliza modelos de Deep Learning entrenados con el dataset SIPaKMeD para 
                clasificar automáticamente células cervicales en 5 categorías diferentes.
                Esta herramienta está diseñada para asistir en el análisis citológico.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Cargar modelos con diseño mejorado
    st.markdown("### 🤖 Sistema de Inteligencia Artificial")
    
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.spinner('🔄 Inicializando modelos de Deep Learning...'):
                models = load_models()
        
        if not models:
            st.error("❌ Error al cargar los modelos")
            st.markdown("""
            <div class="warning-box-professional" style="background: linear-gradient(135deg, #FC424A 0%, #FF6B6B 100%);">
                <h4>🚨 No se pudieron cargar los modelos</h4>
                <p>Por favor verifica que los archivos de modelos estén en la carpeta correcta:</p>
                <ul style="margin: 1rem 0; padding-left: 1.5rem;">
                    <li><code>data/models/sipakmed_MobileNetV2.h5</code></li>
                    <li><code>data/models/sipakmed_ResNet50.h5</code></li>
                    <li><code>data/models/sipakmed_EfficientNetB0.h5</code></li>
                </ul>
                <p>Si no tienes los modelos entrenados, ejecuta primero <code>main_real.py</code></p>
            </div>
            """, unsafe_allow_html=True)
            st.stop()
        
        # Mostrar modelos cargados de forma más visual
        st.markdown("""
        <div class="professional-card" style="text-align: center;">
            <h4 style="color: var(--primary-color); margin-bottom: 1rem;">
                ✅ Sistema Listo para Análisis
            </h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        metrics_display = [
            ("🧠 Modelos", f"{len(models)}", "Cargados"),
            ("⚡ Modo", "GPU" if tf.config.list_physical_devices('GPU') else "CPU", "Procesamiento"),
            ("🎯 Precisión", "84-90%", "Rango"),
            ("📊 Clases", "5", "Tipos de células")
        ]
        
        for col, (icon_label, value, sublabel) in zip([col1, col2, col3, col4], metrics_display):
            with col:
                st.markdown(f"""
                <div class="metric-card" style="padding: 1.5rem;">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon_label}</div>
                    <div style="font-size: 2rem; font-weight: 700; color: var(--primary-color);">{value}</div>
                    <div style="font-size: 0.9rem; color: var(--text-secondary);">{sublabel}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Mostrar resultados del entrenamiento
    display_training_results()
    
    # Sección de análisis
    st.markdown("### 📤 Análisis de Imagen")
    
    uploaded_file = st.file_uploader(
        "Selecciona una imagen microscópica de células cervicales",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Formatos soportados: PNG, JPG, JPEG, BMP, TIFF • Resolución recomendada: 224x224 o superior"
    )
    
    if uploaded_file is None:
        # Mostrar instrucciones cuando no hay archivo
        st.markdown("""
        <div class="professional-card" style="text-align: center; padding: 3rem;">
            <h3 style="color: var(--primary-color); margin-bottom: 1rem;">
                📸 Esperando imagen para analizar
            </h3>
            <p style="color: var(--text-secondary); line-height: 1.8; max-width: 600px; margin: 0 auto;">
                Por favor, carga una imagen microscópica de células cervicales para comenzar el análisis.
                El sistema clasificará automáticamente las células en una de las 5 categorías definidas
                utilizando los modelos de Deep Learning entrenados.
            </p>
            <div style="margin-top: 2rem; padding: 1rem; background: rgba(102, 126, 234, 0.1); 
                        border-radius: 8px; border: 1px solid rgba(102, 126, 234, 0.3);">
                <p style="color: var(--primary-color); font-weight: 600; margin: 0;">
                    💡 Tip: Para mejores resultados, use imágenes de alta calidad con buena iluminación
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    elif uploaded_file is not None:
        # Cargar imagen
        original_image = Image.open(uploaded_file)
        
        # Layout mejorado para mostrar imagen
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            <div class="professional-card">
                <h4 style="text-align: center; color: var(--primary-color); margin-bottom: 1rem;">
                    📷 Imagen Original
                </h4>
            </div>
            """, unsafe_allow_html=True)
            st.image(original_image, use_container_width=True)
            
            # Información de la imagen en formato tabla
            image_info = {
                'filename': uploaded_file.name,
                'size': f"{original_image.size[0]} x {original_image.size[1]}",
                'format': original_image.format,
                'mode': original_image.mode
            }
            
            info_df = pd.DataFrame([
                ["📄 Archivo", image_info['filename']],
                ["📐 Dimensiones", image_info['size']],
                ["🎨 Formato", image_info['format']],
                ["🔧 Modo", image_info['mode']]
            ], columns=["Propiedad", "Valor"])
            
            st.dataframe(info_df, use_container_width=True, hide_index=True)
        
        with col2:
            if enhance_image:
                st.markdown("""
                <div class="professional-card">
                    <h4 style="text-align: center; color: var(--primary-color); margin-bottom: 1rem;">
                        ✨ Imagen Mejorada
                    </h4>
                </div>
                """, unsafe_allow_html=True)
                
                with st.spinner('Aplicando mejoras CLAHE...'):
                    enhanced_img = enhance_cervical_cell_image(original_image)
                    enhanced_pil = Image.fromarray(enhanced_img.astype(np.uint8))
                    st.image(enhanced_pil, use_container_width=True)
                    analysis_image = enhanced_pil
            else:
                analysis_image = original_image
            
            # Realizar predicciones
            st.markdown("#### 🔍 Analizando con IA...")
            predictions = predict_cervical_cells(analysis_image, models)
        
        # Mostrar resultados
        if predictions:
            st.markdown("### 📊 Resultados del Análisis")
            
            # Cards de resultados por modelo
            cols = st.columns(len(predictions))
            for i, (model_name, pred) in enumerate(predictions.items()):
                with cols[i]:
                    clinical_info = pred['clinical_info']
                    
                    # Determinar color según riesgo
                    if clinical_info['riesgo'] == 'Alto':
                        gradient = "linear-gradient(135deg, #FC424A 0%, #FF6B6B 100%)"
                    elif clinical_info['riesgo'] == 'Moderado':
                        gradient = "linear-gradient(135deg, #FFAB00 0%, #FFC107 100%)"
                    else:
                        gradient = "linear-gradient(135deg, #00D25B 0%, #00E676 100%)"
                    
                    st.markdown(f"""
                    <div class="professional-card" style="text-align: center;">
                        <div style="background: {gradient}; 
                                    color: white; padding: 1rem; 
                                    border-radius: 12px; margin: -1rem -1rem 1rem -1rem;">
                            <h4 style="margin: 0;">{model_name}</h4>
                        </div>
                        <h3 style="color: var(--text-primary); margin: 1rem 0;">
                            {pred['confidence']:.1%}
                        </h3>
                        <p style="color: var(--text-secondary); font-weight: 600;">
                            {pred['class_friendly']}
                        </p>
                        <div class="status-badge {['status-normal', 'status-warning', 'status-danger'][['Normal', 'Bajo', 'Moderado', 'Alto'].index(clinical_info['riesgo']) if clinical_info['riesgo'] in ['Normal', 'Bajo'] else 2]}">
                            {clinical_info['icon']} {clinical_info['riesgo']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Gráficos interactivos
            st.markdown("### 📈 Análisis Visual Detallado")
            
            tab1, tab2 = st.tabs(["📊 Distribución de Probabilidades", "🎯 Consenso entre Modelos"])
            
            with tab1:
                fig_probs = create_interactive_plots(predictions)
                st.plotly_chart(fig_probs, use_container_width=True)
            
            with tab2:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    fig_consensus = create_consensus_chart(predictions)
                    st.plotly_chart(fig_consensus, use_container_width=True)
            
            # Interpretación clínica
            display_clinical_interpretation(predictions)
            
            # Sección de descarga
            display_download_section(predictions, image_info)
    
    # Footer profesional
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0; color: var(--text-secondary);'>
        <p style='font-size: 1.1rem; margin-bottom: 0.5rem;'>
            <strong>Sistema de Clasificación de Células Cervicales</strong>
        </p>
        <p style='font-size: 0.9rem;'>
            Desarrollado con TensorFlow y modelos CNN • Dataset SIPaKMeD (5,015 imágenes)
        </p>
        <p style='font-size: 0.85rem; color: var(--text-secondary); margin-top: 1rem;'>
            © 2024 - Solo para fines de investigación y educación médica
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()