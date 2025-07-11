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
import plotly.io as pio
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
import logging
from typing import Dict, List, Tuple, Optional
import tempfile

# ============================================================================
# CONFIGURACIÓN MULTILENGUAJE
# ============================================================================

LANGUAGES = {
    "es": "🇪🇸 Español",
    "en": "🇺🇸 English", 
    "pt": "🇧🇷 Português",
    "fr": "🇫🇷 Français"
}

TRANSLATIONS = {
    "es": {
        # Títulos principales
        "main_title": "🔬 Clasificador de Células Cervicales",
        "subtitle": "Sistema de análisis automatizado basado en Deep Learning • Dataset SIPaKMeD",
        "system_ready": "Sistema de Análisis Automatizado",
        "ai_system": "Sistema de Inteligencia Artificial",
        
        # Sidebar
        "sidebar_title": "🔬 SIPAKMED AI",
        "cell_types": "📊 TIPOS DE CÉLULAS",
        "configuration": "⚙️ CONFIGURACIÓN",
        "system_info": "📊 INFORMACIÓN DEL SISTEMA",
        "legal_notice": "⚠️ AVISO LEGAL",
        "clahe_enhancement": "🖼️ Mejora de imagen CLAHE",
        "clahe_help": "Aplica mejora de contraste adaptativa",
        "models_info": "Modelos: 3 CNNs pre-entrenadas",
        "dataset_info": "Dataset: 5,015 imágenes",
        "accuracy_info": "Accuracy: 84-90%",
        "validation_info": "Validación: 20% holdout",
        "legal_text": "Esta herramienta es solo para investigación y educación. NO reemplaza el diagnóstico médico profesional.",
        
        # Tipos de células
        "dyskeratotic": "Células Displásicas",
        "koilocytotic": "Células Koilocitóticas", 
        "metaplastic": "Células Metaplásicas",
        "parabasal": "Células Parabasales",
        "superficial_intermediate": "Células Superficiales-Intermedias",
        
        # Información clínica
        "dyskeratotic_desc": "Células con alteraciones displásicas que pueden indicar cambios precancerosos.",
        "dyskeratotic_meaning": "Requiere seguimiento médico y posibles estudios adicionales.",
        "koilocytotic_desc": "Células con cambios citopáticos característicos de infección por VPH.",
        "koilocytotic_meaning": "Indica presencia de virus del papiloma humano (VPH).",
        "metaplastic_desc": "Células de la zona de transformación cervical en proceso de cambio.",
        "metaplastic_meaning": "Proceso normal de reparación, generalmente benigno.",
        "parabasal_desc": "Células de las capas profundas del epitelio cervical.",
        "parabasal_meaning": "Parte normal del epitelio cervical estratificado.",
        "superficial_intermediate_desc": "Células de las capas superficiales e intermedias del epitelio.",
        "superficial_intermediate_meaning": "Células maduras normales del epitelio cervical.",
        
        # Niveles de riesgo
        "high_risk": "Alto",
        "moderate_risk": "Moderado",
        "low_risk": "Bajo",
        "normal_risk": "Normal",
        
        # Entrenamiento
        "training_results": "📊 Resultados del Entrenamiento",
        "training_subtitle": "Análisis completo del entrenamiento con <strong>5,015 imágenes</strong> del dataset SIPaKMeD",
        "general_comparison": "📈 Comparación General",
        "confusion_matrices": "🎯 Matrices de Confusión",
        "training_histories": "📉 Historiales de Entrenamiento", 
        "dataset_info_tab": "📊 Dataset Info",
        "statistical_analysis": "📊 Análisis Estadístico",
        
        # Análisis de imagen
        "image_analysis": "📤 Análisis de Imagen",
        "upload_instruction": "Selecciona una imagen microscópica de células cervicales",
        "upload_help": "Formatos soportados: PNG, JPG, JPEG, BMP, TIFF • Resolución recomendada: 224x224 o superior",
        "waiting_image": "📸 Esperando imagen para analizar",
        "upload_description": "Por favor, carga una imagen microscópica de células cervicales para comenzar el análisis. El sistema clasificará automáticamente las células en una de las 5 categorías definidas utilizando los modelos de Deep Learning entrenados.",
        "tip_quality": "💡 Tip: Para mejores resultados, use imágenes de alta calidad con buena iluminación",
        
        # Resultados
        "analysis_results": "📊 Resultados del Análisis",
        "visual_analysis": "📈 Análisis Visual Detallado",
        "probability_distribution": "📊 Distribución de Probabilidades",
        "model_consensus": "🎯 Consenso entre Modelos",
        "clinical_interpretation": "🏥 Interpretación Clínica",
        "description": "📋 Descripción",
        "clinical_meaning": "🎯 Significado Clínico",
        
        # Estados
        "normal": "NORMAL",
        "benign": "BENIGNO", 
        "requires_attention": "REQUIERE ATENCIÓN",
        "consensus": "Consenso",
        "models": "modelos",
        "models_agree": "modelos coinciden",
        
        # Recomendaciones
        "important_recommendations": "⚠️ Recomendaciones Importantes",
        "recommendations": "✅ Recomendaciones",
        "consult_specialist": "Consulte con un especialista en ginecología inmediatamente",
        "additional_studies": "Considere estudios adicionales (colposcopía, biopsia)",
        "regular_followup": "Mantenga seguimiento médico regular",
        "pathologist_interpretation": "Este resultado requiere interpretación por un patólogo certificado",
        "routine_controls": "Mantenga controles ginecológicos rutinarios",
        "continue_screening": "Continúe con el programa de tamizaje regular",
        "consult_doctor": "Consulte con su médico para interpretación final",
        
        # PDF y descarga
        "download_report": "📥 Descargar Reporte Completo",
        "patient_info": "📋 Información del Paciente (Opcional)",
        "patient_name": "Nombre del Paciente",
        "patient_id": "ID/Historia Clínica",
        "generate_pdf": "🔽 Generar Reporte PDF",
        "download_pdf": "📄 Descargar PDF",
        "generating_report": "Generando reporte profesional...",
        "report_generated": "✅ Reporte generado exitosamente",
        
        # Mensajes del sistema
        "loading_models": "🔄 Inicializando modelos de Deep Learning...",
        "system_ready_msg": "✅ Sistema Listo para Análisis",
        "models_loaded": "Cargados",
        "processing_mode": "Procesamiento",
        "accuracy_range": "Rango",
        "cell_types_count": "Tipos de células",
        "applying_clahe": "Aplicando mejoras CLAHE...",
        "analyzing_ai": "🔍 Analizando con IA...",
        
        # Errores
        "model_error": "❌ Error al cargar los modelos",
        "model_error_solution": "Solución:",
        "verify_files": "Verifica que los archivos .h5 estén en data/models/",
        "run_training": "Ejecuta main_real.py para entrenar los modelos", 
        "restart_app": "Reinicia la aplicación",
        "pdf_error": "Error generando PDF:",
        
        # Estadísticas
        "statistical_analysis_title": "📊 Análisis Estadístico Inferencial",
        "statistical_subtitle": "Evaluación rigurosa de los modelos mediante pruebas estadísticas",
        "mcc_title": "📈 Matthews Correlation Coefficient",
        "mcnemar_title": "🔬 Prueba de McNemar",
        "mcc_description": "El MCC es una medida de calidad para clasificaciones que considera verdaderos y falsos positivos/negativos. Es especialmente útil para datasets desbalanceados. Rango: [-1, 1]",
        "mcnemar_description": "La prueba de McNemar compara el rendimiento de dos modelos evaluando las predicciones discordantes. Es útil para determinar si un modelo es significativamente mejor que otro.",
        "perfect_prediction": "Predicción perfecta",
        "very_good_agreement": "Muy buena concordancia", 
        "no_better_random": "No mejor que aleatorio",
        "total_disagreement": "Desacuerdo total",
        "statistically_significant": "Diferencia estadísticamente significativa",
        "no_significant_difference": "No hay diferencia significativa",
        "model_ranking": "🏆 Ranking de Modelos por MCC",
        "interpretation": "Interpretación",
        "excellent": "⭐ Excelente",
        "good": "✅ Bueno", 
        "regular": "⚠️ Regular",
        "detailed_comparisons": "📋 Comparaciones Detalladas",
        
        # Métricas
        "time": "⏱️ Tiempo:",
        "parameters": "🔧 Parámetros:",
        "total_images": "TOTAL IMÁGENES",
        "training": "ENTRENAMIENTO",
        "validation": "VALIDACIÓN",
        "complete_dataset": "Dataset completo",
        "dataset_percentage": "del dataset",
        
        # Footer
        "footer_title": "🔬 Sistema de Clasificación de Células Cervicales", 
        "footer_subtitle": "Desarrollado con TensorFlow y modelos CNN • Dataset SIPaKMeD (5,015 imágenes)",
        "footer_disclaimer": "© 2024 - Solo para fines de investigación y educación médica",
        
        # Títulos principales del PDF
        "pdf_title": "REPORTE DE ANÁLISIS DE CÉLULAS CERVICALES",
        "pdf_subtitle": "Sistema de Clasificación SIPaKMeD",
        "analysis_date": "Fecha de Análisis:",
        "system": "Sistema:",
        "analyzed_image": "Imagen Analizada:",
        "dimensions": "Dimensiones:",
        "format": "Formato:",
        "patient": "Paciente:",
        "id": "ID:",
        
        # Secciones del PDF
        "results_by_model": "RESULTADOS POR MODELO",
        "model": "Modelo",
        "cell_type": "Tipo Celular", 
        "confidence": "Confianza",
        "risk_level": "Nivel de Riesgo",
        "clinical_interpretation_title": "INTERPRETACIÓN CLÍNICA",
        "result": "RESULTADO",
        "predominant_cell_type": "Tipo celular predominante:",
        "consensus_text": "Consenso:",
        "description_text": "Descripción:",
        "clinical_meaning_text": "Significado Clínico:",
        "recommendations_title": "RECOMENDACIONES",
        "analysis_charts": "GRÁFICOS DEL ANÁLISIS",
        "probability_distribution_title": "Distribución de Probabilidades por Modelo",
        "model_consensus_title": "Consenso entre Modelos",
        "chart_not_available": "Gráfico no disponible",
        "chart_error": "Error al incluir gráfico",
        
        # Información de entrenamiento
        "training_info_title": "INFORMACIÓN DE ENTRENAMIENTO DE MODELOS",
        "performance_metrics": "Métricas de Rendimiento",
        "accuracy": "Accuracy",
        "time_col": "Tiempo",
        "parameters_col": "Parámetros",
        "dataset_sipakmed": "Dataset SIPaKMeD",
        "dataset_info_text": """
        • Total de imágenes: 5,015<br/>
        • Imágenes de entrenamiento: 4,010 (80%)<br/>
        • Imágenes de validación: 1,005 (20%)<br/>
        • Número de clases: 5 tipos de células cervicales<br/>
        • Formato: Imágenes JPG de alta resolución<br/>
        • Fuente: Dataset médico real de citología cervical
        """,
        
        # Análisis estadístico
        "statistical_analysis_pdf": "ANÁLISIS ESTADÍSTICO INFERENCIAL",
        "mcc_full": "Matthews Correlation Coefficient (MCC)",
        "mcc_explanation": "El MCC es una medida de calidad para clasificaciones que considera verdaderos y falsos positivos/negativos. Rango: [-1, 1] donde 1 indica predicción perfecta y 0 indica no mejor que aleatorio.",
        "mcnemar_full": "Prueba de McNemar entre Modelos",
        "mcnemar_explanation": "La prueba de McNemar evalúa si las diferencias entre modelos son estadísticamente significativas (α = 0.05).",
        "model_comparison": "COMPARACIÓN DE MODELOS",
        "confusion_matrices_title": "MATRICES DE CONFUSIÓN",
        "training_histories_title": "HISTORIALES DE ENTRENAMIENTO",
        "detailed_comparisons_text": "Comparaciones Detalladas:",
        "legend_text": "NS: No significativo, *: p<0.05, **: p<0.01, ***: p<0.001",
        "comparison_precision": "Comparación de precisión, pérdida y tiempo de entrenamiento",
        
        # Disclaimer
        "important_notice": "IMPORTANTE:",
        "disclaimer_text": "Este reporte es generado por un sistema de inteligencia artificial y tiene fines educativos y de investigación únicamente. NO reemplaza el diagnóstico médico profesional. Siempre consulte con un especialista calificado."
    },
    
    "en": {
        # Main titles
        "main_title": "🔬 Cervical Cell Classifier",
        "subtitle": "Automated analysis system based on Deep Learning • SIPaKMeD Dataset",
        "system_ready": "Automated Analysis System", 
        "ai_system": "Artificial Intelligence System",
        
        # Sidebar
        "sidebar_title": "🔬 SIPAKMED AI",
        "cell_types": "📊 CELL TYPES",
        "configuration": "⚙️ CONFIGURATION",
        "system_info": "📊 SYSTEM INFORMATION",
        "legal_notice": "⚠️ LEGAL NOTICE",
        "clahe_enhancement": "🖼️ CLAHE Image Enhancement",
        "clahe_help": "Applies adaptive contrast enhancement",
        "models_info": "Models: 3 pre-trained CNNs",
        "dataset_info": "Dataset: 5,015 images",
        "accuracy_info": "Accuracy: 84-90%",
        "validation_info": "Validation: 20% holdout",
        "legal_text": "This tool is for research and education only. Does NOT replace professional medical diagnosis.",
        
        # Cell types
        "dyskeratotic": "Dyskeratotic Cells",
        "koilocytotic": "Koilocytotic Cells",
        "metaplastic": "Metaplastic Cells", 
        "parabasal": "Parabasal Cells",
        "superficial_intermediate": "Superficial-Intermediate Cells",
        
        # Clinical information
        "dyskeratotic_desc": "Cells with dysplastic alterations that may indicate precancerous changes.",
        "dyskeratotic_meaning": "Requires medical follow-up and possible additional studies.",
        "koilocytotic_desc": "Cells with cytopathic changes characteristic of HPV infection.",
        "koilocytotic_meaning": "Indicates presence of human papillomavirus (HPV).",
        "metaplastic_desc": "Cells from the cervical transformation zone undergoing change.",
        "metaplastic_meaning": "Normal repair process, generally benign.",
        "parabasal_desc": "Cells from the deep layers of the cervical epithelium.",
        "parabasal_meaning": "Normal part of the stratified cervical epithelium.",
        "superficial_intermediate_desc": "Cells from the superficial and intermediate epithelial layers.",
        "superficial_intermediate_meaning": "Normal mature cells of the cervical epithelium.",
        
        # Risk levels
        "high_risk": "High",
        "moderate_risk": "Moderate",
        "low_risk": "Low", 
        "normal_risk": "Normal",
        
        # Training
        "training_results": "📊 Training Results",
        "training_subtitle": "Complete training analysis with <strong>5,015 images</strong> from SIPaKMeD dataset",
        "general_comparison": "📈 General Comparison",
        "confusion_matrices": "🎯 Confusion Matrices",
        "training_histories": "📉 Training Histories",
        "dataset_info_tab": "📊 Dataset Info", 
        "statistical_analysis": "📊 Statistical Analysis",
        
        # Image analysis
        "image_analysis": "📤 Image Analysis",
        "upload_instruction": "Select a microscopic image of cervical cells",
        "upload_help": "Supported formats: PNG, JPG, JPEG, BMP, TIFF • Recommended resolution: 224x224 or higher",
        "waiting_image": "📸 Waiting for image to analyze",
        "upload_description": "Please upload a microscopic image of cervical cells to begin analysis. The system will automatically classify cells into one of 5 defined categories using trained Deep Learning models.",
        "tip_quality": "💡 Tip: For best results, use high-quality images with good lighting",
        
        # Results
        "analysis_results": "📊 Analysis Results", 
        "visual_analysis": "📈 Detailed Visual Analysis",
        "probability_distribution": "📊 Probability Distribution",
        "model_consensus": "🎯 Model Consensus",
        "clinical_interpretation": "🏥 Clinical Interpretation",
        "description": "📋 Description",
        "clinical_meaning": "🎯 Clinical Meaning",
        
        # States
        "normal": "NORMAL",
        "benign": "BENIGN",
        "requires_attention": "REQUIRES ATTENTION",
        "consensus": "Consensus",
        "models": "models",
        "models_agree": "models agree",
        
        # Recommendations
        "important_recommendations": "⚠️ Important Recommendations",
        "recommendations": "✅ Recommendations",
        "consult_specialist": "Consult with a gynecology specialist immediately",
        "additional_studies": "Consider additional studies (colposcopy, biopsy)",
        "regular_followup": "Maintain regular medical follow-up",
        "pathologist_interpretation": "This result requires interpretation by a certified pathologist",
        "routine_controls": "Maintain routine gynecological controls",
        "continue_screening": "Continue with regular screening program",
        "consult_doctor": "Consult with your doctor for final interpretation",
        
        # PDF and download
        "download_report": "📥 Download Complete Report",
        "patient_info": "📋 Patient Information (Optional)",
        "patient_name": "Patient Name",
        "patient_id": "ID/Medical Record",
        "generate_pdf": "🔽 Generate PDF Report",
        "download_pdf": "📄 Download PDF",
        "generating_report": "Generating professional report...",
        "report_generated": "✅ Report generated successfully",
        
        # System messages
        "loading_models": "🔄 Initializing Deep Learning models...",
        "system_ready_msg": "✅ System Ready for Analysis",
        "models_loaded": "Loaded",
        "processing_mode": "Processing",
        "accuracy_range": "Range",
        "cell_types_count": "Cell types",
        "applying_clahe": "Applying CLAHE enhancements...",
        "analyzing_ai": "🔍 Analyzing with AI...",
        
        # Errors
        "model_error": "❌ Error loading models",
        "model_error_solution": "Solution:",
        "verify_files": "Verify that .h5 files are in data/models/",
        "run_training": "Run main_real.py to train models",
        "restart_app": "Restart the application",
        "pdf_error": "Error generating PDF:",
        
        # Statistics
        "statistical_analysis_title": "📊 Inferential Statistical Analysis",
        "statistical_subtitle": "Rigorous model evaluation through statistical tests",
        "mcc_title": "📈 Matthews Correlation Coefficient",
        "mcnemar_title": "🔬 McNemar Test",
        "mcc_description": "MCC is a quality measure for classifications that considers true and false positives/negatives. Especially useful for imbalanced datasets. Range: [-1, 1]",
        "mcnemar_description": "McNemar test compares the performance of two models by evaluating discordant predictions. Useful for determining if one model is significantly better than another.",
        "perfect_prediction": "Perfect prediction",
        "very_good_agreement": "Very good agreement",
        "no_better_random": "No better than random",
        "total_disagreement": "Total disagreement",
        "statistically_significant": "Statistically significant difference",
        "no_significant_difference": "No significant difference",
        "model_ranking": "🏆 Model Ranking by MCC",
        "interpretation": "Interpretation",
        "excellent": "⭐ Excellent",
        "good": "✅ Good",
        "regular": "⚠️ Regular",
        "detailed_comparisons": "📋 Detailed Comparisons",
        
        # Metrics
        "time": "⏱️ Time:",
        "parameters": "🔧 Parameters:",
        "total_images": "TOTAL IMAGES",
        "training": "TRAINING",
        "validation": "VALIDATION",
        "complete_dataset": "Complete dataset",
        "dataset_percentage": "of dataset",
        
        # Títulos principales del PDF
        "pdf_title": "CERVICAL CELL ANALYSIS REPORT",
        "pdf_subtitle": "SIPaKMeD Classification System",
        "analysis_date": "Analysis Date:",
        "system": "System:",
        "analyzed_image": "Analyzed Image:",
        "dimensions": "Dimensions:",
        "format": "Format:",
        "patient": "Patient:",
        "id": "ID:",
        
        # Secciones del PDF
        "results_by_model": "RESULTS BY MODEL",
        "model": "Model",
        "cell_type": "Cell Type", 
        "confidence": "Confidence",
        "risk_level": "Risk Level",
        "clinical_interpretation_title": "CLINICAL INTERPRETATION",
        "result": "RESULT",
        "predominant_cell_type": "Predominant cell type:",
        "consensus_text": "Consensus:",
        "description_text": "Description:",
        "clinical_meaning_text": "Clinical Meaning:",
        "recommendations_title": "RECOMMENDATIONS",
        "analysis_charts": "ANALYSIS CHARTS",
        "probability_distribution_title": "Probability Distribution by Model",
        "model_consensus_title": "Model Consensus",
        "chart_not_available": "Chart not available",
        "chart_error": "Error including chart",
        
        # Información de entrenamiento
        "training_info_title": "MODEL TRAINING INFORMATION",
        "performance_metrics": "Performance Metrics",
        "accuracy": "Accuracy",
        "time_col": "Time",
        "parameters_col": "Parameters",
        "dataset_sipakmed": "SIPaKMeD Dataset",
        "dataset_info_text": """
        • Total images: 5,015<br/>
        • Training images: 4,010 (80%)<br/>
        • Validation images: 1,005 (20%)<br/>
        • Number of classes: 5 cervical cell types<br/>
        • Format: High resolution JPG images<br/>
        • Source: Real medical cytology dataset
        """,
        
        # Análisis estadístico
        "statistical_analysis_pdf": "INFERENTIAL STATISTICAL ANALYSIS",
        "mcc_full": "Matthews Correlation Coefficient (MCC)",
        "mcc_explanation": "MCC is a quality measure for classifications that considers true and false positives/negatives. Range: [-1, 1] where 1 indicates perfect prediction and 0 indicates no better than random.",
        "mcnemar_full": "McNemar Test between Models",
        "mcnemar_explanation": "McNemar test evaluates whether differences between models are statistically significant (α = 0.05).",
        "model_comparison": "MODEL COMPARISON",
        "confusion_matrices_title": "CONFUSION MATRICES",
        "training_histories_title": "TRAINING HISTORIES",
        "detailed_comparisons_text": "Detailed Comparisons:",
        "legend_text": "NS: Not significant, *: p<0.05, **: p<0.01, ***: p<0.001",
        "comparison_precision": "Comparison of precision, loss and training time",
        
        # Disclaimer
        "important_notice": "IMPORTANT:",
        "disclaimer_text": "This report is generated by an artificial intelligence system and is for educational and research purposes only. It does NOT replace professional medical diagnosis. Always consult with a qualified specialist."
    },
    
    "pt": {
        # Títulos principais
        "main_title": "🔬 Classificador de Células Cervicais",
        "subtitle": "Sistema de análise automatizada baseado em Deep Learning • Dataset SIPaKMeD",
        "system_ready": "Sistema de Análise Automatizada",
        "ai_system": "Sistema de Inteligência Artificial",
        
        # Sidebar
        "sidebar_title": "🔬 SIPAKMED AI",
        "cell_types": "📊 TIPOS DE CÉLULAS",
        "configuration": "⚙️ CONFIGURAÇÃO",
        "system_info": "📊 INFORMAÇÕES DO SISTEMA",
        "legal_notice": "⚠️ AVISO LEGAL",
        "clahe_enhancement": "🖼️ Melhoria de imagem CLAHE",
        "clahe_help": "Aplica melhoria de contraste adaptativo",
        "models_info": "Modelos: 3 CNNs pré-treinadas",
        "dataset_info": "Dataset: 5.015 imagens",
        "accuracy_info": "Precisão: 84-90%",
        "validation_info": "Validação: 20% holdout",
        "legal_text": "Esta ferramenta é apenas para pesquisa e educação. NÃO substitui o diagnóstico médico profissional.",
        
        # Tipos de células
        "dyskeratotic": "Células Disqueratóticas",
        "koilocytotic": "Células Coilocitóticas",
        "metaplastic": "Células Metaplásicas",
        "parabasal": "Células Parabasais",
        "superficial_intermediate": "Células Superficiais-Intermediárias",
        
        # Informações clínicas
        "dyskeratotic_desc": "Células com alterações displásicas que podem indicar mudanças pré-cancerosas.",
        "dyskeratotic_meaning": "Requer acompanhamento médico e possíveis estudos adicionais.",
        "koilocytotic_desc": "Células com mudanças citopatológicas características de infecção por HPV.",
        "koilocytotic_meaning": "Indica presença do papilomavírus humano (HPV).",
        "metaplastic_desc": "Células da zona de transformação cervical em processo de mudança.",
        "metaplastic_meaning": "Processo normal de reparação, geralmente benigno.",
        "parabasal_desc": "Células das camadas profundas do epitélio cervical.",
        "parabasal_meaning": "Parte normal do epitélio cervical estratificado.",
        "superficial_intermediate_desc": "Células das camadas superficiais e intermediárias do epitélio.",
        "superficial_intermediate_meaning": "Células maduras normais do epitélio cervical.",
        
        # Níveis de risco
        "high_risk": "Alto",
        "moderate_risk": "Moderado",
        "low_risk": "Baixo",
        "normal_risk": "Normal",
        
        # Treinamento
        "training_results": "📊 Resultados do Treinamento",
        "training_subtitle": "Análise completa do treinamento com <strong>5.015 imagens</strong> do dataset SIPaKMeD",
        "general_comparison": "📈 Comparação Geral",
        "confusion_matrices": "🎯 Matrizes de Confusão",
        "training_histories": "📉 Históricos de Treinamento",
        "dataset_info_tab": "📊 Info do Dataset",
        "statistical_analysis": "📊 Análise Estatística",
        
        # Análise de imagem
        "image_analysis": "📤 Análise de Imagem",
        "upload_instruction": "Selecione uma imagem microscópica de células cervicais",
        "upload_help": "Formatos suportados: PNG, JPG, JPEG, BMP, TIFF • Resolução recomendada: 224x224 ou superior",
        "waiting_image": "📸 Aguardando imagem para analisar",
        "upload_description": "Por favor, carregue uma imagem microscópica de células cervicais para começar a análise. O sistema classificará automaticamente as células em uma das 5 categorias definidas usando os modelos de Deep Learning treinados.",
        "tip_quality": "💡 Dica: Para melhores resultados, use imagens de alta qualidade com boa iluminação",
        
        # Resultados
        "analysis_results": "📊 Resultados da Análise",
        "visual_analysis": "📈 Análise Visual Detalhada",
        "probability_distribution": "📊 Distribuição de Probabilidades",
        "model_consensus": "🎯 Consenso entre Modelos",
        "clinical_interpretation": "🏥 Interpretação Clínica",
        "description": "📋 Descrição",
        "clinical_meaning": "🎯 Significado Clínico",
        
        # Estados
        "normal": "NORMAL",
        "benign": "BENIGNO",
        "requires_attention": "REQUER ATENÇÃO",
        "consensus": "Consenso",
        "models": "modelos",
        "models_agree": "modelos concordam",
        
        # Recomendações
        "important_recommendations": "⚠️ Recomendações Importantes",
        "recommendations": "✅ Recomendações",
        "consult_specialist": "Consulte um especialista em ginecologia imediatamente",
        "additional_studies": "Considere estudos adicionais (colposcopia, biópsia)",
        "regular_followup": "Mantenha acompanhamento médico regular",
        "pathologist_interpretation": "Este resultado requer interpretação por um patologista certificado",
        "routine_controls": "Mantenha controles ginecológicos de rotina",
        "continue_screening": "Continue com o programa de rastreamento regular",
        "consult_doctor": "Consulte seu médico para interpretação final",
        
        # PDF e download
        "download_report": "📥 Baixar Relatório Completo",
        "patient_info": "📋 Informações do Paciente (Opcional)",
        "patient_name": "Nome do Paciente",
        "patient_id": "ID/Prontuário Médico",
        "generate_pdf": "🔽 Gerar Relatório PDF",
        "download_pdf": "📄 Baixar PDF",
        "generating_report": "Gerando relatório profissional...",
        "report_generated": "✅ Relatório gerado com sucesso",
        
        # Mensagens do sistema
        "loading_models": "🔄 Inicializando modelos de Deep Learning...",
        "system_ready_msg": "✅ Sistema Pronto para Análise",
        "models_loaded": "Carregados",
        "processing_mode": "Processamento",
        "accuracy_range": "Faixa",
        "cell_types_count": "Tipos de células",
        "applying_clahe": "Aplicando melhorias CLAHE...",
        "analyzing_ai": "🔍 Analisando com IA...",
        
        # Erros
        "model_error": "❌ Erro ao carregar os modelos",
        "model_error_solution": "Solução:",
        "verify_files": "Verifique se os arquivos .h5 estão em data/models/",
        "run_training": "Execute main_real.py para treinar os modelos",
        "restart_app": "Reinicie a aplicação",
        "pdf_error": "Erro gerando PDF:",
        
        # Estatísticas
        "statistical_analysis_title": "📊 Análise Estatística Inferencial",
        "statistical_subtitle": "Avaliação rigorosa dos modelos através de testes estatísticos",
        "mcc_title": "📈 Matthews Correlation Coefficient",
        "mcnemar_title": "🔬 Teste de McNemar",
        "mcc_description": "O MCC é uma medida de qualidade para classificações que considera verdadeiros e falsos positivos/negativos. É especialmente útil para datasets desbalanceados. Faixa: [-1, 1]",
        "mcnemar_description": "O teste de McNemar compara o desempenho de dois modelos avaliando as predições discordantes. É útil para determinar se um modelo é significativamente melhor que outro.",
        "perfect_prediction": "Predição perfeita",
        "very_good_agreement": "Muito boa concordância",
        "no_better_random": "Não melhor que aleatório",
        "total_disagreement": "Desacordo total",
        "statistically_significant": "Diferença estatisticamente significativa",
        "no_significant_difference": "Não há diferença significativa",
        "model_ranking": "🏆 Ranking de Modelos por MCC",
        "interpretation": "Interpretação",
        "excellent": "⭐ Excelente",
        "good": "✅ Bom",
        "regular": "⚠️ Regular",
        "detailed_comparisons": "📋 Comparações Detalhadas",
        
        # Métricas
        "time": "⏱️ Tempo:",
        "parameters": "🔧 Parâmetros:",
        "total_images": "TOTAL DE IMAGENS",
        "training": "TREINAMENTO",
        "validation": "VALIDAÇÃO",
        "complete_dataset": "Dataset completo",
        "dataset_percentage": "do dataset",
        
        # Footer
        "footer_title": "🔬 Sistema de Classificação de Células Cervicais",
        "footer_subtitle": "Desenvolvido com TensorFlow e modelos CNN • Dataset SIPaKMeD (5.015 imagens)",
        "footer_disclaimer": "© 2024 - Apenas para fins de pesquisa e educação médica",
        
        # PDF principais
        "pdf_title": "RELATÓRIO DE ANÁLISE DE CÉLULAS CERVICAIS",
        "pdf_subtitle": "Sistema de Classificação SIPaKMeD",
        "analysis_date": "Data da Análise:",
        "system": "Sistema:",
        "analyzed_image": "Imagem Analisada:",
        "dimensions": "Dimensões:",
        "format": "Formato:",
        "patient": "Paciente:",
        "id": "ID:",
        "results_by_model": "RESULTADOS POR MODELO",
        "model": "Modelo",
        "cell_type": "Tipo Celular",
        "confidence": "Confiança",
        "risk_level": "Nível de Risco",
        "clinical_interpretation_title": "INTERPRETAÇÃO CLÍNICA",
        "result": "RESULTADO",
        "predominant_cell_type": "Tipo celular predominante:",
        "consensus_text": "Consenso:",
        "description_text": "Descrição:",
        "clinical_meaning_text": "Significado Clínico:",
        "recommendations_title": "RECOMENDAÇÕES",
        "analysis_charts": "GRÁFICOS DA ANÁLISE",
        "probability_distribution_title": "Distribuição de Probabilidades por Modelo",
        "model_consensus_title": "Consenso entre Modelos",
        "chart_not_available": "Gráfico não disponível",
        "chart_error": "Erro ao incluir gráfico",
        "training_info_title": "INFORMAÇÕES DE TREINAMENTO DOS MODELOS",
        "performance_metrics": "Métricas de Desempenho",
        "accuracy": "Precisão",
        "time_col": "Tempo",
        "parameters_col": "Parâmetros",
        "dataset_sipakmed": "Dataset SIPaKMeD",
        "dataset_info_text": """
        • Total de imagens: 5.015<br/>
        • Imagens de treinamento: 4.010 (80%)<br/>
        • Imagens de validação: 1.005 (20%)<br/>
        • Número de classes: 5 tipos de células cervicais<br/>
        • Formato: Imagens JPG de alta resolução<br/>
        • Fonte: Dataset médico real de citologia cervical
        """,
        "statistical_analysis_pdf": "ANÁLISE ESTATÍSTICA INFERENCIAL",
        "mcc_full": "Matthews Correlation Coefficient (MCC)",
        "mcc_explanation": "O MCC é uma medida de qualidade para classificações que considera verdadeiros e falsos positivos/negativos. Faixa: [-1, 1] onde 1 indica predição perfeita e 0 indica não melhor que aleatório.",
        "mcnemar_full": "Teste de McNemar entre Modelos",
        "mcnemar_explanation": "O teste de McNemar avalia se as diferenças entre modelos são estatisticamente significativas (α = 0.05).",
        "model_comparison": "COMPARAÇÃO DE MODELOS",
        "confusion_matrices_title": "MATRIZES DE CONFUSÃO",
        "training_histories_title": "HISTÓRICOS DE TREINAMENTO",
        "detailed_comparisons_text": "Comparações Detalhadas:",
        "legend_text": "NS: Não significativo, *: p<0.05, **: p<0.01, ***: p<0.001",
        "comparison_precision": "Comparação de precisão, perda e tempo de treinamento",
        "important_notice": "IMPORTANTE:",
        "disclaimer_text": "Este relatório é gerado por um sistema de inteligência artificial e tem fins educacionais e de pesquisa apenas. NÃO substitui o diagnóstico médico profissional. Sempre consulte um especialista qualificado."
    },
    
    "fr": {
        # Titres principaux
        "main_title": "🔬 Classificateur de Cellules Cervicales",
        "subtitle": "Système d'analyse automatisé basé sur Deep Learning • Dataset SIPaKMeD",
        "system_ready": "Système d'Analyse Automatisé",
        "ai_system": "Système d'Intelligence Artificielle",
        
        # Sidebar
        "sidebar_title": "🔬 SIPAKMED AI",
        "cell_types": "📊 TYPES DE CELLULES",
        "configuration": "⚙️ CONFIGURATION",
        "system_info": "📊 INFORMATIONS SYSTÈME",
        "legal_notice": "⚠️ AVIS LÉGAL",
        "clahe_enhancement": "🖼️ Amélioration d'image CLAHE",
        "clahe_help": "Applique une amélioration de contraste adaptatif",
        "legal_text": "Cet outil est uniquement pour la recherche et l'éducation. NE remplace PAS le diagnostic médical professionnel.",
        
        # Types de cellules
        "dyskeratotic": "Cellules Dyskératosiques",
        "koilocytotic": "Cellules Koïlocytotiques",
        "metaplastic": "Cellules Métaplasiques",
        "parabasal": "Cellules Parabasales",
        "superficial_intermediate": "Cellules Superficielles-Intermédiaires",
        
        # États
        "normal": "NORMAL",
        "benign": "BÉNIN",
        "requires_attention": "NÉCESSITE ATTENTION",
        "consensus": "Consensus",
        "models": "modèles",
        "models_agree": "modèles concordent",
        
        # Analyse d'image
        "image_analysis": "📤 Analyse d'Image",
        "upload_instruction": "Sélectionnez une image microscopique de cellules cervicales",
        "analysis_results": "📊 Résultats de l'Analyse",
        "clinical_interpretation": "🏥 Interprétation Clinique",
        
        # PDF
        "download_report": "📥 Télécharger Rapport Complet",
        "generate_pdf": "🔽 Générer Rapport PDF",
        "download_pdf": "📄 Télécharger PDF",
        
        # Pied de page
        "footer_title": "🔬 Système de Classification de Cellules Cervicales",
        "footer_subtitle": "Développé avec TensorFlow et modèles CNN • Dataset SIPaKMeD (5.015 images)",
        "footer_disclaimer": "© 2024 - À des fins de recherche et d'éducation médicale uniquement",
        
        # PDF principaux
        "pdf_title": "RAPPORT D'ANALYSE DE CELLULES CERVICALES",
        "pdf_subtitle": "Système de Classification SIPaKMeD",
        "analysis_date": "Date d'Analyse:",
        "system": "Système:",
        "analyzed_image": "Image Analysée:",
        "results_by_model": "RÉSULTATS PAR MODÈLE",
        "model": "Modèle",
        "cell_type": "Type Cellulaire",
        "confidence": "Confiance",
        "risk_level": "Niveau de Risque",
        "clinical_interpretation_title": "INTERPRÉTATION CLINIQUE",
        "result": "RÉSULTAT",
        "recommendations_title": "RECOMMANDATIONS",
        "analysis_charts": "GRAPHIQUES D'ANALYSE",
        "training_info_title": "INFORMATIONS D'ENTRAÎNEMENT DES MODÈLES",
        "performance_metrics": "Métriques de Performance",
        "statistical_analysis_pdf": "ANALYSE STATISTIQUE INFÉRENTIELLE",
        "important_notice": "IMPORTANT:",
        "disclaimer_text": "Ce rapport est généré par un système d'intelligence artificielle et est à des fins éducatives et de recherche uniquement. Il NE remplace PAS le diagnostic médical professionnel. Consultez toujours un spécialiste qualifié."
    }
}

# ============================================================================
# CONFIGURACIÓN Y CONSTANTES
# ============================================================================

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de la página
st.set_page_config(
    page_title="🔬 Clasificador de Células Cervicales - SIPaKMeD",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# ============================================================================
# FUNCIONES DE MULTILENGUAJE
# ============================================================================

def get_language():
    """Obtiene el idioma seleccionado"""
    if 'language' not in st.session_state:
        st.session_state.language = 'es'
    return st.session_state.language

def t(key: str) -> str:
    """Función de traducción"""
    lang = get_language()
    return TRANSLATIONS.get(lang, {}).get(key, key)

def get_class_names_friendly():
    """Obtiene los nombres amigables de las clases según el idioma"""
    lang = get_language()
    return {
        "dyskeratotic": t("dyskeratotic"),
        "koilocytotic": t("koilocytotic"),
        "metaplastic": t("metaplastic"), 
        "parabasal": t("parabasal"),
        "superficial_intermediate": t("superficial_intermediate")
    }

def get_clinical_info():
    """Obtiene la información clínica según el idioma"""
    return {
        "dyskeratotic": {
            "descripcion": t("dyskeratotic_desc"),
            "significado": t("dyskeratotic_meaning"),
            "color": "#FC424A",
            "riesgo": t("high_risk"),
            "icon": "🔴"
        },
        "koilocytotic": {
            "descripcion": t("koilocytotic_desc"),
            "significado": t("koilocytotic_meaning"),
            "color": "#FFAB00",
            "riesgo": t("moderate_risk"),
            "icon": "🟠"
        },
        "metaplastic": {
            "descripcion": t("metaplastic_desc"),
            "significado": t("metaplastic_meaning"),
            "color": "#0066CC",
            "riesgo": t("low_risk"),
            "icon": "🟡"
        },
        "parabasal": {
            "descripcion": t("parabasal_desc"),
            "significado": t("parabasal_meaning"),
            "color": "#00D25B",
            "riesgo": t("normal_risk"),
            "icon": "🟢"
        },
        "superficial_intermediate": {
            "descripcion": t("superficial_intermediate_desc"),
            "significado": t("superficial_intermediate_meaning"),
            "color": "#00D25B", 
            "riesgo": t("normal_risk"),
            "icon": "🟢"
        }
    }

# ============================================================================
# ESTILOS CSS MEJORADOS
# ============================================================================

# Estilos CSS personalizados mejorados y más profesionales
st.markdown("""
<style>
    /* Importar fuentes profesionales */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Variables de color optimizadas */
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

# ============================================================================
# FUNCIONES OPTIMIZADAS
# ============================================================================

@st.cache_resource
def load_models():
    """Carga los modelos entrenados de SIPaKMeD - OPTIMIZADA CON CACHE"""
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
        status_text.text(f'{t("loading_models").replace("🔄 Inicializando", "🔄")} {name}...')
        
        if os.path.exists(model_path):
            try:
                models[name] = load_model(model_path)
                progress_bar.progress((i + 1) / len(model_files))
                logger.info(f"Modelo {name} cargado exitosamente")
            except Exception as e:
                st.error(f"❌ Error cargando {name}: {str(e)}")
                logger.error(f"Error cargando {name}: {str(e)}")
        else:
            st.warning(f"⚠️ No se encontró el archivo: {model_path}")
            logger.warning(f"Archivo no encontrado: {model_path}")
    
    progress_bar.empty()
    status_text.empty()
    
    return models

@st.cache_data
def load_training_images():
    """Carga las imágenes generadas durante el entrenamiento - OPTIMIZADA"""
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

@st.cache_data
def load_statistical_results():
    """Carga los resultados del análisis estadístico si existen - OPTIMIZADA"""
    try:
        stats_path = Path("reports/statistical_analysis.json")
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error cargando estadísticas: {e}")
    return None

def save_plotly_as_image(fig, filename_prefix="plot"):
    """Convierte una figura de Plotly a imagen para incluir en PDF"""
    try:
        # Verificar que plotly.io esté disponible
        if not hasattr(pio, 'to_image'):
            logger.error("plotly.io.to_image no está disponible")
            return None
        
        # Convertir figura a bytes
        img_bytes = pio.to_image(fig, format='png', width=800, height=600, scale=2)
        
        # Crear archivo temporal con manejo mejorado
        temp_dir = tempfile.gettempdir()
        temp_filename = f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        # Escribir imagen al archivo
        with open(temp_path, 'wb') as f:
            f.write(img_bytes)
        
        # Verificar que el archivo existe y tiene contenido
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            logger.info(f"Gráfico guardado exitosamente en: {temp_path}")
            return temp_path
        else:
            logger.error(f"Archivo temporal no creado correctamente: {temp_path}")
            return None
            
    except Exception as e:
        logger.error(f"Error guardando gráfico como imagen: {e}")
        # Intentar alternativa: guardar información sobre el gráfico
        logger.info("Los gráficos no se pudieron incluir en el PDF. Esto puede deberse a:")
        logger.info("1. Kaleido no está instalado: pip install kaleido")
        logger.info("2. Problemas de permisos en archivos temporales")
        logger.info("3. Limitaciones del sistema operativo")
        return None

def display_statistical_analysis(statistical_results):
    """Muestra el análisis estadístico inferencial (MCC y McNemar)"""
    st.markdown(f"""
    <div class="results-section animate-fadeIn">
        <h2 style="text-align: center; color: var(--text-primary); margin-bottom: 1rem; font-weight: 700;">
            {t('statistical_analysis_title')}
        </h2>
        <p style="text-align: center; color: var(--text-secondary); font-size: 1.1rem; margin-bottom: 2rem;">
            {t('statistical_subtitle')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if not statistical_results:
        st.warning("⚠️ No se encontraron resultados estadísticos. Ejecuta el análisis completo primero.")
        return
    
    # Tabs para organizar los resultados
    tab1, tab2 = st.tabs([t("mcc_title"), t("mcnemar_title")])
    
    with tab1:
        st.markdown(f"### {t('mcc_title')}")
        st.markdown(f"""
        <div class="info-box-professional">
            <h4 style="margin-bottom: 1rem;">ℹ️ ¿Qué es el MCC?</h4>
            <p style="margin: 0;">
                {t('mcc_description')}
            </p>
            <ul style="margin-top: 0.5rem; padding-left: 1.5rem;">
                <li><strong>MCC = 1</strong>: {t('perfect_prediction')}</li>
                <li><strong>MCC > 0.5</strong>: {t('very_good_agreement')}</li>
                <li><strong>MCC = 0</strong>: {t('no_better_random')}</li>
                <li><strong>MCC = -1</strong>: {t('total_disagreement')}</li>
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
                             annotation_text=t('very_good_agreement'), annotation_position="top")
            
            st.plotly_chart(fig_mcc, use_container_width=True)
            
            # Tabla de resultados
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"#### {t('model_ranking')}")
                mcc_display = mcc_df.copy()
                mcc_display['MCC'] = mcc_display['MCC'].apply(lambda x: f"{x:.4f}")
                mcc_display[t('interpretation')] = mcc_df['MCC'].apply(
                    lambda x: t('excellent') if x > 0.5 else t('good') if x > 0.3 else t('regular')
                )
                st.dataframe(mcc_display, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown(f"### {t('mcnemar_title')}")
        st.markdown(f"""
        <div class="info-box-professional">
            <h4 style="margin-bottom: 1rem;">ℹ️ ¿Qué es la prueba de McNemar?</h4>
            <p style="margin: 0;">
                {t('mcnemar_description')}
            </p>
            <ul style="margin-top: 0.5rem; padding-left: 1.5rem;">
                <li><strong>p < 0.05</strong>: {t('statistically_significant')}</li>
                <li><strong>p ≥ 0.05</strong>: {t('no_significant_difference')}</li>
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
            st.markdown(f"#### {t('detailed_comparisons')}")
            
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

def display_training_results():
    """Muestra los resultados del entrenamiento realizado"""
    st.markdown(f"""
    <div class="results-section animate-fadeIn">
        <h2 style="text-align: center; color: var(--text-primary); margin-bottom: 1rem; font-weight: 700;">
            {t('training_results')}
        </h2>
        <p style="text-align: center; color: var(--text-secondary); font-size: 1.1rem; margin-bottom: 2rem;">
            {t('training_subtitle')}
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
                            <span>{t('time')}</span>
                            <strong>{metrics['time']}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                            <span>{t('parameters')}</span>
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
        tab_list = [t("general_comparison"), t("confusion_matrices"), t("training_histories"), t("dataset_info_tab")]
        if statistical_results:
            tab_list.append(t("statistical_analysis"))
        
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
                st.markdown(f"##### 🎯 {t('confusion_matrices')} por Modelo")
                
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
                st.markdown(f"##### 📉 {t('training_histories')}")
                
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
            st.markdown(f"##### 📊 Dataset SIPaKMeD - Estadísticas")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-container" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                    <div class="metric-label">{t('total_images')}</div>
                    <div class="metric-value">5,015</div>
                    <div style="font-size: 0.9rem; opacity: 0.9;">{t('complete_dataset')}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                    <div class="metric-label">{t('training')}</div>
                    <div class="metric-value">4,010</div>
                    <div style="font-size: 0.9rem; opacity: 0.9;">80% {t('dataset_percentage')}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-container" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                    <div class="metric-label">{t('validation')}</div>
                    <div class="metric-value">1,005</div>
                    <div style="font-size: 0.9rem; opacity: 0.9;">20% {t('dataset_percentage')}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Distribución por clases
            st.markdown(f"##### 📊 Distribución por Tipo de Célula")
            
            class_names_friendly = get_class_names_friendly()
            class_distribution = pd.DataFrame({
                'Tipo de Célula': list(class_names_friendly.values()),
                'Cantidad': [1003, 1003, 1003, 1003, 1003],  # Ajusta estos valores según tu dataset real
                'Porcentaje': ['20%', '20%', '20%', '20%', '20%']
            })
            
            st.dataframe(class_distribution, use_container_width=True, hide_index=True)
        
        # Tab de análisis estadístico si existe
        if statistical_results and len(tabs) > 4:
            with tabs[4]:
                display_statistical_analysis(statistical_results)

def enhance_cervical_cell_image(image):
    """Mejora específica para imágenes de células cervicales - OPTIMIZADA"""
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
        logger.error(f"Error en mejora de imagen: {e}")
        return np.array(image) if isinstance(image, Image.Image) else image

def preprocess_image(image, model_name):
    """Preprocesa la imagen según el modelo específico - OPTIMIZADA"""
    try:
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
    except Exception as e:
        logger.error(f"Error en preprocesamiento para {model_name}: {e}")
        raise

def predict_cervical_cells(image, models):
    """Realiza predicciones con todos los modelos disponibles - OPTIMIZADA"""
    predictions = {}
    class_names_friendly = get_class_names_friendly()
    clinical_info = get_clinical_info()
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (model_name, model) in enumerate(models.items()):
        try:
            status_text.text(f'{t("analyzing_ai").replace("🔍", "")} {model_name}...')
            processed_image = preprocess_image(image, model_name)
            pred = model.predict(processed_image, verbose=0)
            pred_class_idx = np.argmax(pred[0])
            pred_class = CLASS_NAMES[pred_class_idx]
            confidence = float(pred[0][pred_class_idx])

            predictions[model_name] = {
                'class': pred_class,
                'class_friendly': class_names_friendly[pred_class],
                'confidence': confidence,
                'probabilities': pred[0],
                'clinical_info': clinical_info[pred_class]
            }
            
            progress_bar.progress((i + 1) / len(models))
            
        except Exception as e:
            st.error(f"Error en predicción con {model_name}: {e}")
            logger.error(f"Error en predicción con {model_name}: {e}")
    
    progress_bar.empty()
    status_text.empty()
    
    return predictions

def create_interactive_plots(predictions):
    """Crea gráficos interactivos con Plotly - OPTIMIZADA"""
    models = list(predictions.keys())
    n_models = len(models)
    class_names_friendly = get_class_names_friendly()
    
    fig = make_subplots(
        rows=1, cols=n_models,
        subplot_titles=[f'{model}' for model in models],
        specs=[[{"type": "bar"} for _ in range(n_models)]]
    )
    
    colors_plot = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe']
    
    for idx, (model_name, pred) in enumerate(predictions.items()):
        friendly_names = [class_names_friendly[class_name] for class_name in CLASS_NAMES]
        
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
            'text': t("probability_distribution"),
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
    """Crea gráfico de consenso entre modelos - OPTIMIZADA"""
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
            'text': t("model_consensus"),
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
                text=f'{len(predictions)}<br>{t("models")}',
                x=0.5, y=0.5,
                font=dict(size=20, family='Inter, sans-serif', weight=700),
                showarrow=False
            )
        ]
    )
    
    return fig

def generate_pdf_report(predictions, image_info, patient_info=None, statistical_results=None, 
                       probability_fig=None, consensus_fig=None):
    """Genera un reporte en PDF multilenguaje con los resultados del análisis e imágenes del entrenamiento"""
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
    
    # Título del reporte (multilenguaje)
    story.append(Paragraph(t("pdf_title"), title_style))
    story.append(Paragraph(t("pdf_subtitle"), styles['Normal']))
    story.append(Spacer(1, 30))
    
    # Información del análisis (multilenguaje)
    fecha_analisis = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    info_data = [
        [t("analysis_date"), fecha_analisis],
        [t("system"), t("pdf_subtitle")],
        [t("analyzed_image"), image_info.get('filename', 'N/A')],
        [t("dimensions"), f"{image_info.get('size', 'N/A')}"],
        [t("format"), image_info.get('format', 'N/A')]
    ]
    
    if patient_info:
        info_data.extend([
            [t("patient"), patient_info.get('nombre', 'N/A')],
            [t("id"), patient_info.get('id', 'N/A')]
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
    
    # Resultados por modelo (multilenguaje)
    story.append(Paragraph(t("results_by_model"), heading_style))
    
    results_data = [[t("model"), t("cell_type"), t("confidence"), t("risk_level")]]
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
    
    # GRÁFICOS DEL ANÁLISIS (NUEVA FUNCIONALIDAD) - MULTILENGUAJE
    story.append(Paragraph(t("analysis_charts"), heading_style))
    
    # Lista para rastrear archivos temporales
    temp_files_to_clean = []
    
    # Incluir gráfico de probabilidades si existe
    if probability_fig:
        try:
            prob_img_path = save_plotly_as_image(probability_fig, "probability_distribution")
            if prob_img_path and os.path.exists(prob_img_path):
                story.append(Paragraph(t("probability_distribution_title"), styles['Normal']))
                story.append(Spacer(1, 12))
                img = RLImage(prob_img_path, width=6*inch, height=3.5*inch)
                story.append(img)
                story.append(Spacer(1, 20))
                temp_files_to_clean.append(prob_img_path)
            else:
                story.append(Paragraph(t("chart_not_available"), styles['Normal']))
                story.append(Spacer(1, 12))
        except Exception as e:
            logger.error(f"Error incluyendo gráfico de probabilidades: {e}")
            story.append(Paragraph(t("chart_error"), styles['Normal']))
            story.append(Spacer(1, 12))
    
    # Incluir gráfico de consenso si existe
    if consensus_fig:
        try:
            consensus_img_path = save_plotly_as_image(consensus_fig, "consensus_chart")
            if consensus_img_path and os.path.exists(consensus_img_path):
                story.append(Paragraph(t("model_consensus_title"), styles['Normal']))
                story.append(Spacer(1, 12))
                img = RLImage(consensus_img_path, width=4*inch, height=3*inch)
                story.append(img)
                story.append(Spacer(1, 20))
                temp_files_to_clean.append(consensus_img_path)
            else:
                story.append(Paragraph(t("chart_not_available"), styles['Normal']))
                story.append(Spacer(1, 12))
        except Exception as e:
            logger.error(f"Error incluyendo gráfico de consenso: {e}")
            story.append(Paragraph(t("chart_error"), styles['Normal']))
            story.append(Spacer(1, 12))
    
    # Interpretación clínica (multilenguaje)
    story.append(Paragraph(t("clinical_interpretation_title"), heading_style))
    
    # Obtener predicción más común
    prediction_counts = {}
    for pred in predictions.values():
        class_name = pred['class']
        prediction_counts[class_name] = prediction_counts.get(class_name, 0) + 1
    
    most_common = max(prediction_counts.items(), key=lambda x: x[1])
    consensus_class = most_common[0]
    consensus_count = most_common[1]
    
    clinical_info = get_clinical_info()[consensus_class]
    class_names_friendly = get_class_names_friendly()
    
    # Determinar resultado (multilenguaje)
    if consensus_class in ['parabasal', 'superficial_intermediate']:
        status = t("normal")
        color = colors.green
    elif consensus_class in ['metaplastic']:
        status = t("benign")
        color = colors.orange
    else:
        status = t("requires_attention")
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
    
    story.append(Paragraph(f"<b>{t('result')}: {status}</b>", resultado_style))
    story.append(Paragraph(f"{t('predominant_cell_type')} {class_names_friendly[consensus_class]}", styles['Normal']))
    story.append(Paragraph(f"{t('consensus_text')} {consensus_count}/{len(predictions)} {t('models')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Descripción clínica (multilenguaje)
    story.append(Paragraph(f"<b>{t('description_text')}</b> {clinical_info['descripcion']}", styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>{t('clinical_meaning_text')}</b> {clinical_info['significado']}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Recomendaciones (multilenguaje)
    story.append(Paragraph(t("recommendations_title"), heading_style))
    
    if status == t("requires_attention"):
        recomendaciones = [
            f"• {t('consult_specialist')}",
            f"• {t('additional_studies')}",
            f"• {t('regular_followup')}",
            f"• {t('pathologist_interpretation')}"
        ]
    else:
        recomendaciones = [
            f"• {t('routine_controls')}",
            f"• {t('continue_screening')}",
            f"• {t('consult_doctor')}"
        ]
    
    for rec in recomendaciones:
        story.append(Paragraph(rec, styles['Normal']))
    
    # SECCIÓN DE ENTRENAMIENTO (multilenguaje)
    story.append(Spacer(1, 40))
    story.append(Paragraph(t("training_info_title"), heading_style))
    
    # Métricas de entrenamiento (multilenguaje)
    story.append(Paragraph(t("performance_metrics"), heading_style))
    
    training_metrics_data = [
        [t("model"), t("accuracy"), t("time_col"), t("parameters_col")],
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
    
    # Dataset información (multilenguaje)
    story.append(Paragraph(t("dataset_sipakmed"), heading_style))
    story.append(Paragraph(t("dataset_info_text"), styles['Normal']))
    
    # SECCIÓN DE ANÁLISIS ESTADÍSTICO RESTAURADA CON TABLA DE MCNEMAR (multilenguaje)
    if statistical_results:
        story.append(Spacer(1, 30))
        story.append(Paragraph(t("statistical_analysis_pdf"), heading_style))
        
        # Matthews Correlation Coefficient (multilenguaje)
        story.append(Paragraph(t("mcc_full"), heading_style))
        story.append(Paragraph(t("mcc_explanation"), styles['Normal']))
        story.append(Spacer(1, 12))
        
        mcc_scores = statistical_results.get('mcc_scores', {})
        if mcc_scores:
            # Crear tabla de MCC (multilenguaje)
            mcc_data = [[t("model"), 'MCC', t("interpretation")]]
            for model, mcc in sorted(mcc_scores.items(), key=lambda x: x[1], reverse=True):
                interpretation = t("excellent") if mcc > 0.5 else t("good") if mcc > 0.3 else t("regular")
                mcc_data.append([model, f'{mcc:.4f}', interpretation])
            
            mcc_table = Table(mcc_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
            mcc_table.setStyle(TableStyle([
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
            story.append(mcc_table)
        
        # Prueba de McNemar - TABLA RESTAURADA (multilenguaje)
        story.append(Spacer(1, 20))
        story.append(Paragraph(t("mcnemar_full"), heading_style))
        story.append(Paragraph(t("mcnemar_explanation"), styles['Normal']))
        story.append(Spacer(1, 12))
        
        mcnemar_results = statistical_results.get('mcnemar_tests', {})
        if mcnemar_results:
            # CREAR MATRIZ DE COMPARACIÓN (TABLA RESTAURADA) (multilenguaje)
            models = ['MobileNetV2', 'ResNet50', 'EfficientNetB0']
            
            # Encabezados de la tabla
            mcnemar_matrix_data = [['Comparación'] + models]
            
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
            
            # Crear y estilizar la tabla
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
            story.append(Paragraph(f"<i>{t('legend_text')}</i>", styles['Normal']))
            
            # Detalles de comparaciones significativas (multilenguaje)
            story.append(Spacer(1, 15))
            story.append(Paragraph(t("detailed_comparisons_text"), heading_style))
            
            for comparison, result in mcnemar_results.items():
                models_compared = comparison.replace('_vs_', ' vs ')
                story.append(Paragraph(
                    f"<b>{models_compared}</b>: {result['interpretation']} "
                    f"(χ²={result['statistic']:.2f}, p={result['p_value']:.4f})",
                    styles['Normal']
                ))
    
    # Cargar imágenes del entrenamiento (multilenguaje)
    training_images = load_training_images()
    
    if training_images:
        # Comparación de modelos (multilenguaje)
        if 'model_comparison' in training_images and training_images['model_comparison']:
            story.append(PageBreak())
            story.append(Paragraph(t("model_comparison"), heading_style))
            story.append(Spacer(1, 20))
            
            try:
                img_path = training_images['model_comparison']['path']
                img = RLImage(img_path, width=5.5*inch, height=3.5*inch)
                story.append(img)
                story.append(Spacer(1, 12))
                story.append(Paragraph(f"<i>{t('comparison_precision')}</i>", styles['Normal']))
            except Exception as e:
                logger.error(f"Error incluyendo imagen de comparación: {e}")
        
        # Matrices de confusión (multilenguaje)
        if 'confusion_matrices' in training_images and training_images['confusion_matrices']:
            story.append(PageBreak())
            story.append(Paragraph(t("confusion_matrices_title"), heading_style))
            story.append(Spacer(1, 20))
            
            for img_info in training_images['confusion_matrices']:
                try:
                    story.append(Paragraph(f"<b>{img_info['model']}</b>", styles['Normal']))
                    story.append(Spacer(1, 12))
                    
                    img = RLImage(img_info['path'], width=4*inch, height=3*inch)
                    story.append(img)
                    story.append(Spacer(1, 20))
                except Exception as e:
                    logger.error(f"Error incluyendo matriz de confusión: {e}")
        
        # Historiales de entrenamiento (multilenguaje)
        if 'training_histories' in training_images and training_images['training_histories']:
            story.append(PageBreak())
            story.append(Paragraph(t("training_histories_title"), heading_style))
            story.append(Spacer(1, 20))
            
            for img_info in training_images['training_histories']:
                try:
                    story.append(Paragraph(f"<b>{img_info['model']}</b>", styles['Normal']))
                    story.append(Spacer(1, 12))
                    
                    img = RLImage(img_info['path'], width=4.5*inch, height=3*inch)
                    story.append(img)
                    story.append(Spacer(1, 20))
                except Exception as e:
                    logger.error(f"Error incluyendo historial de entrenamiento: {e}")
    
    # Disclaimer final (multilenguaje)
    story.append(PageBreak())
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.red,
        alignment=TA_CENTER,
        spaceAfter=30
    )
    
    story.append(Paragraph(f"<b>{t('important_notice')}</b> {t('disclaimer_text')}", disclaimer_style))
    
    # Construir PDF
    try:
        doc.build(story)
        buffer.seek(0)
        
        # Limpiar archivos temporales después de construir el PDF
        for temp_file in temp_files_to_clean:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    logger.info(f"Archivo temporal eliminado: {temp_file}")
            except Exception as e:
                logger.warning(f"No se pudo eliminar archivo temporal {temp_file}: {e}")
        
        return buffer
        
    except Exception as e:
        # Limpiar archivos temporales en caso de error
        for temp_file in temp_files_to_clean:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass
        raise e

def display_download_section(predictions, image_info, probability_fig=None, consensus_fig=None):
    """Muestra la sección de descarga de reportes"""
    st.markdown(f"### {t('download_report')}")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.expander(t("patient_info"), expanded=False):
            patient_name = st.text_input(t("patient_name"), placeholder="Ej: María García")
            patient_id = st.text_input(t("patient_id"), placeholder="Ej: HC-001234")
            
            patient_info = None
            if patient_name or patient_id:
                patient_info = {
                    'nombre': patient_name if patient_name else 'N/A',
                    'id': patient_id if patient_id else 'N/A'
                }
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button(t("generate_pdf"), use_container_width=True, type="primary"):
            try:
                with st.spinner(t("generating_report")):
                    # Cargar resultados estadísticos si existen
                    statistical_results = load_statistical_results()
                    
                    # Mostrar advertencia sobre gráficos si es necesario
                    try:
                        # Test rápido de plotly.io
                        test_fig = go.Figure()
                        pio.to_image(test_fig, format='png', width=100, height=100)
                    except Exception as e:
                        st.warning("""
                        ⚠️ **Nota**: Los gráficos podrían no incluirse en el PDF debido a limitaciones del sistema.
                        Para incluir gráficos, instala: `pip install kaleido`
                        
                        El reporte se generará sin los gráficos pero con todo el contenido restante.
                        """)
                        logger.warning(f"Plotly imagen test falló: {e}")
                    
                    # Generar PDF con resultados estadísticos y gráficos
                    pdf_buffer = generate_pdf_report(
                        predictions, image_info, patient_info, statistical_results, 
                        probability_fig, consensus_fig
                    )
                    fecha = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"reporte_celulas_cervicales_{fecha}.pdf"
                    
                    st.download_button(
                        label=t("download_pdf"),
                        data=pdf_buffer.getvalue(),
                        file_name=filename,
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                    st.success(t("report_generated"))
                    
            except Exception as e:
                st.error(f"{t('pdf_error')} {str(e)}")
                logger.error(f"Error generando PDF: {str(e)}")

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
    
    clinical_info = get_clinical_info()[consensus_class]
    class_names_friendly = get_class_names_friendly()
    
    # Determinar el tipo de resultado
    if consensus_class in ['parabasal', 'superficial_intermediate']:
        status = t("normal")
        status_class = "status-normal"
        recommendation_type = "info"
    elif consensus_class in ['metaplastic']:
        status = t("benign")
        status_class = "status-warning"
        recommendation_type = "info"
    else:
        status = t("requires_attention")
        status_class = "status-danger"
        recommendation_type = "warning"
    
    # Card de resultado principal
    st.markdown(f"""
    <div class="professional-card" style="text-align: center;">
        <h2 style="color: var(--text-primary); margin-bottom: 1rem;">{t('clinical_interpretation')}</h2>
        <div class="status-badge {status_class}" style="font-size: 1.2rem; margin: 1rem 0;">
            {clinical_info['icon']} {status}
        </div>
        <p style="font-size: 1.1rem; color: var(--text-primary); margin: 1rem 0;">
            <strong>{class_names_friendly[consensus_class]}</strong>
        </p>
        <p style="color: var(--text-secondary);">
            {t('consensus')}: {consensus_count} de {len(predictions)} {t('models_agree')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Información clínica
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="professional-card">
            <h4 style="color: var(--primary-color); margin-bottom: 1rem;">{t('description')}</h4>
            <p style="color: var(--text-secondary);">{clinical_info['descripcion']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="professional-card">
            <h4 style="color: var(--primary-color); margin-bottom: 1rem;">{t('clinical_meaning')}</h4>
            <p style="color: var(--text-secondary);">{clinical_info['significado']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recomendaciones
    if recommendation_type == "warning":
        st.markdown(f"""
        <div class="warning-box-professional">
            <h4 style="margin-bottom: 1rem;">{t('important_recommendations')}</h4>
            <ul style="margin: 0; padding-left: 1.5rem;">
                <li>{t('consult_specialist')}</li>
                <li>{t('additional_studies')}</li>
                <li>{t('regular_followup')}</li>
                <li>{t('pathologist_interpretation')}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="info-box-professional">
            <h4 style="margin-bottom: 1rem;">{t('recommendations')}</h4>
            <ul style="margin: 0; padding-left: 1.5rem;">
                <li>{t('routine_controls')}</li>
                <li>{t('continue_screening')}</li>
                <li>{t('consult_doctor')}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def create_sidebar():
    """Crea el sidebar multilenguaje optimizado"""
    with st.sidebar:
        # Selector de idioma al principio
        st.markdown("### 🌍 Language / Idioma")
        selected_lang = st.selectbox(
            "Select language:",
            options=list(LANGUAGES.keys()),
            format_func=lambda x: LANGUAGES[x],
            index=list(LANGUAGES.keys()).index(get_language()),
            key="language_selector"
        )
        
        # Actualizar idioma en session state
        if selected_lang != st.session_state.language:
            st.session_state.language = selected_lang
            st.rerun()
        
        st.markdown(f"""
        <div class="sidebar-header">
            {t('sidebar_title')}
        </div>
        """, unsafe_allow_html=True)
        
        # Información de tipos de células
        st.markdown(f"### {t('cell_types')}")
        
        clinical_info = get_clinical_info()
        class_names_friendly = get_class_names_friendly()
        
        for class_key, info in clinical_info.items():
            with st.expander(f"{info['icon']} {class_names_friendly[class_key]}"):
                st.markdown(f"""
                **{t('high_risk' if info['riesgo'] == t('high_risk') else 'moderate_risk' if info['riesgo'] == t('moderate_risk') else 'low_risk' if info['riesgo'] == t('low_risk') else 'normal_risk')}:** {info['riesgo']}  
                **{t('description')}:** {info['descripcion']}
                """)
        
        # Configuración
        st.markdown(f"### {t('configuration')}")
        
        enhance_image = st.checkbox(
            t("clahe_enhancement"),
            value=True,
            help=t("clahe_help")
        )
        
        # Información del sistema
        st.markdown(f"### {t('system_info')}")
        st.info(f"""
        **{t('models_info')}**  
        **{t('dataset_info')}**  
        **{t('accuracy_info')}**  
        **GPU:** {'✅' if tf.config.list_physical_devices('GPU') else '❌'}
        """)
        
        # Disclaimer
        st.markdown(f"### {t('legal_notice')}")
        st.error(t("legal_text"))
        
        return enhance_image

def main():
    """FUNCIÓN PRINCIPAL MULTILENGUAJE - MANTIENE TODA LA FUNCIONALIDAD ORIGINAL"""
    # Header principal mejorado
    st.markdown(f"""
    <div class="main-header">
        {t('main_title')}
    </div>
    <p class="subtitle">
        {t('subtitle')}
    </p>
    """, unsafe_allow_html=True)
    
    # Crear sidebar y obtener configuración
    enhance_image = create_sidebar()
    
    # Contenido principal
    # Sección de introducción mejorada
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div class="professional-card" style="text-align: center;">
            <h3 style="color: var(--primary-color); margin-bottom: 1rem;">
                🎯 {t('system_ready')}
            </h3>
            <p style="color: var(--text-secondary); line-height: 1.8;">
                {t('upload_description')}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Cargar modelos con diseño mejorado
    st.markdown(f"### 🤖 {t('ai_system')}")
    
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.spinner(t('loading_models')):
                models = load_models()
        
        if not models:
            st.error(t("model_error"))
            st.markdown(f"""
            <div class="warning-box-professional" style="background: linear-gradient(135deg, #FC424A 0%, #FF6B6B 100%);">
                <h4>🚨 {t('model_error')}</h4>
                <p>{t('model_error_solution')}</p>
                <ul style="margin: 1rem 0; padding-left: 1.5rem;">
                    <li>{t('verify_files')}</li>
                    <li>{t('run_training')}</li>
                    <li>{t('restart_app')}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            st.stop()
        
        # Mostrar modelos cargados de forma más visual
        st.markdown(f"""
        <div class="professional-card" style="text-align: center;">
            <h4 style="color: var(--primary-color); margin-bottom: 1rem;">
                ✅ {t('system_ready_msg')}
            </h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        metrics_display = [
            (f"🧠 {t('models')}", f"{len(models)}", t('models_loaded')),
            ("⚡ Modo", "GPU" if tf.config.list_physical_devices('GPU') else "CPU", t('processing_mode')),
            ("🎯 Precisión", "84-90%", t('accuracy_range')),
            ("📊 Clases", "5", t('cell_types_count'))
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
    
    # Mostrar resultados del entrenamiento COMPLETOS (matrices, estadísticas, etc.)
    display_training_results()
    
    # Sección de análisis
    st.markdown(f"### {t('image_analysis')}")
    
    uploaded_file = st.file_uploader(
        t("upload_instruction"),
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help=t("upload_help")
    )
    
    if uploaded_file is None:
        # Mostrar instrucciones cuando no hay archivo
        st.markdown(f"""
        <div class="professional-card" style="text-align: center; padding: 3rem;">
            <h3 style="color: var(--primary-color); margin-bottom: 1rem;">
                {t('waiting_image')}
            </h3>
            <p style="color: var(--text-secondary); line-height: 1.8; max-width: 600px; margin: 0 auto;">
                {t('upload_description')}
            </p>
            <div style="margin-top: 2rem; padding: 1rem; background: rgba(102, 126, 234, 0.1); 
                        border-radius: 8px; border: 1px solid rgba(102, 126, 234, 0.3);">
                <p style="color: var(--primary-color); font-weight: 600; margin: 0;">
                    {t('tip_quality')}
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
                
                with st.spinner(t('applying_clahe')):
                    enhanced_img = enhance_cervical_cell_image(original_image)
                    enhanced_pil = Image.fromarray(enhanced_img.astype(np.uint8))
                    st.image(enhanced_pil, use_container_width=True)
                    analysis_image = enhanced_pil
            else:
                analysis_image = original_image
            
            # Realizar predicciones
            st.markdown(f"#### {t('analyzing_ai')}")
            predictions = predict_cervical_cells(analysis_image, models)
        
        # Mostrar resultados
        if predictions:
            st.markdown(f"### {t('analysis_results')}")
            
            # Cards de resultados por modelo
            clinical_info = get_clinical_info()
            class_names_friendly = get_class_names_friendly()
            
            cols = st.columns(len(predictions))
            for i, (model_name, pred) in enumerate(predictions.items()):
                with cols[i]:
                    pred_clinical_info = pred['clinical_info']
                    
                    # Determinar color según riesgo
                    risk_colors = {
                        t('high_risk'): "#FC424A",
                        t('moderate_risk'): "#FFAB00",
                        t('low_risk'): "#0066CC",
                        t('normal_risk'): "#00D25B"
                    }
                    
                    color = risk_colors.get(pred_clinical_info['riesgo'], "#6C63FF")
                    
                    # Determinar clase de status badge
                    if pred_clinical_info['riesgo'] in [t('normal_risk'), t('low_risk')]:
                        status_class = 'status-normal'
                    elif pred_clinical_info['riesgo'] == t('moderate_risk'):
                        status_class = 'status-warning'
                    else:
                        status_class = 'status-danger'
                    
                    st.markdown(f"""
                    <div class="professional-card" style="text-align: center;">
                        <div style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%); 
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
                        <div class="status-badge {status_class}">
                            {pred_clinical_info['icon']} {pred_clinical_info['riesgo']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Gráficos interactivos
            st.markdown(f"### {t('visual_analysis')}")
            
            tab1, tab2 = st.tabs([t("probability_distribution"), t("model_consensus")])
            
            # Crear los gráficos
            probability_fig = create_interactive_plots(predictions)
            consensus_fig = create_consensus_chart(predictions)
            
            with tab1:
                st.plotly_chart(probability_fig, use_container_width=True)
            
            with tab2:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.plotly_chart(consensus_fig, use_container_width=True)
            
            # Interpretación clínica
            display_clinical_interpretation(predictions)
            
            # Sección de descarga CON PDF COMPLETO Y GRÁFICOS
            display_download_section(predictions, image_info, probability_fig, consensus_fig)
    
    # Footer profesional multilenguaje
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; padding: 2rem 0; color: var(--text-secondary);'>
        <p style='font-size: 1.1rem; margin-bottom: 0.5rem;'>
            <strong>{t('footer_title')}</strong>
        </p>
        <p style='font-size: 0.9rem;'>
            {t('footer_subtitle')}
        </p>
        <p style='font-size: 0.85rem; color: var(--text-secondary); margin-top: 1rem;'>
            {t('footer_disclaimer')}
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Error crítico: {str(e)}")
        logger.error(f"Error crítico en main: {str(e)}")
