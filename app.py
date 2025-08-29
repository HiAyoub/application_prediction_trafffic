"""
Traffic Prediction MLOps Application
====================================

A comprehensive Streamlit application for traffic prediction using MLOps best practices.
Integrates preprocessing, modeling, and geospatial visualization.
"""

import streamlit as st
# Inject Font Awesome for icons everywhere
st.markdown(
    '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">',
    unsafe_allow_html=True
)

import pandas as pd
import numpy as np
import os
import tempfile
from pathlib import Path

import logging
from datetime import datetime
import yaml
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
# Ensure logs directory exists before configuring logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import custom modules
from src.data_processor import DataProcessor
from src.model_manager import ModelManager
from src.visualization import GeospatialVisualizer, MetricsVisualizer
from src.utils import detect_coordinate_columns, validate_file, save_uploaded_file
from src.chatbot import create_chatbot_interface, notify_step_completion
from src.fusion_donnees_fcd_debits import (
    calculer_moyenne_horaire_periode,
    cree_dataset_unique_horaire,
    fusionner_debits_fcd,
    charger_fichiers
)

# Page configuration
st.set_page_config(
    page_title="Traffic Prediction MLOps",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application function"""
    
    # Title and description
    st.markdown("""
    <span style='font-size:28px;'><i class="fa fa-car"></i> <b>Traffic Prediction MLOps Platform</b></span>
    <br>
    <span style='font-size:16px'>Cette application vous guide étape par étape pour importer, traiter, modéliser et visualiser vos données de trafic, même si vous n'êtes pas expert en informatique.</span>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'model_results' not in st.session_state:
        st.session_state.model_results = None
    
    # Sidebar configuration
    with st.sidebar:
        logo_path = "assets/logo.png"
        if os.path.exists(logo_path):
            st.image(logo_path, width=120)
        else:
            st.markdown('<i class="fa fa-car fa-2x"></i>', unsafe_allow_html=True)
        st.markdown('<span style="font-size:22px;"><i class="fa fa-cogs"></i> <b>Configuration</b></span>', unsafe_allow_html=True)
        st.markdown("**Suivez les étapes ci-dessous pour configurer votre analyse.**")
        
        # Configuration file selection
        config_option = st.selectbox(
            "Select Configuration Mode",
            ["Default", "Custom Upload"],
            help="Choose to use default configurations or upload custom YAML files",
            key="config_mode_unique_1"
        )
        
        preprocessing_config = None
        modeling_config = None
        
        if config_option == "Custom Upload":
            st.markdown("Télécharger les fichiers de configuration")
            
            prep_config_file = st.file_uploader(
                "Configuration du prétraitement (YAML)",
                type=['yaml', 'yml'],
                key="prep_config"
            )
            
            model_config_file = st.file_uploader(
                "Configuration du modèle (YAML)",
                type=['yaml', 'yml'],
                key="model_config"
            )
            
            if prep_config_file:
                preprocessing_config = yaml.safe_load(prep_config_file)
            if model_config_file:
                modeling_config = yaml.safe_load(model_config_file)
        else:
            # Load default configurations
            try:
                with open('config/config_preprocess.yaml', 'r') as f:
                    preprocessing_config = yaml.safe_load(f)
                with open('config/config_modeling.yaml', 'r') as f:
                    modeling_config = yaml.safe_load(f)
            except FileNotFoundError as e:
                st.error(f"Fichiers de configuration par défaut introuvables : {e}")
                st.stop()
        
        # Display configuration status
        if preprocessing_config and modeling_config:
            st.success("✅ Configurations chargées avec succès")
        else:
            st.warning("⚠️ Veuillez vous assurer que tous les fichiers de configuration sont chargés")
    
    # Add chatbot sidebar
    with st.sidebar:
        st.markdown("---")
        st.info("💡 Besoin d'aide ? Utilisez le chatbot ci-dessous ou survolez les options pour plus d'explications.")
        create_chatbot_interface()
    
    # Main content tabs
    # Use st.markdown to inject Font Awesome CSS into the main page (not just sidebar)
    st.markdown(
        """
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
        <style>
        .stTabs [data-baseweb="tab"] span {
            font-size: 18px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        'Import de données', 
        'Traitement des données', 
        'Entraînement du modèle', 
        'Résultats & Métriques', 
        'Analyse géospatiale',
        'Fusion Débits & FCD'
    ])
    
    with tab1:
        data_upload_section(preprocessing_config)
    
    with tab2:
        data_processing_section(preprocessing_config)
    
    with tab3:
        model_training_section(modeling_config)
    
    with tab4:
        results_metrics_section()
    
    with tab5:
        geospatial_analysis_section(modeling_config)
    
    with tab6:
        fusion_debits_fcd_section()

def data_upload_section(preprocessing_config):
    st.markdown('<span style="font-size:20px;"><i class="fa fa-upload"></i> <b>Étape 1 : Importation et Validation des Données</b></span>', unsafe_allow_html=True)
    st.markdown("""
    <span style='font-size:16px'>Déposez vos fichiers CSV de trafic ci-dessous ou utilisez le résultat de la fusion Débits & FCD.<br>
    <b>Astuce :</b> Vous pouvez glisser-déposer plusieurs fichiers à la fois.</span>
    """, unsafe_allow_html=True)

    # --- Choix de la source de données à importer ---
    fusion_available = st.session_state.get('fusion_data') is not None
    import_options = []
    if fusion_available:
        import_options.append("Fichier fusionné Débits & FCD")
    import_options.append("Importer un fichier CSV")
    selected_import = st.radio(
        "Source des données à importer",
        import_options,
        horizontal=True,
        key="import_data_source"
    )

    uploaded_files = None
    fusion_data = None

    if selected_import == "Fichier fusionné Débits & FCD" and fusion_available:
        fusion_data = st.session_state['fusion_data']
        st.success("Le fichier fusionné Débits & FCD est prêt à être utilisé.")
    elif selected_import == "Importer un fichier CSV":
        uploaded_files = st.file_uploader(
            "Choisissez des fichiers CSV",
            type=['csv'],
            accept_multiple_files=True,
            help="Déposez vos fichiers de données de trafic ici. Format CSV recommandé."
        )

    # --- Utilisation du fichier fusionné ---
    if fusion_data:
        # Affiche un aperçu et permet la confirmation
        st.subheader("Aperçu du fichier fusionné")
        df = fusion_data['dataframe']
        coord_info = fusion_data['coordinates']
        # Correction Arrow: conversion geometry en str pour affichage
        df_preview = df.copy()
        if 'geometry' in df_preview.columns:
            df_preview['geometry'] = df_preview['geometry'].apply(lambda g: g.wkt if hasattr(g, 'wkt') else str(g))
        st.dataframe(df_preview.head(10).astype(str), use_container_width=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total des lignes", len(df))
        with col2:
            st.metric("Total des colonnes", len(df.columns))
        with col3:
            st.metric("Valeurs manquantes", df.isnull().sum().sum())
        if st.button('Confirmer l\'utilisation du fichier fusionné', type="primary"):
            st.session_state.uploaded_data = {
                'dataframe': df,
                'filename': "dataset_debit_fcd.csv",
                'coordinates': coord_info
            }
            st.session_state['original_data'] = df.copy()
            st.success("Données confirmées et prêtes à être traitées !")
            
            st.rerun()
            
        return  # On ne propose pas l'import CSV si fusion choisie

    # --- Import classique CSV (si pas fusion ou choix explicite) ---
    if uploaded_files:
        # Utilise st.subheader pour les sous-titres sans HTML
        st.subheader("Aperçu et validation des fichiers")
        
        file_info = []
        valid_files = []
        
        for file in uploaded_files:
            try:
                # Notify chatbot about file upload step
                notify_step_completion('file_upload', f"Processing file: {file.name}")
                
                # Validate file
                df = pd.read_csv(file)
                is_valid, message = validate_file(df, preprocessing_config)
                
                # Notify chatbot about validation step
                notify_step_completion('data_validation', f"Checking data quality for {file.name}")
                
                # Detect coordinate columns
                coord_cols = detect_coordinate_columns(df)
                notify_step_completion('coordinate_detection', f"Found coordinates: {coord_cols}")
                
                file_info.append({
                    'Filename': file.name,
                    'Size': f"{file.size / 1024:.1f} Ko",
                    'Rows': len(df),
                    'Columns': len(df.columns),
                    'Coordinates': f"{coord_cols['lat']}, {coord_cols['lon']}" if coord_cols['lat'] and coord_cols['lon'] else "Non détectées",
                    'Valid': "✅" if is_valid else "❌",
                    'Message': message
                })
                
                if is_valid:
                    valid_files.append((file, df, coord_cols))
                    
            except Exception as e:
                file_info.append({
                    'Filename': file.name,
                    'Size': f"{file.size / 1024:.1f} Ko",
                    'Rows': 'Erreur',
                    'Columns': 'Erreur',
                    'Coordinates': 'Erreur',
                    'Valid': "❌",
                    'Message': f"Erreur lors de la lecture du fichier: {str(e)}"
                })
        
        # Display file information
        info_df = pd.DataFrame(file_info)
        st.dataframe(info_df.astype(str), use_container_width=True)
        
        if valid_files:
            st.success(f"✅ {len(valid_files)} fichier(s) valide(s) prêt(s) à être traité(s)")
            
            # Select primary dataset
            if len(valid_files) > 1:
                selected_file_idx = st.selectbox(
                    "Sélectionnez le jeu de données principal pour le traitement",
                    range(len(valid_files)),
                    format_func=lambda x: valid_files[x][0].name,
                    key="selected_file_idx"
                )
                selected_file, selected_df, coord_info = valid_files[selected_file_idx]
            else:
                selected_file, selected_df, coord_info = valid_files[0]
            
            # Display data preview
            st.subheader("Aperçu des Données")
            df_preview = selected_df.copy()
            if 'geometry' in df_preview.columns:
                df_preview['geometry'] = df_preview['geometry'].apply(lambda g: g.wkt if hasattr(g, 'wkt') else str(g))
            st.dataframe(df_preview.head(10).astype(str), use_container_width=True)
            
            # Data statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total des lignes", len(selected_df))
            with col2:
                st.metric("Total des colonnes", len(selected_df.columns))
            with col3:
                st.metric("Valeurs manquantes", selected_df.isnull().sum().sum())
            
            # Store in session state
            if st.button('Confirmer la sélection des données', type="primary"):
                st.session_state.uploaded_data = {
                    'dataframe': selected_df,
                    'filename': selected_file.name,
                    'coordinates': coord_info
                }
                st.success("Données confirmées et prêtes à être traitées !")
                st.rerun()
        else:
            st.error("❌ Aucun fichier valide trouvé. Veuillez vérifier le format de vos données et réessayer.")

def data_processing_section(preprocessing_config):
    st.markdown('<span style="font-size:20px;"><i class="fa fa-sliders"></i> <b>Étape 2 : Pipeline de Traitement des Données</b></span>', unsafe_allow_html=True)
    st.markdown("""
    <span style='font-size:16px'>Personnalisez le traitement de vos données.<br>
    <b>Astuce :</b> Les options avancées sont disponibles ci-dessous.</span>
    """, unsafe_allow_html=True)
    
    # --- Choix de la source de données ---
    data_sources = []
    data_labels = []
    if st.session_state.get('fusion_data'):
        data_sources.append(st.session_state.fusion_data)
        data_labels.append("Données fusionnées")
    if st.session_state.get('uploaded_data'):
        data_sources.append(st.session_state.uploaded_data)
        data_labels.append("Données importées")
    if not data_sources:
        st.warning("⚠️ Veuillez d'abord télécharger ou fusionner vos données.")
        return
    default_idx = 0 if st.session_state.get('fusion_data') else 0
    selected_idx = st.radio("Sélectionnez la source de données à traiter :", data_labels, index=default_idx, horizontal=True)
    selected_data = data_sources[data_labels.index(selected_idx)]

    # --- Fin choix source ---

    # Display current data info
    df = selected_data['dataframe']
    st.info(f"📄 Traitement : {selected_data['filename']} ({len(df)} lignes, {len(df.columns)} colonnes)")
    
    # Processing options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Options de Traitement")
        run_quality_checks = st.checkbox("Analyse de la Qualité des Données", value=True)
        run_drift_detection = st.checkbox("Détection de Dérive des Données", value=True)
        run_feature_engineering = st.checkbox("Ingénierie des Caractéristiques", value=True)
        run_temporal_analysis = st.checkbox("Analyse Temporelle", value=False)
    
    with col2:
        st.subheader("Options avancées")
        with st.expander("Afficher/Masquer les options avancées"):
            remove_outliers = st.checkbox("Supprimer les Valeurs Abérantes", value=False)
            handle_missing = st.selectbox("Stratégie pour les Valeurs Manquantes", ["drop", "mean", "median", "mode"],key='handle_missing')
            feature_selection = st.checkbox("Sélection Automatique des Caractéristiques", value=False)
    
    # Process data button
    if st.button('Démarrer le Traitement', type="primary"):
        try:
            with st.spinner("Traitement des données... Cela peut prendre quelques instants."):
                # Initialize data processor
                processor = DataProcessor(preprocessing_config)
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Processing steps
                processed_data = df.copy()
                reports = {}
                
                # Step 1: Data Quality Analysis
                if run_quality_checks:
                    status_text.text("Exécution de l'analyse de la qualité des données...")
                    progress_bar.progress(20)
                    notify_step_completion('data_preprocessing', "Exécution de vérifications complètes de la qualité des données")
                    quality_report = processor.run_quality_tests(processed_data)
                    reports['quality'] = quality_report
                
                # Step 2: Feature Engineering
                if run_feature_engineering:
                    status_text.text("Ingénierie des caractéristiques...")
                    progress_bar.progress(40)
                    notify_step_completion('feature_engineering', "Création de caractéristiques avancées à partir de vos données")
                    processed_data = processor.engineer_features(processed_data)
                
                # Step 3: Data Drift Detection
                if run_drift_detection:
                    status_text.text("Détection de dérive des données...")
                    progress_bar.progress(60)
                    drift_report = processor.detect_drift(processed_data)
                    reports['drift'] = drift_report
                
                # Step 4: Handle missing values and outliers
                status_text.text("Nettoyage des données...")
                progress_bar.progress(80)
                
                if remove_outliers:
                    processed_data = processor.remove_outliers(processed_data)
                
                processed_data = processor.handle_missing_values(processed_data, strategy=handle_missing)
                
                # Step 5: Feature selection
                if feature_selection:
                    status_text.text("Sélection des caractéristiques...")
                    processed_data = processor.select_features(processed_data)
                
                progress_bar.progress(100)
                status_text.text("Traitement terminé !")
                
                # Store results
                st.session_state.processed_data = {
                    'dataframe': processed_data,
                    'reports': reports,
                    'original_shape': df.shape,
                    'processed_shape': processed_data.shape
                }
                st.session_state.data_processed = True
                
                st.success("✅ Traitement des données terminé avec succès !")
                
                # Display summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Lignes originales", df.shape[0])
                with col2:
                    st.metric("Lignes traitées", processed_data.shape[0], 
                             delta=processed_data.shape[0] - df.shape[0])
                with col3:
                    st.metric("Colonnes originales", df.shape[1])
                with col4:
                    st.metric("Colonnes traitées", processed_data.shape[1], 
                             delta=processed_data.shape[1] - df.shape[1])
                
        except Exception as e:
            logger.error(f"Error during data processing: {str(e)}")
            st.error(f"❌ Échec du traitement : {str(e)}")
    
    # Display processing results
    if st.session_state.data_processed:
        st.subheader("Résultats du Traitement")
        
        # Data quality report
        if 'quality' in st.session_state.processed_data['reports']:
            with st.expander("Rapport sur la Qualité des Données", expanded=False):
                quality_report = st.session_state.processed_data['reports']['quality']
                
                if 'summary' in quality_report:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Tests Réussis", quality_report['summary'].get('tests_passed', 0))
                    with col2:
                        st.metric("Tests Échoués", quality_report['summary'].get('tests_failed', 0))
                
                # Display individual test results
                if 'tests' in quality_report:
                    for test_name, test_result in quality_report['tests'].items():
                        if isinstance(test_result, dict) and 'passed' in test_result:
                            status = "✅" if test_result['passed'] else "❌"
                            st.write(f"{status} **{test_name}**: {test_result.get('message', 'Pas de détails')}")
        
        # Data drift report
        if 'drift' in st.session_state.processed_data['reports']:
            with st.expander("Rapport de Dérive des Données", expanded=False):
                drift_report = st.session_state.processed_data['reports']['drift']
                
                if 'summary' in drift_report:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Dérive Détectée", drift_report['summary'].get('drift_detected', 0))
                    with col2:
                        st.metric("Caractéristiques Stables", drift_report['summary'].get('stable_features', 0))

def model_training_section(modeling_config):
    st.markdown('<span style="font-size:20px;"><i class="fa fa-robot"></i> <b>Étape 3 : Entraînement & Évaluation du Modèle</b></span>', unsafe_allow_html=True)
    st.markdown("""
    <span style='font-size:16px'>Sélectionnez vos modèles et paramètres d'entraînement.<br>
    <b>Astuce :</b> Vous pouvez comparer plusieurs modèles en même temps.</span>
    """, unsafe_allow_html=True)
    
    # --- Choix de la source de données ---
    data_sources = []
    data_labels = []
    if st.session_state.get('fusion_data'):
        data_sources.append(st.session_state.fusion_data)
        data_labels.append("Données fusionnées")
    if st.session_state.get('processed_data'):
        # processed_data doit être associé à la source sélectionnée
        # On suppose que processed_data['dataframe'] correspond à la dernière source traitée
        # On propose donc la même logique que pour data_processing_section
        # Mais si processed_data existe, on l'utilise pour la source sélectionnée
        pass
    if st.session_state.get('uploaded_data'):
        data_sources.append(st.session_state.uploaded_data)
        data_labels.append("Données importées")
    if not data_sources:
        st.warning("⚠️ Veuillez d'abord traiter vos données dans l'onglet 'Traitement des données' d'abord.")
        return
    default_idx = 0 if st.session_state.get('fusion_data') else 0
    selected_idx = st.radio("Sélectionnez la source de données pour la modélisation :", data_labels, index=default_idx, horizontal=True)
    selected_data = data_sources[data_labels.index(selected_idx)]

    # --- Fin choix source ---

    # Get processed data
    if st.session_state.get('processed_data'):
        processed_data = st.session_state.processed_data['dataframe']
        # Vérifie si processed_data correspond à la source sélectionnée
        # Si ce n'est pas le cas (ex : l'utilisateur a changé de source), on traite à nouveau
        # Pour simplifier, on utilise processed_data si la forme correspond, sinon on repart de la source brute
        if processed_data.shape == selected_data['dataframe'].shape:
            pass
        else:
            processed_data = selected_data['dataframe']
    else:
        processed_data = selected_data['dataframe']

    # Model configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sélection du Modèle")
        available_models = list(modeling_config.get('training', {}).get('models', {}).keys())
        selected_models = st.multiselect(
            "Sélectionnez les modèles à entraîner",
            available_models,
            default=available_models[:3] if len(available_models) >= 3 else available_models
        )
        
        target_column = st.selectbox(
            "Sélectionnez la colonne cible",
            processed_data.columns.tolist(),
            index=processed_data.columns.tolist().index(modeling_config.get('features', {}).get('target_column', processed_data.columns[0])),
            key='target_column'

        )
    
    with col2:
        st.subheader("Paramètres d'Entraînement")
        test_size = st.slider("Taille du jeu de test", 0.1, 0.5, 0.2, 0.05)
        cv_folds = st.slider("Nombre de plis pour la validation croisée", 3, 10, 5)
        random_state = st.number_input("État aléatoire", value=42, min_value=0)
    
    # Feature information
    features = [col for col in processed_data.columns if col != target_column]
    st.info(f"🎯 Cible: **{target_column}** | 📊 Caractéristiques: **{len(features)}** colonnes")
    
    # Train models button
    if st.button('Entraîner les Modèles', type="primary") and selected_models:
        try:
            with st.spinner("Entraînement des modèles... Cela peut prendre plusieurs minutes."):
                # Initialize model manager
                model_manager = ModelManager(modeling_config)
                
                # Create progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Prepare data and features
                status_text.text("Préparation des caractéristiques et encodage des données catégorielles...")
                progress_bar.progress(0.1)
                notify_step_completion('data_preprocessing', "Préparation des caractéristiques et encodage des colonnes catégorielles")
                
                data_for_training = processed_data.drop(columns=['geometry'], errors='ignore')
                X, y = model_manager.prepare_features(data_for_training, target_column)
                # Show categorical encoding information
                if hasattr(model_manager, 'encoders') and model_manager.encoders:
                    st.success("Colonnes catégorielles encodées avec succès !")
                    with st.expander("Voir les Détails de l'Encodage"):
                        for col, encoder in model_manager.encoders.items():
                            st.write(f"**{col}**: Encodage par étiquettes avec {len(encoder.classes_)} catégories")
                            if len(encoder.classes_) <= 10:
                                st.write(f"Catégories: {', '.join(encoder.classes_)}")
                            else:
                                st.write(f"Catégories: {', '.join(encoder.classes_[:10])}... (et {len(encoder.classes_)-10} autres)")
                
                # Show data types after encoding
                with st.expander("Types de Données des Caractéristiques Après Encodage"):
                    dtype_counts = X.dtypes.value_counts()
                    for dtype, count in dtype_counts.items():
                        st.write(f"**{dtype}**: {count} colonnes")
                
                st.info(f"Caractéristiques préparées : {X.shape[1]} caractéristiques, {X.shape[0]} échantillons")
                
                # Train models
                results = {}
                total_models = len(selected_models)
                
                for i, model_name in enumerate(selected_models):
                    status_text.text(f"Entraînement de {model_name}...")
                    progress_bar.progress((i + 1) / total_models)
                    notify_step_completion('model_training', f"Entraînement du modèle {model_name} avec vos données")
                    
                    try:
                        model_result = model_manager.train_model(
                            model_name, X, y, 
                            test_size=test_size, 
                            cv_folds=cv_folds,
                            random_state=random_state
                        )
                        results[model_name] = model_result
                        notify_step_completion('model_evaluation', f"Test de la performance du modèle {model_name}")
                        
                    except Exception as e:
                        st.warning(f"⚠️ Échec de l'entraînement de {model_name} : {str(e)}")
                        continue
                
                status_text.text("Entraînement terminé !")
                
                if results:
                    st.session_state.model_results = results
                    st.session_state.model_trained = True
                    st.success(f"✅ {len(results)} modèles entraînés avec succès !")
                else:
                    st.error("❌ Aucun modèle n'a été entraîné avec succès.")
                
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            st.error(f"❌ Échec de l'entraînement : {str(e)}")
    
    # Display training results
    if st.session_state.model_trained and st.session_state.model_results:
        st.subheader("Comparaison des Performances des Modèles")
        
        results = st.session_state.model_results
        
        # Create performance comparison table
        performance_data = []
        for model_name, result in results.items():
            metrics = result.get('metrics', {})
            performance_data.append({
                'Model': model_name,
                'MAE': round(metrics.get('mae', 0), 4),
                'RMSE': round(metrics.get('rmse', 0), 4),
                'R²': round(metrics.get('r2', 0), 4),
                'Training Time': f"{result.get('training_time', 0):.2f}s"
            })
        
        performance_df = pd.DataFrame(performance_data)
    st.dataframe(performance_df.astype(str), use_container_width=True)
        
        # Performance visualization
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Erreur Absolue Moyenne', 'Erreur Quadratique Moyenne', 'Score R²'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        models = performance_df['Model'].tolist()
        mae_values = performance_df['MAE'].tolist()
        rmse_values = performance_df['RMSE'].tolist()
        r2_values = performance_df['R²'].tolist();
        
        fig.add_trace(go.Bar(x=models, y=mae_values, name='MAE', showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(x=models, y=rmse_values, name='RMSE', showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(x=models, y=r2_values, name='R²', showlegend=False), row=1, col=3)
        
        fig.update_layout(height=400, title_text="Métriques de Performance des Modèles")
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model selection
        best_model = min(results.keys(), key=lambda x: results[x]['metrics'].get('rmse', float('inf')))
        st.success(f"🏆 Meilleur modèle : **{best_model}** (RMSE le plus bas)")

def results_metrics_section():
    st.markdown('<span style="font-size:20px;"><i class="fa fa-chart-bar"></i> <b>Étape 4 : Résultats & Métriques Détaillées</b></span>', unsafe_allow_html=True)
    st.markdown("""
    <span style='font-size:16px'>Analysez les performances de vos modèles en détail.<br>
    <b>Astuce :</b> Utilisez les graphiques pour comparer les résultats.</span>
    """, unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Veuillez entraîner les modèles dans l'onglet 'Entraînement du modèle' d'abord.")
        return
    
    results = st.session_state.model_results
    
    # Model selection for detailed analysis
    selected_model = st.selectbox(
        "Sélectionnez le modèle pour une analyse détaillée",
        list(results.keys()),
        key='selected_model'
    )
    
    if selected_model and selected_model in results:
        model_result = results[selected_model]
        
        # Metrics overview
        st.subheader(f"{selected_model} - Métriques Détaillées")
        
        metrics = model_result.get('metrics', {})
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Erreur Absolue Moyenne", f"{metrics.get('mae', 0):.4f}")
        with col2:
            st.metric("Erreur Quadratique Moyenne", f"{metrics.get('rmse', 0):.4f}")
        with col3:
            st.metric("Score R²", f"{metrics.get('r2', 0):.4f}")
        with col4:
            st.metric("Temps d'Entraînement", f"{model_result.get('training_time', 0):.2f}s")
        
        # Predictions vs Actual
        if 'predictions' in model_result and 'y_test' in model_result:
            st.subheader("Prédictions vs Valeurs Réelles")
            
            y_test = model_result['y_test']
            y_pred = model_result['predictions']
            
            # Scatter plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test, y=y_pred,
                mode='markers',
                name='Prédictions',
                opacity=0.6
            ))
            
            # Perfect prediction line
            min_val = min(min(y_test), min(y_pred))
            max_val = max(max(y_test), max(y_pred))
            fig.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines',
                name='Prédiction Parfaite',
                line=dict(dash='dash', color='red')
            ))
            
            fig.update_layout(
                xaxis_title="Valeurs Réelles",
                yaxis_title="Valeurs Prédites",
                title=f"{selected_model} - Prédictions vs Valeurs Réelles"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Residuals plot
            residuals = np.array(y_test) - np.array(y_pred)
            
            fig_residuals = go.Figure()
            fig_residuals.add_trace(go.Scatter(
                x=y_pred, y=residuals,
                mode='markers',
                name='Résidus',
                opacity=0.6
            ))
            fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
            fig_residuals.update_layout(
                xaxis_title="Valeurs Prédites",
                yaxis_title="Résidus",
                title=f"{selected_model} - Analyse des Résidus"
            )
            
            st.plotly_chart(fig_residuals, use_container_width=True)
        
        # Feature importance (if available)
        if 'feature_importance' in model_result:
            st.subheader("Importance des Caractéristiques")
            
            importance = model_result['feature_importance']
            
            fig_importance = go.Figure(go.Bar(
                x=list(importance.values()),
                y=list(importance.keys()),
                orientation='h'
            ))
            fig_importance.update_layout(
                xaxis_title="Importance",
                yaxis_title="Caractéristiques",
                title=f"{selected_model} - Importance des Caractéristiques"
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Cross-validation results
        if 'cv_scores' in model_result:
            st.subheader("Résultats de la Validation Croisée")
            
            cv_scores = model_result['cv_scores']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Score Moyen CV", f"{np.mean(cv_scores):.4f}")
            with col2:
                st.metric("Écart Type CV", f"{np.std(cv_scores):.4f}")
            
            # CV scores distribution
            fig_cv = go.Figure(data=go.Histogram(x=cv_scores, nbinsx=10))
            fig_cv.update_layout(
                xaxis_title="Scores de Validation Croisée",
                yaxis_title="Fréquence",
                title=f"{selected_model} - Distribution des Scores de Validation Croisée"
            )
            
            st.plotly_chart(fig_cv, use_container_width=True)

def geospatial_analysis_section(modeling_config):
    st.markdown('<span style="font-size:20px;"><i class="fa fa-map"></i> <b>Étape 5 : Analyse & Visualisation Géospatiale</b></span>', unsafe_allow_html=True)
    st.markdown("""
    <span style='font-size:16px'>Visualisez vos données et prédictions sur une carte interactive.<br>
    <b>Astuce :</b> Filtrez et personnalisez l'affichage pour mieux comprendre vos données.</span>
    """, unsafe_allow_html=True)
    
    # --- Choix de la source de données ---
    data_sources = []
    data_labels = []
    if st.session_state.get('fusion_data'):
        data_sources.append(st.session_state.fusion_data)
        data_labels.append("Données fusionnées")
    if st.session_state.get('uploaded_data'):
        data_sources.append(st.session_state.uploaded_data)
        data_labels.append("Données importées")
    if not data_sources:
        st.warning("⚠️ Veuillez d'abord télécharger ou fusionner vos données.")
        return
    default_idx = 0 if st.session_state.get('fusion_data') else 0
    selected_idx = st.radio("Sélectionnez la source de données à visualiser :", data_labels, index=default_idx, horizontal=True)
    selected_data = data_sources[data_labels.index(selected_idx)]

    # --- Fin choix source ---

    # Get data and coordinate information
    df = selected_data['dataframe']
    coord_info = selected_data['coordinates']
    
    # Ajout : extraction latitude/longitude si geometry existe
    if 'geometry' in df.columns:
        from shapely.geometry import Point, LineString
        # Vérifie que tous les objets sont bien des Points ou LineString
        def linestring_to_middle_point(geom):
            if geom is None or (hasattr(geom, 'is_empty') and geom.is_empty):
                return None
            if isinstance(geom, LineString):
                coords = list(geom.coords)
                if len(coords) == 0:
                    return None
                mid_idx = len(coords) // 2
                return Point(coords[mid_idx])
            elif isinstance(geom, Point):
                return geom
            else:
                return None
        df['geometry'] = df['geometry'].apply(linestring_to_middle_point)
        # Vérifie que tous les objets sont bien des Points
        if df['geometry'].apply(lambda g: isinstance(g, Point)).all():
            df['latitude'] = df['geometry'].apply(lambda g: g.y)
            df['longitude'] = df['geometry'].apply(lambda g: g.x)
            # Met à jour coord_info pour utiliser les nouvelles colonnes
            coord_info['lat'] = 'latitude'
            coord_info['lon'] = 'longitude'

    if not coord_info['lat'] or not coord_info['lon']:
        st.error("❌ Aucune colonne de coordonnées détectée dans vos données. Veuillez vous assurer que vos données contiennent des colonnes de latitude et de longitude.")
        return
    
    st.success(f"✅ Coordonnées détectées : {coord_info['lat']}, {coord_info['lon']}")
    
    # Use processed data if available
    if st.session_state.data_processed:
        processed_df = st.session_state.processed_data['dataframe']
        # Check if coordinate columns still exist after processing
        if coord_info['lat'] in processed_df.columns and coord_info['lon'] in processed_df.columns:
            df = processed_df
            st.info("Utilisation des données traitées pour la visualisation")
        else:
            st.warning("Colonnes de coordonnées non trouvées dans les données traitées. Utilisation des données originales.")
    
    # Visualization options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Options de Visualisation")
        
        # Color coding options
        color_column = st.selectbox(
            "Colorer les points par",
            [None] + [col for col in df.columns if col not in [coord_info['lat'], coord_info['lon']]],
            help="Sélectionnez une colonne pour colorer les points de la carte",
            key='color_column'
        )
        
        # Size coding options
        size_column = st.selectbox(
            "Taille des points par",
            [None] + [col for col in df.select_dtypes(include=[np.number]).columns if col not in [coord_info['lat'], coord_info['lon']]],
            help="Sélectionnez une colonne numérique pour ajuster la taille des points de la carte",
            key='size_column'
        )
        
        # Map style
        map_style = st.selectbox(
            "Style de la carte",
            ["OpenStreetMap", "CartoDB positron", "CartoDB dark_matter", "Stamen Terrain"],
            help="Choisissez le style de base de la carte",
            key='map_style'
        )
    
    with col2:
        st.subheader("Filtrage des Données")
        
        # Liste déroulante des adresses avec normalisation d'encodage
        streetname_search = None
        streetname_options = []
        if 'StreetName' in df.columns:
            # Normalise les caractères pour l'affichage
            streetname_options = sorted(
                df['StreetName']
                .dropna()
                .astype(str)
                .map(lambda s: s.encode('utf-8', errors='replace').decode('utf-8', errors='replace').strip())
                .unique()
            )
            streetname_options_display = ["(Toutes les adresses)"] + streetname_options
            streetname_search = st.selectbox(
                "Filtrer par adresse (StreetName)",
                streetname_options_display,
                index=0,
                help="Sélectionnez une adresse pour filtrer les points correspondants"
            )
        
        # Sample data for performance
        sample_size = st.slider(
            "Taille de l'échantillon (pour la performance)",
            min_value=100,
            max_value=min(10000, len(df)),
            value=min(1000, len(df)),
            help="Réduisez la taille de l'échantillon pour une meilleure performance avec de grands ensembles de données"
        )
        
        # Filter by numeric column if available
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns:
            filter_column = st.selectbox("Filtrer par colonne", [None] + numeric_columns,key='filter_column')
            if filter_column:
                min_val, max_val = float(df[filter_column].min()), float(df[filter_column].max())
                filter_range = st.slider(
                    f"Filtrer la plage de {filter_column}",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val)
                )
    
    # Generate map button
    if st.button('Générer la Carte', type="primary"):
        try:
            with st.spinner("Génération de la visualisation géospatiale..."):
                filtered_df = df.copy()
                
                # Filtrage par adresse (StreetName) via liste déroulante
                if (
                    streetname_search
                    and streetname_search != "(Toutes les adresses)"
                    and 'StreetName' in filtered_df.columns
                ):
                    filtered_df = filtered_df[
                        filtered_df['StreetName'].astype(str) == streetname_search
                    ]
                
                # Apply numeric filter if selected
                if 'filter_column' in locals() and filter_column and 'filter_range' in locals():
                    filtered_df = filtered_df[
                        (filtered_df[filter_column] >= filter_range[0]) & 
                        (filtered_df[filter_column] <= filter_range[1])
                    ]
                
                # Sample data if needed
                if len(filtered_df) > sample_size:
                    filtered_df = filtered_df.sample(n=sample_size, random_state=42)
                
                # Remove rows with missing coordinates
                filtered_df = filtered_df.dropna(subset=[coord_info['lat'], coord_info['lon']])
                
                if len(filtered_df) == 0:
                    st.error("❌ Aucune donnée de coordonnées valide trouvée après filtrage.")
                    return
                
                # Initialize geospatial visualizer
                visualizer = GeospatialVisualizer()
                
                # Create map
                map_obj = visualizer.create_traffic_map(
                    df=filtered_df,
                    geometry_col='geometry',
                    color_col=color_column,
                    size_col=size_column,
                    tile_style=map_style,
                    lat_col='latitude',      # <-- Ajoute cette ligne
                    lon_col='longitude'
                )
                
                # Display map
                st.markdown('<i class="fa fa-map"></i> Carte Interactive du Trafic', unsafe_allow_html=True)
                st.components.v1.html(map_obj._repr_html_(), height=600)
                
                # Map statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Points sur la Carte", len(filtered_df))
                with col2:
                    if coord_info['lat'] in filtered_df.columns:
                        lat_range = filtered_df[coord_info['lat']].max() - filtered_df[coord_info['lat']].min()
                        st.metric("Plage de Latitude", f"{lat_range:.4f}°")
                with col3:
                    if coord_info['lon'] in filtered_df.columns:
                        lon_range = filtered_df[coord_info['lon']].max() - filtered_df[coord_info['lon']].min()
                        st.metric("Plage de Longitude", f"{lon_range:.4f}°")
                with col4:
                    center_lat = filtered_df[coord_info['lat']].mean()
                    center_lon = filtered_df[coord_info['lon']].mean()
                    st.metric("Centre de la Carte", f"{center_lat:.3f}, {center_lon:.3f}")
                
        except Exception as e:
            logger.error(f"Error generating map: {str(e)}")
            st.error(f"❌ Échec de la génération de la carte : {str(e)}")
    
    # Additional geospatial analysis
    if st.session_state.model_trained and st.session_state.model_results:
        st.subheader("Cartographie des Prédictions")
        
        # Liste déroulante pour filtrer les prédictions par adresse avec normalisation
        streetname_search_pred = None
        streetname_options_pred = []
        if 'StreetName' in df.columns:
            streetname_options_pred = sorted(
                df['StreetName']
                .dropna()
                .astype(str)
                .map(lambda s: s.encode('utf-8', errors='replace').decode('utf-8', errors='replace').strip())
                .unique()
            )
            streetname_options_pred_display = ["(Toutes les adresses)"] + streetname_options_pred
            streetname_search_pred = st.selectbox(
                "Filtrer par adresse (StreetName) pour les prédictions",
                streetname_options_pred_display,
                index=0,
                help="Sélectionnez une adresse pour filtrer les points prédits correspondants"
            )
        
        if st.button('Cartographier les Prédictions du Modèle'):
            try:
                with st.spinner("Génération de la carte des prédictions..."):
                    # Get best model predictions
                    results = st.session_state.model_results
                    best_model = min(results.keys(), key=lambda x: results[x]['metrics'].get('rmse', float('inf')))
                    model_result = results[best_model]
                    if 'X_test' in model_result and 'predictions' in model_result:
                        X_test = model_result['X_test']
                        predictions = model_result['predictions']

                        logger.info(f"Colonnes de X_test: {X_test.columns.tolist()}")
                        logger.info(f"Index de X_test: {X_test.index}")

                        # --- Correction : Réintégration de la colonne geometry dans X_test ---
                        if 'geometry' not in X_test.columns:
                            df_source = selected_data['dataframe']
                            # logger.info("geometry non présent, tentative de récupération depuis le df source")
                            if 'geometry' in df_source.columns:
                                if all(idx in df_source.index for idx in X_test.index):
                                    X_test = X_test.copy()
                                    X_test['geometry'] = df_source.loc[X_test.index, 'geometry'].values
                                    # logger.info("geometry récupéré par index")
                                elif 'Id' in X_test.columns and 'Id' in df_source.columns:
                                    X_test = X_test.merge(df_source[['Id', 'geometry']], on='Id', how='left')
                                    # logger.info("geometry récupéré par merge sur Id")
                        # --- Conversion LineString -> Point ---
                        if 'geometry' in X_test.columns:
                            from shapely.geometry import Point, LineString
                            def linestring_to_middle_point(geom):
                                if geom is None or (hasattr(geom, 'is_empty') and geom.is_empty):
                                    return None
                                if isinstance(geom, LineString):
                                    coords = list(geom.coords)
                                    if len(coords) == 0:
                                        return None
                                    mid_idx = len(coords) // 2
                                    return Point(coords[mid_idx])
                                elif isinstance(geom, Point):
                                    return geom
                                else:
                                    return None
                            X_test['geometry'] = X_test['geometry'].apply(linestring_to_middle_point)
                            # (Supprimer les st.write de debug ici)
                            # logger.info(f"Types de geometry: {X_test['geometry'].apply(lambda g: type(g)).value_counts()}")
                            mask_valid_geom = X_test['geometry'].apply(lambda g: isinstance(g, Point) and g is not None and not pd.isnull(g.x) and not pd.isnull(g.y))
                            # logger.info(f"Nombre de géométries valides: {mask_valid_geom.sum()}")
                            pred_df = X_test[mask_valid_geom].copy()
                            # Filtrage par adresse (StreetName) via liste déroulante
                            if (
                                streetname_search_pred
                                and streetname_search_pred != "(Toutes les adresses)"
                                and 'StreetName' in pred_df.columns
                            ):
                                pred_df = pred_df[
                                    pred_df['StreetName'].astype(str) == streetname_search_pred
                                ]
                            if pred_df.empty:
                                st.error("❌ Aucune donnée de coordonnées valide trouvée pour la cartographie des prédictions.")
                                logger.error("Aucune donnée de coordonnées valide trouvée pour la cartographie des prédictions.")
                                return
                            pred_df['latitude'] = pred_df['geometry'].apply(lambda g: g.y)
                            pred_df['longitude'] = pred_df['geometry'].apply(lambda g: g.x)
                            coord_info['lat'] = 'latitude'
                            coord_info['lon'] = 'longitude'
                            pred_df['predictions'] = np.array(predictions)[mask_valid_geom.values]
                            pred_df['actual'] = np.array(model_result.get('y_test', [None] * len(predictions)))[mask_valid_geom.values]
                            notify_step_completion('visualization', "Création de cartes interactives des prédictions")
                            visualizer = GeospatialVisualizer()
                            pred_map = visualizer.create_prediction_map(
                                df=pred_df,
                                prediction_col='predictions',
                                actual_col='actual' if 'y_test' in model_result else None
                            )
                            st.markdown(f'<i class="fa fa-bullseye"></i> Carte des Prédictions - {best_model}', unsafe_allow_html=True)
                            st.components.v1.html(pred_map._repr_html_(), height=600)
                        else:
                            st.warning("⚠️ Les colonnes de coordonnées ne sont pas disponibles dans les données de test pour la cartographie des prédictions.")
                            logger.warning("geometry non présent dans X_test après tentative de récupération")
            except Exception as e:
                logger.error(f"Error generating prediction map: {str(e)}")
                st.error(f"❌ Échec de la cartographie des prédictions : {str(e)}")

    st.markdown("---")
    st.subheader("Cartographie des Prédictions sur Données Incomplètes")

    # Prépare le DataFrame de base
    df_display = None
    color_column_incomplete = None
    size_column_incomplete = None
    map_style_incomplete = None
    streetname_search_incomplete = None
    sample_size_incomplete = None
    filter_column_incomplete = None
    filter_range_incomplete = None

    # Vérifie la disponibilité des données
    if 'original_data' in st.session_state and st.session_state['original_data'] is not None:
        df_orig = st.session_state['original_data'].copy()
        if 'debit' in df_orig.columns:
            df_missing = df_orig[df_orig['debit'].isna()].copy()
            df_known = df_orig[df_orig['debit'].notna()].copy()
            results = st.session_state.model_results
            if results :
                best_model = min(results.keys(), key=lambda x: results[x]['metrics'].get('rmse', float('inf')))
                model_manager = ModelManager(modeling_config)
                if 'model' in results[best_model]:
                    model_manager.trained_models[best_model] = results[best_model]
                if 'encoders' in results[best_model]:
                    model_manager.encoders = results[best_model]['encoders']
                df_missing = df_missing.reset_index(drop=True)
                cols_to_remove = ['geometry']
                features = [col for col in df_missing.columns if col not in cols_to_remove + ['debit']]
                X_missing = model_manager.prepare_features(df_missing[features + ['debit']].fillna(0), 'debit')[0]
                X_missing = X_missing.reset_index(drop=True)
                trained_model = results[best_model]['model']
                if hasattr(trained_model, 'feature_names_in_'):
                    expected_features = list(trained_model.feature_names_in_)
                else:
                    expected_features = results[best_model].get('X_test', df_missing).columns.tolist()
                expected_features = [col for col in expected_features if col in X_missing.columns]
                X_missing = X_missing.loc[:, expected_features]
                X_missing = X_missing.fillna(0).reset_index(drop=True)
                y_pred_missing = model_manager.predict(best_model, X_missing)
                df_missing['debit_pred'] = pd.Series(np.array(y_pred_missing).reshape(-1), index=df_missing.index)
                df_known['debit_pred'] = np.nan
                df_known['type'] = 'réel'
                df_missing['type'] = 'prédit'
                df_display = pd.concat([df_known, df_missing], ignore_index=True)
                color_map = {'réel': 'green', 'prédit': 'orange'}
                df_display['color'] = df_display['type'].map(color_map)
                # --- Bouton de téléchargement du CSV avec prédictions ---
                import io

                csv_buffer = io.StringIO()
                df_display.to_csv(csv_buffer, index=False)

                st.download_button(
                    label="📥 Télécharger les prédictions (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="predictions_incompletes.csv",
                    mime="text/csv"
                )

                # Conversion geometry -> latitude/longitude
                from shapely.geometry import Point, LineString
                def linestring_to_middle_point(geom):
                    if geom is None or (hasattr(geom, 'is_empty') and geom.is_empty):
                        return None
                    if isinstance(geom, LineString):
                        coords = list(geom.coords)
                        if len(coords) == 0:
                            return None
                        mid_idx = len(coords) // 2
                        return Point(coords[mid_idx])
                    elif isinstance(geom, Point):
                        return geom
                    else:
                        return None

                if 'geometry' in df_display.columns:
                    df_display['geometry'] = df_display['geometry'].apply(linestring_to_middle_point)
                    mask_valid_geom = df_display['geometry'].apply(lambda g: isinstance(g, Point) and g is not None and not pd.isnull(g.x) and not pd.isnull(g.y))
                    df_display = df_display[mask_valid_geom].copy()
                    df_display['latitude'] = df_display['geometry'].apply(lambda g: g.y)
                    df_display['longitude'] = df_display['geometry'].apply(lambda g: g.x)

                # --- Options de visualisation et filtrage (toujours affichées) ---
                col1, col2 = st.columns(2)
                with col1:
                    color_column_incomplete = st.selectbox(
                        "Colorer les points par",
                        [None] + [col for col in df_display.columns if col not in ['latitude', 'longitude']],
                        index=0,
                        key='color_column_incomplete'
                    )
                    size_column_incomplete = st.selectbox(
                        "Taille des points par",
                        [None] + [col for col in df_display.select_dtypes(include=[np.number]).columns if col not in ['latitude', 'longitude']],
                        index=0,
                        key='size_column_incomplete'
                    )
                    map_style_incomplete = st.selectbox(
                        "Style de la carte",
                        ["OpenStreetMap", "CartoDB positron", "CartoDB dark_matter", "Stamen Terrain"],
                        index=0,
                        key='map_style_incomplete'
                    )
                with col2:
                    # Filtrer par adresse
                    streetname_options_incomplete = []
                    if 'StreetName' in df_display.columns:
                        streetname_options_incomplete = sorted(
                            df_display['StreetName']
                            .dropna()
                            .astype(str)
                            .map(lambda s: s.encode('utf-8', errors='replace').decode('utf-8', errors='replace').strip())
                            .unique()
                        )
                        streetname_options_display_incomplete = ["(Toutes les adresses)"] + streetname_options_incomplete
                        streetname_search_incomplete = st.selectbox(
                            "Filtrer par adresse (StreetName)",
                            streetname_options_display_incomplete,
                            index=0,
                            key='streetname_search_incomplete'
                        )
                    else:
                        streetname_search_incomplete = None
                    # Filtrer par range de débit
                    st.write("Filtrer par plage de débit (réel ou prédit)")
                    min_debit = float(df_display[['debit', 'debit_pred']].min().min())
                    max_debit = float(df_display[['debit', 'debit_pred']].max().max())
                    debit_min = st.number_input("Débit minimum", value=min_debit, min_value=min_debit, max_value=max_debit, key='debit_min_incomplete')
                    debit_max = st.number_input("Débit maximum", value=max_debit, min_value=min_debit, max_value=max_debit, key='debit_max_incomplete')

                # --- Génération de la carte au clic ---
                if st.button("Cartographier les Prédictions sur Données Incomplètes"):
                    # Applique les filtres
                    df_display_filtered = df_display.copy()
                    # Filtre par adresse
                    if (
                        streetname_search_incomplete
                        and streetname_search_incomplete != "(Toutes les adresses)"
                        and 'StreetName' in df_display_filtered.columns
                    ):
                        df_display_filtered = df_display_filtered[
                            df_display_filtered['StreetName'].astype(str) == streetname_search_incomplete
                        ].copy()
                    # Filtre par plage de débit (réel ou prédit)
                    df_display_filtered = df_display_filtered[
                        (
                            (df_display_filtered['type'] == 'réel') &
                            (df_display_filtered['debit'] >= debit_min) &
                            (df_display_filtered['debit'] <= debit_max)
                        ) |
                        (
                            (df_display_filtered['type'] == 'prédit') &
                            (df_display_filtered['debit_pred'] >= debit_min) &
                            (df_display_filtered['debit_pred'] <= debit_max)
                        )
                    ].copy()
                    MAX_POINTS = 5000
                    if len(df_display_filtered) > MAX_POINTS:
                        df_display_filtered = df_display_filtered.sample(n=MAX_POINTS, random_state=42).copy()
                    if (
                        df_display_filtered.empty or
                        'latitude' not in df_display_filtered.columns or
                        'longitude' not in df_display_filtered.columns or
                        df_display_filtered['latitude'].isnull().all() or
                        df_display_filtered['longitude'].isnull().all()
                    ):
                        st.error("❌ Aucune donnée géospatiale valide à afficher sur la carte après filtrage.")
                        return

                    # --- Appel de la carte ---
                    visualizer = GeospatialVisualizer()
                    def custom_popup(row):
                        if row['color'] == 'green':
                            zone = row['StreetName'] if 'StreetName' in row else ''
                            capteur = row['Id'] if 'Id' in row else ''
                            debit = row['debit'] if 'debit' in row else ''
                            popup_content = f"""
                            <div style="font-family: Arial, sans-serif; width: 200px;">
                                <h4 style="margin: 0 0 10px 0; color: #333;">Mesure réelle</h4>
                                <table style="width: 100%; font-size: 12px;">
                                    <tr><td><b>Zone :</b></td><td>{zone}</td></tr>
                                    <tr><td><b>Capteur :</b></td><td>{capteur}</td></tr>
                                    <tr><td><b>Débit :</b></td><td>{debit}</td></tr>
                                </table>
                            </div>
                            """
                        else:
                            debit_pred = row['debit_pred'] if 'debit_pred' in row else ''
                            popup_content = f"""
                            <div style="font-family: Arial, sans-serif; width: 200px;">
                                <h4 style="margin: 0 0 10px 0; color: #333;">Prédiction du modèle</h4>
                                <table style="width: 100%; font-size: 12px;">
                                    <tr><td><b>Débit estimé :</b></td><td>{debit_pred}</td></tr>
                                </table>
                            </div>
                            """
                        return popup_content

                    m = visualizer.create_traffic_map(
                        df=df_display_filtered,
                        geometry_col='geometry',
                        color_col='color',
                        size_col=None,
                        tile_style="OpenStreetMap",
                        lat_col='latitude',
                        lon_col='longitude',
                        popup_func=custom_popup  
                    )
                    st.markdown("🟢 Points verts : mesures réelles<br>🟠 Points orange : prédictions du modèle là où il n'y avait pas de valeur mesurée.", unsafe_allow_html=True)
                    st.components.v1.html(m._repr_html_(), height=600)
        else:
            st.warning("La colonne 'debit' n'est pas présente dans le dataset original.")
    else:
        st.warning("Le dataset original n'est pas disponible pour cette fonctionnalité.")    

def detect_encoding(file_obj):
    import chardet
    file_obj.seek(0)
    rawdata = file_obj.read(10000)
    file_obj.seek(0)
    result = chardet.detect(rawdata)
    return result['encoding'] if result['encoding'] else 'utf-8'

def safe_read_csv(file_obj, sep=','):
    import pandas as pd
    encodings_to_try = []
    detected = detect_encoding(file_obj)
    if detected:
        encodings_to_try.append(detected)
    encodings_to_try += ['utf-8', 'latin1', 'cp1252']
    file_obj.seek(0)
    for enc in encodings_to_try:
        try:
            file_obj.seek(0)
            return pd.read_csv(file_obj, sep=sep, encoding=enc)
        except Exception:
            continue
    raise UnicodeDecodeError("Aucun encodage compatible trouvé pour ce fichier.")


def safe_read_any_table(file_obj, sep=','):
    import pandas as pd
    from dbfread import DBF
    import os
    import tempfile

    filename = file_obj.name.lower()

    # Contrôle taille fichier
    file_obj.seek(0, os.SEEK_END)
    file_size_MB = file_obj.tell() / (1024 * 1024)
    file_obj.seek(0)
    if file_size_MB > 200:
        raise ValueError(f"Fichier trop volumineux ({file_size_MB:.2f} MB) : {file_obj.name}")

    # Cas DBF
    if filename.endswith('.dbf'):
        try:
            file_obj.seek(0)
            with tempfile.NamedTemporaryFile(suffix=".dbf", delete=False) as tmp:
                tmp.write(file_obj.read())
                tmp.flush()
                file_path = tmp.name
            return pd.DataFrame(DBF(file_path))
        except Exception as e:
            raise ValueError(f"Erreur lors de la lecture du fichier DBF '{file_obj.name}' : {e}")

    # Cas CSV
    encodings_to_try = ['utf-8', 'latin1', 'cp1252']
    for enc in encodings_to_try:
        try:
            file_obj.seek(0)
            return pd.read_csv(file_obj, sep=sep, encoding=enc)
        except Exception:
            continue
    raise UnicodeDecodeError("Aucun encodage compatible trouvé pour ce fichier.")



def fusion_donnees_debits_fcd(
    file_debits, file_fcd_corresp, fcd_files_uploaded,
    annee_debut, annee_fin, mois_debut, mois_fin,
    file_shp=None, file_dbf=None, file_shx=None
):
    import pandas as pd
    from pathlib import Path
    import tempfile
    import logging
    import geopandas as gpd
    import os
    logger = logging.getLogger("fusion_donnees_debits_fcd")

    # 1. Lecture des débits
    try:
        df_debits_brut = safe_read_any_table(file_debits, sep=';')
        logger.info("Importation du fichier de débits réussie.")
        df_debits = calculer_moyenne_horaire_periode(df_debits_brut, annee_debut, annee_fin, mois_debut, mois_fin)
    except Exception as e:
        logger.error(f"Erreur lors de l'importation ou du traitement des débits : {e}")
        return None

    # 2. Lecture du fichier de correspondance
    try:
        df_fcd_corresp = safe_read_any_table(file_fcd_corresp)
        logger.info("Importation du fichier de correspondance FCD/Capteurs réussie.")
    except Exception as e:
        logger.error(f"Erreur lors de l'importation du fichier de correspondance : {e}")
        return None

    # 3. Lecture du fichier SHP
    gdf_shp = None
    if file_shp and file_dbf and file_shx:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Utilise juste le nom si c'est un fichier par défaut (open renvoie _io.BufferedReader)
                def get_safe_name(f):
                    return os.path.basename(f.name) if isinstance(f, (io.BufferedReader,)) else f.name

                shp_path = Path(tmpdir) / get_safe_name(file_shp)
                dbf_path = Path(tmpdir) / get_safe_name(file_dbf)
                shx_path = Path(tmpdir) / get_safe_name(file_shx)
                with open(shp_path, "wb") as out: out.write(file_shp.read())
                with open(dbf_path, "wb") as out: out.write(file_dbf.read())
                with open(shx_path, "wb") as out: out.write(file_shx.read())
                # Ajout : lire aussi StreetName si elle existe
                gdf_tmp = gpd.read_file(str(shp_path))
                cols = ['Id', 'geometry']
                if 'StreetName' in gdf_tmp.columns:
                    cols.append('StreetName')
                gdf_shp = gdf_tmp[cols]
                print(gdf_shp.head())
            logger.info("Importation du shapefile réussie.")
        except Exception as e:
            logger.warning(f"Erreur lecture shapefile : {e}")

    # 4. Lecture des fichiers FCD
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_fcd_paths = []
        dfs_fcd = []

        def get_safe_name(f):
            return os.path.basename(f.name) if isinstance(f, (io.BufferedReader,)) else f.name

        for fcd_file in fcd_files_uploaded:
            try:
                temp_path = Path(tmpdir) / get_safe_name(fcd_file)
                with open(temp_path, "wb") as out:
                    out.write(fcd_file.read())

                # ✅ Ouvrir le fichier pour lecture correcte
                with open(temp_path, "rb") as f:
                    df_fcd = safe_read_any_table(f)
                # Correction : détecte toutes les variantes de colonne Id
                id_cols = [col for col in df_fcd.columns if col.lower().endswith('_id')]
                if id_cols:
                    # Renomme toutes les colonnes trouvées en 'Id'
                    for col in id_cols:
                        df_fcd.rename(columns={col: 'Id'}, inplace=True)
                # Vérifie que la colonne existe et convertis le type
                if 'Id' in df_fcd.columns:
                    print(" ID se trouve dans fcd")
                    df_fcd['Id'] = df_fcd['Id'].astype(int)
                if gdf_shp is not None and 'Id' in gdf_shp.columns:
                    gdf_shp['Id'] = gdf_shp['Id'].astype(int)
                    # Le merge ajoutera aussi StreetName si elle existe
                    df_fcd = df_fcd.merge(gdf_shp, on='Id', how='left')
                    logger.info(df_fcd.head())
                    logger.info("Fusion a succes.")
                else:
                    logger.error("Colonne 'Id' manquante dans FCD ou shapefile.")
                    continue
                
                df_fcd.to_csv(temp_path, index=False)
                temp_fcd_paths.append(fcd_file.name)
                logger.info(f"Fichier FCD '{fcd_file.name}' importé avec succès.")
                dfs_fcd.append(df_fcd)
            except Exception as e:
                logger.warning(f"Fichier FCD '{fcd_file.name}' ignoré (erreur : {e})")

        df_fcd_final = pd.concat(dfs_fcd, ignore_index=True) if dfs_fcd else None

        # 5. Fusion finale
        if df_fcd_final is not None and not df_debits.empty:
            try:
                # Récupère les heures présentes dans les fichiers FCD
                heures_fcd_uniques = sorted(df_fcd_final['heure'].unique()) if 'heure' in df_fcd_final.columns else []
                # Filtre le DataFrame de débits pour ne garder que les colonnes correspondant aux heures FCD
                debit_cols = [f"debit_{h}" for h in heures_fcd_uniques]
                debit_cols_base = ['zone', 'capteur']
                debit_cols_to_keep = debit_cols_base + [col for col in debit_cols if col in df_debits.columns]
                df_debits_fcd = df_debits[debit_cols_to_keep].copy() if all(col in df_debits.columns for col in debit_cols_to_keep) else df_debits.copy()
                # Fusionne le DataFrame FCD issu des fichiers FCD avec les débits filtrés
                df_fusion = fusionner_debits_fcd(df_fcd_final, df_debits_fcd)
                # Filtre pour ne garder que les heures présentes dans les fichiers FCD (sécurité)
                df_fusion = df_fusion[df_fusion['heure'].isin(heures_fcd_uniques)]
                print(f"Heures FCD utilisées pour la fusion : {heures_fcd_uniques}")
                print(f"Colonnes de débits utilisées : {[col for col in debit_cols if col in df_debits.columns]}")
                print(df_fusion.columns)
                logger.info("Fusion des données réussie.")
                return df_fusion
            except Exception as e:
                logger.error(f"Erreur fusion finale : {e}")
                return None
        else:
            logger.error("Données FCD ou débits manquantes.")
            return None


def fusion_debits_fcd_section():
        # L'affichage de l'aperçu du fichier de débits filtré doit être fait uniquement si les fichiers sont présents et le DataFrame existe
        # ...existing code...
    st.markdown('<span style="font-size:20px;"><i class="fa fa-link"></i> <b>Étape 6 : Fusion Débits & FCD</b></span>', unsafe_allow_html=True)
    st.markdown("""
    <span style='font-size:16px'>Fusionnez vos fichiers pour créer un dataset unique.<br>
    <b>Astuce :</b> Des fichiers d'exemple sont proposés par défaut. Vous pouvez importer vos propres fichiers si besoin.</span>
    """, unsafe_allow_html=True)
    
    mode = st.radio("Choisissez la méthode :", ["Manuelle (fichiers)", "Automatique via API TomTom MOVE"], horizontal=True)
    import io
    import glob
    # Répertoire des fichiers par défaut
    
    default_dir = "data/default_fusion/"
    # default_files = {...}  # Désactivé : plus de fichiers par défaut
    col1, col2 = st.columns(2)
    with col1:
        # Fichier débits
        file_debits = st.file_uploader("Fichier débits (CSV, séparateur ';')", type=['csv'], key="fusion_debits")
        # Fichier correspondance
        file_fcd_corresp = st.file_uploader("Fichier de correspondance FCD/Capteurs (CSV)", type=['csv'], key="fusion_corresp")
        # Fichiers FCD (multi)
        fcd_files_uploaded = st.file_uploader("Fichiers FCD (DBF, multi)", type=['dbf'], accept_multiple_files=True, key="fusion_fcd")
        # Fichier SHP
        file_shp = st.file_uploader("Fichier SHP (Shapefile, .shp)", type=['shp'], key="fusion_shp")
        # Fichier DBF
        file_dbf = st.file_uploader("Fichier DBF associé (Shapefile, .dbf)", type=['dbf'], key="fusion_dbf")
        # Fichier SHX
        file_shx = st.file_uploader("Fichier SHX associé (Shapefile, .shx)", type=['shx'], key="fusion_shx")
    with col2:
        
        annee_debut = st.number_input("Année début", min_value=2000, max_value=2100, value=2022)
        annee_fin = st.number_input("Année fin", min_value=2000, max_value=2100, value=2024)
        mois_debut = st.number_input("Mois début", min_value=1, max_value=12, value=1)
        mois_fin = st.number_input("Mois fin", min_value=1, max_value=12, value=4)
        
        
    ready = file_debits and file_fcd_corresp and fcd_files_uploaded and len(fcd_files_uploaded) > 0
    if file_shp:
        st.success(f"✅ Fichier SHP importé : {getattr(file_shp, 'name', 'par défaut')}")
    if file_dbf:
        st.success(f"✅ Fichier DBF importé : {getattr(file_dbf, 'name', 'par défaut')}")
    if file_shx:
        st.success(f"✅ Fichier SHX importé : {getattr(file_shx, 'name', 'par défaut')}")
    if st.button("Fusionner Débits & FCD", type="primary") and ready:
        with st.spinner("Fusion des données en cours..."):
            # --- Contrôle des heures uniques dans les fichiers FCD avant la fusion ---
            heures_fcd = []
            for fcd_file in fcd_files_uploaded:
                try:
                    from src.fusion_donnees_fcd_debits import extraire_heure_depuis_fichier
                    heure = extraire_heure_depuis_fichier(getattr(fcd_file, 'name', ''))
                    if heure is not None:
                        heures_fcd.append(heure)
                except Exception:
                    pass
            if heures_fcd:
                st.info(f"Heures extraites des fichiers FCD importés : {sorted(set(heures_fcd))}")
            else:
                st.warning("Aucune heure extraite des fichiers FCD importés.")

            # --- Fusion avec aperçu des datasets intermédiaires ---
            # 1. Lecture des débits
            df_debits_brut = safe_read_any_table(file_debits, sep=';')
            st.subheader("Aperçu du fichier de débits brut")
            st.dataframe(df_debits_brut.head(10).astype(str), use_container_width=True)
            # Vérification de la période sélectionnée par rapport aux données disponibles
            annees_disponibles = df_debits_brut['periode'].apply(lambda x: pd.to_datetime(x).year).unique()
            mois_disponibles = df_debits_brut['periode'].apply(lambda x: pd.to_datetime(x).month).unique()
            periode_invalide = False
            if annee_debut < min(annees_disponibles) or annee_fin > max(annees_disponibles):
                st.error(f"❌ La période sélectionnée ({annee_debut}-{annee_fin}) déborde des années disponibles dans le fichier de débits ({min(annees_disponibles)}-{max(annees_disponibles)}). Veuillez entrer une période valide.")
                st.info(f"Période valide pour ce fichier de débits : Années {min(annees_disponibles)} à {max(annees_disponibles)}, Mois {min(mois_disponibles)} à {max(mois_disponibles)}.")
                return
            if mois_debut < min(mois_disponibles) or mois_fin > max(mois_disponibles):
                st.error(f"❌ Les mois sélectionnés ({mois_debut}-{mois_fin}) débordent des mois disponibles dans le fichier de débits ({min(mois_disponibles)}-{max(mois_disponibles)}). Veuillez entrer une période valide.")
                st.info(f"Période valide pour ce fichier de débits : Années {min(annees_disponibles)} à {max(annees_disponibles)}, Mois {min(mois_disponibles)} à {max(mois_disponibles)}.")
                return
            df_debits = calculer_moyenne_horaire_periode(df_debits_brut, annee_debut, annee_fin, mois_debut, mois_fin)
            if df_debits is None or df_debits.empty:
                st.error("❌ La période sélectionnée (année/mois) n'existe pas dans le fichier de débits. Veuillez choisir une période valide.")
                return
            st.subheader("Aperçu du fichier de débits après calcul de la moyenne horaire")
            st.dataframe(df_debits.head(10).astype(str), use_container_width=True)

            # 2. Lecture du fichier de correspondance
            df_fcd_corresp = safe_read_any_table(file_fcd_corresp)
            # Ne garder que les colonnes nécessaires à la fusion
            cols_to_keep = ['Id', 'capteurs_segments — Feuil1_capteur', 'capteurs_segments — Feuil1_zone']
            df_fcd_corresp = df_fcd_corresp[[col for col in cols_to_keep if col in df_fcd_corresp.columns]].copy()
            # Renommer les colonnes pour la fusion
            df_fcd_corresp = df_fcd_corresp.rename(columns={
                'capteurs_segments — Feuil1_capteur': 'capteur',
                'capteurs_segments — Feuil1_zone': 'zone'
            })
            # Évite le produit cartésien lors du merge
            df_fcd_corresp = df_fcd_corresp.drop_duplicates(subset='Id')
            st.subheader("Aperçu du fichier de correspondance FCD/Capteurs (colonnes filtrées et renommées)")
            st.dataframe(df_fcd_corresp.head(10).astype(str), use_container_width=True)

            # 3. Lecture du fichier SHP
            gdf_shp = None
            import geopandas as gpd
            if file_shp and file_dbf and file_shx:
                with tempfile.TemporaryDirectory() as tmpdir:
                    def get_safe_name(f):
                        return os.path.basename(f.name) if hasattr(f, 'name') else str(f)
                    shp_path = Path(tmpdir) / get_safe_name(file_shp)
                    dbf_path = Path(tmpdir) / get_safe_name(file_dbf)
                    shx_path = Path(tmpdir) / get_safe_name(file_shx)
                    with open(shp_path, "wb") as out: out.write(file_shp.read())
                    with open(dbf_path, "wb") as out: out.write(file_dbf.read())
                    with open(shx_path, "wb") as out: out.write(file_shx.read())
                    gdf_tmp = gpd.read_file(str(shp_path))
                    cols = ['Id', 'geometry']
                    if 'StreetName' in gdf_tmp.columns:
                        cols.append('StreetName')
                    gdf_shp = gdf_tmp[cols]
                    st.subheader("Aperçu du fichier SHP")
                    st.dataframe(gdf_shp.head(10).astype(str), use_container_width=True)

            # 4. Lecture des fichiers FCD
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_fcd_paths = []
                dfs_fcd = []
                def get_safe_name(f):
                    return os.path.basename(f.name) if hasattr(f, 'name') else str(f)
                from src.fusion_donnees_fcd_debits import extraire_heure_depuis_fichier
                for fcd_file in fcd_files_uploaded:
                    temp_path = Path(tmpdir) / get_safe_name(fcd_file)
                    with open(temp_path, "wb") as out:
                        out.write(fcd_file.read())
                    with open(temp_path, "rb") as f:
                        df_fcd = safe_read_any_table(f)
                    # Ajout de la colonne 'heure' extraite du nom du fichier
                    heure_val = extraire_heure_depuis_fichier(get_safe_name(fcd_file))
                    df_fcd['heure'] = heure_val
                    id_cols = [col for col in df_fcd.columns if col.lower().endswith('_id')]
                    for col in id_cols:
                        df_fcd.rename(columns={col: 'Id'}, inplace=True)
                    if 'Id' in df_fcd.columns and gdf_shp is not None and 'Id' in gdf_shp.columns:
                        gdf_shp['Id'] = gdf_shp['Id'].astype(int)
                        df_fcd['Id'] = df_fcd['Id'].astype(int)
                        df_fcd = df_fcd.merge(gdf_shp, on='Id', how='left')
                    dfs_fcd.append(df_fcd)
                df_fcd_final = pd.concat(dfs_fcd, ignore_index=True) if dfs_fcd else None
                # Merge with correspondence DataFrame to add capteur and zone columns if missing
                if df_fcd_final is not None:
                    if ('capteur' not in df_fcd_final.columns or 'zone' not in df_fcd_final.columns) and not df_fcd_corresp.empty:
                        df_fcd_final = df_fcd_final.merge(df_fcd_corresp, on='Id', how='left')
                    st.subheader("Aperçu du dataset FCD concaténé")
                    st.dataframe(df_fcd_final.head(10).astype(str), use_container_width=True)

                # 5. Fusion finale
                if df_fcd_final is not None and not df_debits.empty:
                    if 'heure' not in df_fcd_final.columns:
                        st.error("❌ La colonne 'heure' est absente du dataset FCD concaténé. Vérifiez vos fichiers FCD ou le format attendu.")
                        df_fusion = None
                    else:
                        heures_fcd_uniques = sorted(df_fcd_final['heure'].unique())
                        debit_cols = [f"debit_{h}" for h in heures_fcd_uniques]
                        debit_cols_base = ['zone', 'capteur']
                        debit_cols_to_keep = debit_cols_base + [col for col in debit_cols if col in df_debits.columns]
                        df_debits_fcd = df_debits[debit_cols_to_keep].copy() if all(col in df_debits.columns for col in debit_cols_to_keep) else df_debits.copy()
                        st.subheader("Aperçu du fichier de débits filtré pour les heures FCD")
                        st.dataframe(df_debits_fcd.head(10).astype(str), use_container_width=True)
                        df_fusion = fusionner_debits_fcd(df_fcd_final, df_debits_fcd)
                        df_fusion = df_fusion[df_fusion['heure'].isin(heures_fcd_uniques)]
                        st.subheader("Aperçu du dataset fusionné final")
                        st.dataframe(df_fusion.head(10).astype(str), use_container_width=True)
                        # Message utilisateur selon la taille du fichier fusionné
                        if df_fusion is not None:
                            nb_lignes = len(df_fusion)
                            if nb_lignes < 100:
                                st.warning(f"⚠️ Le fichier fusionné contient seulement {nb_lignes} lignes. Il est probable que les Id des fichiers FCD ne soient pas compatibles avec le fichier de correspondance. Vérifiez ou créez un nouveau fichier de correspondance adapté.")
                            else:
                                st.success(f"Le fichier fusionné contient {nb_lignes} lignes. La fusion semble correcte.")
                else:
                    df_fusion = None
            # ...existing code...

            # --- Contrôle des heures uniques dans le dataset final ---
            if df_fusion is not None and not df_fusion.empty:
                if 'heure' in df_fusion.columns:
                    heures_finales = sorted(df_fusion['heure'].dropna().unique())
                    st.info(f"Heures présentes dans le dataset fusionné : {heures_finales}")
                    # Affiche les heures qui ne sont pas dans les fichiers FCD
                    heures_fcd_set = set(heures_fcd)
                    heures_finales_set = set(heures_finales)
                    heures_inattendues = sorted(list(heures_finales_set - heures_fcd_set))
                    if heures_inattendues:
                        st.warning(f"Heures présentes dans le dataset final mais absentes des fichiers FCD : {heures_inattendues}")

                st.session_state.fusion_data = {
                    'dataframe': df_fusion,
                    'filename': "dataset_debit_fcd.csv",
                    'coordinates': detect_coordinate_columns(df_fusion)
                }
                st.success("✅ Fusion réalisée avec succès !")
                df_preview = df_fusion.copy()
                if 'geometry' in df_preview.columns:
                    df_preview['geometry'] = df_preview['geometry'].apply(lambda g: g.wkt if hasattr(g, 'wkt') else str(g))
                st.dataframe(df_preview.head(20).astype(str), use_container_width=True)
                st.download_button(
                    "Télécharger le dataset fusionné (CSV)",
                    data=df_fusion.to_csv(index=False).encode('utf-8'),
                    file_name="dataset_debit_fcd.csv",
                    mime="text/csv"
                )
                if st.button("Confirmer l'utilisation du fichier fusionné pour l'import", key="confirm_fusion_use"):
                    st.session_state.uploaded_data = {
                        'dataframe': df_fusion,
                        'filename': "dataset_debit_fcd.csv",
                        'coordinates': detect_coordinate_columns(df_fusion)
                    }
                    st.session_state['original_data'] = df_fusion.copy()
                    st.success("Le fichier fusionné est prêt pour l'étape d'importation et validation.")
                    st.rerun()
            else:
                st.error("❌ La fusion a échoué ou aucun résultat à afficher.")
    elif not ready:
        st.info("Veuillez déposer tous les fichiers nécessaires pour activer la fusion.")
    

if __name__ == "__main__":
    # Ensure required directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/uploads", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
   
    
    main()