# Geospatial Traffic Predictor

Cette application Streamlit permet de fusionner des données de trafic (FCD, débits, shapefiles) et de réaliser des analyses, visualisations et prédictions sur le trafic routier.

## Fonctionnalités principales
- Import de fichiers de débits, fichiers FCD (DBF), shapefiles (SHP, DBF, SHX), et fichier de correspondance.
- Fusion intelligente des données FCD et débits selon la période et la correspondance capteur/zone.
- Contrôles sur la validité des fichiers, la période sélectionnée, et la compatibilité des Id.
- Visualisation des datasets intermédiaires et du résultat final.
- Gestion des erreurs et messages d’aide pour l’utilisateur.
- Prêt pour l’entraînement de modèles et l’analyse géospatiale (modules inclus).

## Installation

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/HiAyoub/predict-traffic.git
   cd predict-traffic/GeospatialPredictor
   ```
2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Lancement de l’application

```bash
streamlit run app.py
```

## Utilisation

- Importez vos fichiers via l’interface.
- Sélectionnez la période et les paramètres de fusion.
- Vérifiez les aperçus et messages d’erreur éventuels.
- Téléchargez le dataset fusionné ou poursuivez l’analyse/modélisation.

## Dépendances principales
- streamlit
- pandas
- numpy
- geopandas
- dbfread
- plotly
- chardet

## Déploiement

Vous pouvez déployer l’application sur Streamlit Cloud ou tout serveur compatible Python/Streamlit.

## Structure du projet
- `app.py` : application principale Streamlit
- `src/` : modules de traitement, fusion, visualisation, modélisation
- `data/` : dossiers pour les fichiers importés, fusionnés, et shapefiles
- `logs/` : logs d’exécution

## Auteur
Ayoub Hidara

---
N’hésitez pas à ouvrir une issue ou une pull request pour toute amélioration ou bug !
