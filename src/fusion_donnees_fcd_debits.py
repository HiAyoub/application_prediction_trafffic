import pandas as pd
import numpy as np
import geopandas as gpd
import os
import logging
from pathlib import Path

# Configuration de la journalisation
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Fonctions de traitement des débits ---


def calculer_moyenne_horaire_periode(
    df: pd.DataFrame,
    annee_debut: int,
    annee_fin: int,
    mois_debut: int,
    mois_fin: int
) -> pd.DataFrame:
    """
    Calcule la moyenne horaire des débits sur une période sélectionnée (plusieurs années et/ou plusieurs mois).

    Args:
        df (pd.DataFrame): DataFrame contenant les données de débit.
        annee_debut (int): Année de début de la période.
        annee_fin (int): Année de fin de la période.
        mois_debut (int): Mois de début (1-12).
        mois_fin (int): Mois de fin (1-12).

    Returns:
        pd.DataFrame: DataFrame contenant les moyennes horaires des débits par zone et capteur sur la période sélectionnée.
    """
    logger.info("Début du calcul de la moyenne horaire sur une période sélectionnée.")

    # Vérification des types d'entrée
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df doit être un DataFrame pandas.")
    if not all(isinstance(x, int) for x in [annee_debut, annee_fin, mois_debut, mois_fin]):
        raise TypeError("annee_debut, annee_fin, mois_debut et mois_fin doivent être des entiers.")
    if not (1 <= mois_debut <= 12) or not (1 <= mois_fin <= 12):
        raise ValueError("mois_debut et mois_fin doivent être compris entre 1 et 12.")

    if 'periode' not in df.columns:
        logger.error("La colonne 'periode' est manquante dans le DataFrame.")
        raise ValueError("La colonne 'periode' est manquante dans le DataFrame.")

    debit_columns = [col for col in df.columns if col.startswith('debit_')]
    if not debit_columns:
        logger.error("Aucune colonne de débit trouvée dans le DataFrame.")
        raise ValueError("Aucune colonne de débit trouvée dans le DataFrame.")

    df['periode'] = pd.to_datetime(df['periode'])

    logger.info(f"Filtrage des données de {mois_debut}/{annee_debut} à {mois_fin}/{annee_fin}.")
    mask = (
        (df['periode'].dt.year > annee_debut) & (df['periode'].dt.year < annee_fin)
    ) | (
        (df['periode'].dt.year == annee_debut) & (df['periode'].dt.month >= mois_debut)
    ) | (
        (df['periode'].dt.year == annee_fin) & (df['periode'].dt.month <= mois_fin)
    )
    df_periode = df[mask]

    if df_periode.empty:
        logger.warning("Aucune donnée trouvée après le filtrage par période.")

    grouped = df_periode.groupby(['zone', 'capteur'])[debit_columns].mean().reset_index()
    return grouped

# --- Fonctions de traitement FCD ---

def charger_fichier_geo(path: str) -> gpd.GeoDataFrame:
    """
    Charge un fichier géographique dans un GeoDataFrame.

    Parameters:
    - path (str): Chemin du fichier géographique à charger.

    Returns:
    - gpd.GeoDataFrame: Le GeoDataFrame chargé.
    """
    if not isinstance(path, str):
        raise TypeError("Le chemin du fichier doit être une chaîne de caractères.")
    logger.info('FONCTION : charger_fichier_geo')
    try:
        gdf = gpd.read_file(path)
        logger.info(f'Importation du fichier "{path}" est terminée avec succès')
        return gdf
    except Exception as e:
        logger.error(f'Erreur lors de l\'importation du fichier "{path}" : {e}')
        return None


def extraire_info_depuis_fichier(nom_fichier: str) -> dict:
    """
    Extrait la tranche horaire, le mois et l'année depuis le nom du fichier.
    Format attendu: aout2024_0_11_00-12_00_11.dbf
    """
    if not isinstance(nom_fichier, str):
        raise TypeError("Le nom du fichier doit être une chaîne de caractères.")
    
    logger.info('FONCTION : extraire_info_depuis_fichier')
    try:
        # Exemple de nom de fichier: aout2024_0_11_00-12_00_11.dbf
        parts = nom_fichier.split('_')
        
        # Extraire mois et année depuis la première partie (ex: "aout2024")
        mois_annee_str = parts[0]
        
        # Séparer le texte du mois et l'année
        mois_str = ''.join([c for c in mois_annee_str if not c.isdigit()])
        annee_str = ''.join([c for c in mois_annee_str if c.isdigit()])
        
        # Mapper les noms de mois français aux numéros
        mois_francais = {
            'janvier': 1, 'fevrier': 2, 'mars': 3, 'avril': 4, 'mai': 5, 'juin': 6,
            'juillet': 7, 'aout': 8, 'septembre': 9, 'octobre': 10, 'novembre': 11, 'decembre': 12
        }
        
        mois = mois_francais.get(mois_str.lower(), None)
        annee = int(annee_str) if annee_str else None
        
        # Extraire l'heure depuis la quatrième partie (ex: "11_00-12_00")
        heure_str = parts[3]
        heure = heure_str.split('-')[0]  # Prend la première heure (11_00)
        heure = heure.replace('_', ':')  # Convertit "11_00" en "11:00"
        
        logger.info(f'Extraction des infos pour {nom_fichier}: {mois_str} {annee}, {heure}')
        
        return {
            'heure': heure,
            'mois': mois,
            'annee': annee,
            'mois_str': mois_str,
            'heure_complete': heure_str
        }
    except Exception as e:
        logger.error(f'Erreur lors de l\'extraction des infos pour {nom_fichier} : {e}')
        return None


def extraire_heure_depuis_fichier(nom_fichier: str) -> int:
    """
    Extrait la tranche horaire depuis le nom du fichier.

    Parameters:
    - nom_fichier (str): Nom du fichier contenant l'information horaire.

    Returns:
    - int: La tranche horaire extraite.
    """
    if not isinstance(nom_fichier, str):
        raise TypeError("Le nom du fichier doit être une chaîne de caractères.")
    logger.info('FONCTION : extraire_heure_depuis_fichier')
    try:
        heure_str = nom_fichier.split('_')[3]  # ex: "00-8_00"
        heure_fin = heure_str.split('-')[1]    # ex: "8_00"
        heure = int(heure_fin.split('_')[0])   # ex: "8"
        logger.info(f'Extraction de l\'heure pour le fichier {nom_fichier} : {heure}')
        return heure
    except Exception as e:
        logger.error(f'Erreur lors de l\'extraction de l\'heure pour le fichier {nom_fichier} : {e}')
        return None

def charger_fichiers(dir: str, ext: str = 'dbf', ignore: str = '') -> list[str]:
    """
    Charge tous les fichiers avec une extension donnée dans un répertoire.

    Parameters:
    - dir (str): Chemin du répertoire contenant les fichiers.
    - ext (str): Extension des fichiers à charger (par défaut 'dbf').
    - ignore (str): Nom du fichier à ignorer (par défaut '').

    Returns:
    - list[str]: Liste des noms de fichiers correspondant.
    """
    if not isinstance(dir, str):
        raise TypeError("Le paramètre 'dir' doit être une chaîne de caractères.")
    if not isinstance(ext, str):
        raise TypeError("Le paramètre 'ext' doit être une chaîne de caractères.")
    if not isinstance(ignore, str):
        raise TypeError("Le paramètre 'ignore' doit être une chaîne de caractères.")
    try:
        logger.info('FONCTION : charger_fichiers')
        fichiers = [f for f in os.listdir(dir) if f.endswith(ext) and f != ignore]
        logger.info(f'Importation des fichiers avec l\'extension {ext} est terminée avec succès')
        return fichiers
    except Exception as e:
        logger.error(f'Erreur lors de l\'importation des fichiers avec l\'extension {ext} : {e}')
        return []

def cree_dataset_unique_horaire(input_dir: Path, fichiers_selectionnes: list[str]) -> gpd.GeoDataFrame:
    """
    Crée un dataset unique issu de la jointure entre les datasets de chaque heure.
    Les fichiers à traiter sont passés en argument (sélectionnés via l'interface).

    Parameters:
    - input_dir (Path): Répertoire contenant les fichiers d'entrée.
    - fichiers_selectionnes (list[str]): Liste des fichiers à traiter.

    Returns:
    - gpd.GeoDataFrame: Le GeoDataFrame final contenant les données jointes.
    """
    if not isinstance(input_dir, Path):
        raise TypeError("Le paramètre 'input_dir' doit être un objet de type Path.")
    if not isinstance(fichiers_selectionnes, list) or not all(isinstance(f, str) for f in fichiers_selectionnes):
        raise TypeError("fichiers_selectionnes doit être une liste de chaînes de caractères.")
    logger.info('FONCTION : cree_dataset_unique_horaire')
    try:
        dfs = []
        heures_extraites = []
        for fichier in fichiers_selectionnes:
            try:
                df = charger_fichier_geo(str(input_dir / fichier))
                print(f"Fichier {fichier} : {len(df.columns)} colonnes chargées -> {df.columns.tolist()}")

                df.columns = ['Id', 'AvgTt', 'MedTt', 'ratio', 'AvgSp',
                              'HvgSp', 'MedSp', 'SdSp', 'Hits', 'P5sp',
                              'P10sp', 'P15sp', 'P20sp', 'P25sp', 'P30sp',
                              'P35sp', 'P40sp', 'P45sp', 'P50sp', 'P55sp',
                              'P60sp', 'P65sp', 'P70sp', 'P75sp', 'P80sp',
                              'P85sp', 'P90sp', 'P95sp', 'geometry']
                heure = extraire_heure_depuis_fichier(fichier)
                df['heure'] = heure
                heures_extraites.append(heure)
                logger.info(f"Heure extraite du fichier {fichier}: {heure}")
                dfs.append(df)
            except Exception as e:
                logger.error(f'Erreur lors du traitement du fichier {fichier} : {e}')
        logger.info(f"Liste des heures extraites de tous les fichiers FCD: {heures_extraites}")
        if not dfs:
            logger.error("Aucun fichier FCD n'a pu être chargé.")
            return None
        df_final = pd.concat(dfs, axis=0, ignore_index=True)
        logger.info(f"Heures uniques dans le dataset final de fusion: {sorted(df_final['heure'].unique())}")
        return df_final
    except Exception as e:
        logger.error(f'Erreur lors de la création du dataset unique horaire : {e}')
        return None


    
   

def fusionner_debits_fcd(df_fcd: pd.DataFrame, df_debits: pd.DataFrame) -> pd.DataFrame:
    """
    Fusionne les datasets FCD et débits sur les colonnes zone, capteur et heure.

    Args:
        df_fcd (pd.DataFrame): DataFrame contenant les données FCD.
        df_debits (pd.DataFrame): DataFrame contenant les données de débits.

    Returns:
        pd.DataFrame: DataFrame fusionné contenant les données FCD et débits.
    """
    logger.info("Début de la fusion des datasets FCD et débits.")
    # S'assurer que les colonnes capteur et zone existent
    if 'capteur' not in df_fcd.columns or 'zone' not in df_fcd.columns:
        # Si le fichier de correspondance a été fusionné, les colonnes existent
        # Sinon, lever une erreur explicite
        raise ValueError("Les colonnes 'capteur' et 'zone' doivent être présentes dans le dataset FCD.")

    # On garde toutes les colonnes FCD
    result = df_fcd.copy()

    # Ajout de la colonne 'debit' par correspondance
    def get_debit(row):
        condition = (df_debits['capteur'] == row['capteur']) & (df_debits['zone'] == row['zone'])
        column_name = "debit_" + str(row['heure'])
        if column_name in df_debits.columns:
            values = df_debits.loc[condition, column_name]
            if isinstance(values, pd.Series):
                if not values.empty:
                    return values.iloc[0]
            elif values is not None:
                return values
        return np.nan
    result['debit'] = result.apply(get_debit, axis=1)

    # Réorganise les colonnes pour correspondre à la structure demandée
    fcd_cols = [
        'Id','BS_AvgTt','BS_MedTt','BS_ratio','BS_AvgSp','BS_HvgSp','BS_MedSp','BS_SdSp','BS_Hits','BS_P5sp','BS_P10sp','BS_P15sp','BS_P20sp','BS_P25sp','BS_P30sp','BS_P35sp','BS_P40sp','BS_P45sp','BS_P50sp','BS_P55sp','BS_P60sp','BS_P65sp','BS_P70sp','BS_P75sp','BS_P80sp','BS_P85sp','BS_P90sp','BS_P95sp','heure','geometry','StreetName','capteur','zone','debit'
    ]
    # Garde les colonnes dans l'ordre si elles existent
    final_cols = [col for col in fcd_cols if col in result.columns]
    result = result[final_cols]
    logger.info("Fusion terminée.")
    return result

# --- Exemple d'utilisation pour intégration Streamlit ou script principal ---

if __name__ == "__main__":
    # --- Exemple d'utilisation sans fichiers par défaut ---
    # Les chemins et fichiers doivent être fournis explicitement par l'utilisateur
    # chemin_debits = "../../data/raw_input/debit_horaire_2022_2025.csv"
    # chemin_fcd_corresp = "../../data/raw_input/fcd_corresp_capteurs.csv"
    # input_dir_fcd = Path("../../data/raw_input/fcd_data/jobs_5928114_results_reims_area_full.shapefile/")
    # fichiers_fcd = charger_fichiers(str(input_dir_fcd), ext='dbf', ignore='network.dbf')
    # fichiers_selectionnes = fichiers_fcd  # À remplacer par la sélection utilisateur

    # Paramètres à définir par l'utilisateur
    annee_debut, annee_fin, mois_debut, mois_fin = 2022, 2024, 1, 4

    # --- Nouvelle importation sans fichiers par défaut ---
    print("Veuillez fournir les chemins des fichiers à importer :")
    chemin_debits = input("Chemin du fichier débits (CSV): ")
    chemin_fcd_corresp = input("Chemin du fichier de correspondance FCD/Capteurs (CSV): ")
    input_dir_fcd_str = input("Répertoire des fichiers FCD (.dbf): ")
    input_dir_fcd = Path(input_dir_fcd_str)
    fichiers_fcd = charger_fichiers(str(input_dir_fcd), ext='dbf', ignore='network.dbf')
    fichiers_selectionnes = fichiers_fcd  # À remplacer par la sélection utilisateur si besoin

    # Chargement et transformation des débits
    df_debits_brut = pd.read_csv(chemin_debits, sep=';')
    df_debits = calculer_moyenne_horaire_periode(df_debits_brut, annee_debut, annee_fin, mois_debut, mois_fin)

    # Chargement du fichier de correspondance FCD/capteurs
    df_fcd_corresp = pd.read_csv(chemin_fcd_corresp)

    # Création du dataset FCD à partir des fichiers sélectionnés
    df_fcd = cree_dataset_unique_horaire(input_dir_fcd, fichiers_selectionnes)

    # Fusion des datasets
    if df_fcd is not None and not df_debits.empty:
        # Correction : fusionne le DataFrame FCD issu des fichiers FCD avec les débits
        df_fusion = fusionner_debits_fcd(df_fcd, df_debits)
        # Filtrer pour ne garder que les heures présentes dans les fichiers FCD
        heures_fcd_uniques = sorted(df_fcd['heure'].unique())
        df_fusion = df_fusion[df_fusion['heure'].isin(heures_fcd_uniques)]
        print(df_fusion.info())
        print(df_fusion.head())
        # Optionnel : sauvegarde si souhaitée
        # df_fusion.to_csv("dataset_debit_fcd.csv", index=False)
    else:
        logger.error("Impossible de créer le dataset fusionné : données manquantes.")

