"""Módulo de Entrada/Salida (I/O) para datos censales y geoespaciales de INEGI."""

from pathlib import Path
import geopandas as gpd
import pandas as pd

# Clave Geográfica Estatal y Municipal para Ensenada, B.C.
CVE_ENT = "02"   # Baja California
CVE_MUN = "001"  # Ensenada

def cargar_censo_csv(filepath: Path) -> pd.DataFrame:
    """Carga y realiza una validación inicial de un archivo CSV censal de INEGI.

    Args:
        filepath (Path): Ruta al archivo CSV con datos del censo.

    Returns:
        pd.DataFrame: DataFrame cargado y estructurado.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"El archivo censal no existe en: {filepath}")

    # Forzamos tipo string en claves geográficas para evitar perder ceros a la izquierda
    df = pd.read_csv(filepath, dtype={"ENTIDAD": str, "MUN": str, "LOC": str, "AGEB": str})
    return df

def filtrar_agebs_urbanas_ensenada(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra los registros censales correspondientes a las AGEBs urbanas de Ensenada.

    Args:
        df (pd.DataFrame): DataFrame con datos censales estatales o municipales.

    Returns:
        pd.DataFrame: Registro filtrado únicamente para la mancha urbana de Ensenada.
    """
    # Filtrar Entidad 02 (BC), Municipio 001 (Ensenada) y Locality 0001 (Cabecera)
    mask = (
        (df["ENTIDAD"].astype(str).str.zfill(2) == CVE_ENT)
        & (df["MUN"].astype(str).str.zfill(3) == CVE_MUN)
        & (df["LOC"].astype(str).str.zfill(4) == "0001")
        & (df["AGEB"].notna())
        & (df["AGEB"] != "0000")  # Excluir totales municipales/localidad
    )
    
    df_filtrado = df[mask].copy()
    
    # Construir la clave AGEB completa de 13 dígitos (CVEGEO)
    # Ej: 02 (Entidad) + 001 (Mun) + 0001 (Loc) + 045A (AGEB)
    df_filtrado["CVEGEO"] = (
        df_filtrado["ENTIDAD"].astype(str).str.zfill(2)
        + df_filtrado["MUN"].astype(str).str.zfill(3)
        + df_filtrado["LOC"].astype(str).str.zfill(4)
        + df_filtrado["AGEB"].astype(str)
    )
    
    return df_filtrado

def cargar_poligonos_agebs(shapefile_path: Path) -> gpd.GeoDataFrame:
    """Carga el Marco Geoestadístico de INEGI y filtra los vectores de las AGEBs.

    Args:
        shapefile_path (Path): Ruta al archivo Shapefile o GeoPackage (.shp / .gpkg).

    Returns:
        gpd.GeoDataFrame: Polígonos de las AGEBs urbanas de Ensenada.
    """
    if not shapefile_path.exists():
        raise FileNotFoundError(f"El archivo geoespacial no existe en: {shapefile_path}")

    gdf = gpd.read_file(shapefile_path)
    
    # Asegurar proyección estándar (WGS84 para mapas web / Streamlit)
    if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)
        
    return gdf