"""Módulo de procesamiento, limpieza y armonización longitudinal (2010-2020)."""

import numpy as np
import pandas as pd

# Mapeo estándar de variables censales INEGI
COLUMNAS_CLAVE = {
    "POBTOT": "poblacion_total",
    "P_18YMAS": "poblacion_18_mas",
    "P18YM_PB": "pob_18_mas_posbasica",  # Educación superior/posbásica
    "PROMHOG": "promedio_ocupantes_hogar",
    "TVIVHAB": "total_viviendas_habitadas",
    "VPH_PC": "viv_con_computadora",
    "VPH_INTER": "viv_con_internet",
    "VPH_AUTOM": "viv_con_automovil",
}

def limpiar_caracteres_especiales_inegi(df: pd.DataFrame) -> pd.DataFrame:
    """Reemplaza valores confidenciales o no disponibles de INEGI ('*', 'N/D') por NaN o cero.

    INEGI utiliza '*' en AGEBs con poca población por motivos de confidencialidad.

    Args:
        df (pd.DataFrame): DataFrame con datos censales sin procesar.

    Returns:
        pd.DataFrame: DataFrame con valores numéricos limpios.
    """
    df_limpio = df.copy()

    # Reemplazar caracteres especiales de confidencialidad censal
    df_limpio = df_limpio.replace({"*": np.nan, "N/D": np.nan, "N/A": np.nan})

    return df_limpio

def convertir_columnas_a_numerico(
    df: pd.DataFrame, columnas: list[str]
) -> pd.DataFrame:
    """Coacciona columnas censales a tipos numéricos (float/int).

    Args:
        df (pd.DataFrame): DataFrame censal.
        columnas (list[str]): Lista de nombres de columnas a convertir.

    Returns:
        pd.DataFrame: DataFrame con columnas numéricas convertidas.
    """
    df_num = df.copy()
    for col in columnas:
        if col in df_num.columns:
            df_num[col] = pd.to_numeric(df_num[col], errors="coerce").fillna(0)
    return df_num

def calcular_indicadores_porcentuales(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula las proporciones y porcentajes socioeconómicos clave.

    Args:
        df (pd.DataFrame): DataFrame censal con variables absolutas limpias.

    Returns:
        pd.DataFrame: DataFrame con variables relativas (%) agregadas.
    """
    df_ind = df.copy()
    
    df_ind.rename(columns=COLUMNAS_CLAVE)

    # % Población con educación postbásica sobre la población de 18 años o más
    # No puedes comparar una AGEB gigantesca de 5,000 habitantes con una pequeña 
    # de 200 habitantes usando conteos absolutos.
    if "poblacion_18_mas" in df_ind.columns and "pob_18_mas_posbasica" in df_ind.columns:
        df_ind["pct_educacion_posbasica"] = np.where(
            df_ind["poblacion_18_mas"] > 0, # protección contra división por cero
            (df_ind["pob_18_mas_posbasica"] / df_ind["poblacion_18_mas"]) * 100,
            0.0, # Si el denominador es mayor a 0, calcula el porcentaje; si es 0, asigna directamente 0.0
        )

    # % Viviendas con computadora
    if "total_viviendas_habitadas" in df_ind.columns and "viv_con_computadora" in df_ind.columns:
        df_ind["pct_viv_computadora"] = np.where(
            df_ind["total_viviendas_habitadas"] > 0, # protección contra división por cero
            (df_ind["viv_con_computadora"] / df_ind["total_viviendas_habitadas"]) * 100,
            0.0,
        )

    # % Viviendas con internet
    if "total_viviendas_habitadas" in df_ind.columns and "viv_con_internet" in df_ind.columns:
        df_ind["pct_viv_internet"] = np.where(
            df_ind["total_viviendas_habitadas"] > 0,
            (df_ind["viv_con_internet"] / df_ind["total_viviendas_habitadas"]) * 100,
            0.0,
        )

    # % Viviendas con automóvil
    if "total_viviendas_habitadas" in df_ind.columns and "viv_con_automovil" in df_ind.columns:
        df_ind["pct_viv_automovil"] = np.where(
            df_ind["total_viviendas_habitadas"] > 0,
            (df_ind["viv_con_automovil"] / df_ind["total_viviendas_habitadas"]) * 100,
            0.0,
        )

    return df_ind

def calcular_deltas_longitudinales(
    df_2010: pd.DataFrame, df_2020: pd.DataFrame
) -> pd.DataFrame:
    """Calcula las variaciones absolutas y porcentuales (deltas 2020 - 2010) por AGEB.

    Args:
        df_2010 (pd.DataFrame): Indicadores procesados de 2010.
        df_2020 (pd.DataFrame): Indicadores procesados de 2020.

    Returns:
        pd.DataFrame: DataFrame unificado con las variaciones intercensales por CVEGEO.
    """
    # Cruzar datos de ambos censos por clave única AGEB (CVEGEO)
    merged = pd.merge(
        df_2010,
        df_2020,
        on="CVEGEO",
        suffixes=("_2010", "_2020"),
        how="inner",
    )

    deltas = pd.DataFrame({"CVEGEO": merged["CVEGEO"]})

    # Deltas de variables clave para el ITSE
    # 1. Crecimiento poblacional relativo (%)
    deltas["delta_poblacion_pct"] = np.where(
        merged["poblacion_total_2010"] > 0,
        ((merged["poblacion_total_2020"] - merged["poblacion_total_2010"])
         / merged["poblacion_total_2010"]) * 100,
        0.0,
    )

    # 2. Cambio en la tasa de educación posbásica (puntos porcentuales)
    deltas["delta_pct_educacion_posbasica"] = (
        merged["pct_educacion_posbasica_2020"] - merged["pct_educacion_posbasica_2010"]
    )

    # 3. Cambio en tamaño medio del hogar (diferencia directa de promedio)
    deltas["delta_promedio_ocupantes_hogar"] = (
        merged["promedio_ocupantes_hogar_2020"] - merged["promedio_ocupantes_hogar_2010"]
    )

    # 4. Cambio en % viviendas con equipamiento tecnológico (puntos porcentuales)
    deltas["delta_pct_viv_computadora"] = (
        merged["pct_viv_computadora_2020"] - merged["pct_viv_computadora_2010"]
    )
    deltas["delta_pct_viv_internet"] = (
        merged["pct_viv_internet_2020"] - merged["pct_viv_internet_2010"]
    )
    deltas["delta_pct_viv_automovil"] = (
        merged["pct_viv_automovil_2020"] - merged["pct_viv_automovil_2010"]
    )

    return deltas