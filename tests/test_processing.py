"""Pruebas unitarias para el módulo de procesamiento y deltas censales."""

import numpy as np
import pandas as pd

from agebs_ensenada.processing import (
    calcular_deltas_longitudinales,
    calcular_indicadores_porcentuales,
    convertir_columnas_a_numerico,
    limpiar_caracteres_especiales_inegi,
)

def test_limpiar_caracteres_especiales_inegi():
    df = pd.DataFrame({"POBTOT": ["100", "*", "N/D", "50"]})
    df_limpio = limpiar_caracteres_especiales_inegi(df)
    
    assert pd.isna(df_limpio.loc[1, "POBTOT"])
    assert pd.isna(df_limpio.loc[2, "POBTOT"])
    assert df_limpio.loc[0, "POBTOT"] == "100"

def test_convertir_columnas_a_numerico():
    df = pd.DataFrame({"POBTOT": ["100", None, "50"]})
    df_num = convertir_columnas_a_numerico(df, ["POBTOT"])
    
    assert df_num["POBTOT"].dtype in [np.float64, np.int64]
    assert df_num.loc[1, "POBTOT"] == 0.0

def test_calcular_indicadores_porcentuales():
    df = pd.DataFrame(
        {
            "poblacion_18_mas": [100, 0],
            "pob_18_mas_posbasica": [40, 0],
            "total_viviendas_habitadas": [50, 10],
            "viv_con_computadora": [25, 5],
            "viv_con_internet": [30, 2],
            "viv_con_automovil": [40, 8],
        }
    )
    df_res = calcular_indicadores_porcentuales(df)
    
    # % Educación posbásica (40/100 * 100)
    assert df_res.loc[0, "pct_educacion_posbasica"] == 40.0
    # Caso borde: división por cero amortiguada
    assert df_res.loc[1, "pct_educacion_posbasica"] == 0.0
    # % Viviendas con computadora (25/50 * 100)
    assert df_res.loc[0, "pct_viv_computadora"] == 50.0

def test_calcular_deltas_longitudinales():
    df_2010 = pd.DataFrame(
        {
            "CVEGEO": ["0200100010010"],
            "poblacion_total": [100.0],
            "pct_educacion_posbasica": [20.0],
            "promedio_ocupantes_hogar": [4.0],
            "pct_viv_computadora": [30.0],
            "pct_viv_internet": [20.0],
            "pct_viv_automovil": [50.0],
        }
    )
    df_2020 = pd.DataFrame(
        {
            "CVEGEO": ["0200100010010"],
            "poblacion_total": [120.0],
            "pct_educacion_posbasica": [30.0],
            "promedio_ocupantes_hogar": [3.5],
            "pct_viv_computadora": [40.0],
            "pct_viv_internet": [50.0],
            "pct_viv_automovil": [60.0],
        }
    )
    deltas = calcular_deltas_longitudinales(df_2010, df_2020)

    # Crecimiento poblacional del 20% ((120-100)/100 * 100)
    assert deltas.loc[0, "delta_poblacion_pct"] == 20.0
    # Incremento de 10 puntos porcentuales en educación (30 - 20)
    assert deltas.loc[0, "delta_pct_educacion_posbasica"] == 10.0
    # Reducción en tamaño del hogar (-0.5)
    assert deltas.loc[0, "delta_promedio_ocupantes_hogar"] == -0.5