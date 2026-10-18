"""Pruebas unitarias para el módulo de entrada/salida y filtrado geoespacial."""

import pandas as pd

from agebs_ensenada.io import filtrar_agebs_urbanas_ensenada

def test_filtrar_agebs_urbanas_ensenada():
    # Simulamos un extracto censal con datos heterogéneos
    df_mock = pd.DataFrame(
        {
            "ENTIDAD": ["2", "2", "2"],
            "MUN": ["1", "1", "2"],       # Municipio 1 (Ensenada) vs 2 (Tijuana)
            "LOC": ["1", "1", "1"],
            "AGEB": ["0010", "0000", "0010"],  # '0000' es el total de la localidad
        }
    )

    df_filtrado = filtrar_agebs_urbanas_ensenada(df_mock)

    # Solo debe sobrevivir la fila 0 (Ensenada urbana, AGEB válida)
    assert len(df_filtrado) == 1
    # Verifica el relleno con ceros (zfill) -> 02 + 001 + 0001 + 0010
    assert df_filtrado.iloc[0]["CVEGEO"] == "0200100010010"