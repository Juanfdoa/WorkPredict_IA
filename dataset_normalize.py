"""
Preparación de datos: respuestas canónicas, coerción de features y deduplicación
para que el entrenamiento no herede artefactos de codificación (typos, espacios,
valores fuera de dominio) ni peso inflado por filas duplicadas exactas.
"""
from __future__ import annotations

import unicodedata
from typing import Any

import numpy as np
import pandas as pd

from model_features import FEATURE_COLUMNS, TARGET_COLUMN

# Dominios acordados con el formulario y DATASET.md
BINARY_COLS = [
    "ha_trabajado_en_sector",
    "disponibilidad_inmediata",
    "vive_cerca",
    "acepta_turnos",
    "documentos_completos",
    "referido_interno",
]
NIVEL_MIN, NIVEL_MAX = 0, 4
EXP_MIN, EXP_MAX = 0.0, 60.0


def _strip_label(val: Any) -> str | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = unicodedata.normalize("NFKC", str(val)).strip()
    s = " ".join(s.split())
    if not s:
        return None
    key = s.lower()
    mapping = {"baja": "Baja", "media": "Media", "alta": "Alta"}
    return mapping.get(key)


def normalize_target_series(series: pd.Series) -> pd.Series:
    """Devuelve serie con solo Baja/Media/Alta; filas no mapeables quedan como NA."""
    return series.map(_strip_label)


def _coerce_float(v: Any, lo: float, hi: float) -> tuple[float | None, bool]:
    """Devuelve (valor_clippeado o None si no es numérico, hubo_corrección_incluye_clip)."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None, True
    if x != x:  # NaN
        return None, True
    orig = x
    x = max(lo, min(hi, x))
    return x, orig != x


def _coerce_binary(v: Any) -> tuple[int, bool]:
    try:
        x = int(round(float(v)))
    except (TypeError, ValueError):
        return 0, True
    if x not in (0, 1):
        return int(0 if x < 0 else 1), True
    return x, False


def _coerce_nivel(v: Any) -> tuple[int, bool]:
    try:
        x = int(round(float(v)))
    except (TypeError, ValueError):
        return 0, True
    orig = x
    x = max(NIVEL_MIN, min(NIVEL_MAX, x))
    return x, orig != x


def coerce_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Coacciona columnas de FEATURE_COLUMNS in-place sobre una copia."""
    out = df[FEATURE_COLUMNS].copy()
    corrections: dict[str, int] = {}

    col = "experiencia_anos"
    fixed = 0
    vals = []
    for v in out[col]:
        nv, ch = _coerce_float(v, EXP_MIN, EXP_MAX)
        vals.append(np.nan if nv is None else float(nv))
        if ch or nv is None:
            fixed += 1
    out[col] = vals
    if fixed:
        corrections[f"{col}_clip_o_invalido"] = fixed

    for c in BINARY_COLS:
        fixed = 0
        vals = []
        for v in out[c]:
            nv, ch = _coerce_binary(v)
            vals.append(nv)
            if ch:
                fixed += 1
        out[c] = vals
        if fixed:
            corrections[f"{c}_ajustado"] = fixed

    col = "nivel_educativo"
    fixed = 0
    vals = []
    for v in out[col]:
        nv, ch = _coerce_nivel(v)
        vals.append(nv)
        if ch:
            fixed += 1
    out[col] = vals
    if fixed:
        corrections[f"{col}_ajustado"] = fixed

    return out, corrections


def coerce_prediction_row(row: dict) -> dict:
    """Mismas reglas que entrenamiento para una fila desde el formulario/API."""
    d = dict(row)
    e, _ = _coerce_float(d.get("experiencia_anos"), EXP_MIN, EXP_MAX)
    d["experiencia_anos"] = float(e) if e is not None else 0.0
    for c in BINARY_COLS:
        v, _ = _coerce_binary(d.get(c))
        d[c] = v
    v, _ = _coerce_nivel(d.get("nivel_educativo"))
    d["nivel_educativo"] = v
    return d


def prepare_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Carga lógica: normaliza target, elimina inválidos, coacciona features,
    elimina duplicados exactos (mismas features + mismo target).

    Devuelve (X, y, informe) con y como strings canónicos.
    """
    report: dict[str, Any] = {"filas_entrada": len(df)}

    work = df.copy()
    y_raw = work[TARGET_COLUMN]
    y_norm = normalize_target_series(y_raw)
    invalid = y_norm.isna()
    report["filas_objetivo_desconocido_o_vacio"] = int(invalid.sum())
    work = work.loc[~invalid].copy()
    y_norm = y_norm.loc[~invalid]
    orig_target = work[TARGET_COLUMN].copy()

    n_antes_nan = len(work)
    work = work.dropna(subset=FEATURE_COLUMNS)
    report["filas_eliminadas_nan_en_features"] = n_antes_nan - len(work)
    y_norm = y_norm.loc[work.index]
    orig_target = orig_target.loc[work.index]

    report["etiquetas_reexpresadas_desde_texto"] = int(
        (orig_target.astype(str) != y_norm.astype(str)).sum()
    )

    X, corrections = coerce_feature_frame(work)
    report["correcciones_por_columna"] = corrections

    bad_exp = X["experiencia_anos"].isna()
    report["filas_eliminadas_experiencia_invalida"] = int(bad_exp.sum())
    X = X.loc[~bad_exp]
    y_norm = y_norm.loc[X.index]

    work_clean = X.copy()
    work_clean[TARGET_COLUMN] = y_norm.values

    dup_mask = work_clean.duplicated(
        subset=FEATURE_COLUMNS + [TARGET_COLUMN], keep="first"
    )
    n_dup = int(dup_mask.sum())
    report["filas_duplicadas_exactas_eliminadas"] = n_dup
    work_clean = work_clean.loc[~dup_mask].copy()

    y_final = work_clean[TARGET_COLUMN]
    X_final = work_clean[FEATURE_COLUMNS]
    report["filas_salida"] = len(X_final)
    report["distribucion_clases"] = y_final.value_counts().to_dict()

    return X_final, y_final, report


def load_and_prepare_csv(path: str, encoding: str = "latin-1") -> tuple[pd.DataFrame, pd.Series, dict]:
    df = pd.read_csv(path, encoding=encoding)
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")
    return prepare_training_frame(df)
