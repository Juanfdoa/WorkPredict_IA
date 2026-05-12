"""Columnas de entrenamiento e inferencia compartidas entre model.py y app.py."""

FEATURE_COLUMNS = [
    "experiencia_anos",
    "ha_trabajado_en_sector",
    "nivel_educativo",
    "disponibilidad_inmediata",
    "vive_cerca",
    "acepta_turnos",
    "documentos_completos",
    "referido_interno",
]

TARGET_COLUMN = "posibilidad_contratacion"
