# Dataset WorkPredict: esquema y plan de mejora

## 1. Esquema actual (`data/dataset.csv`)

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `experiencia_anos` | float | Años de experiencia laboral relevante. |
| `ha_trabajado_en_sector` | 0/1 | Experiencia previa en el sector del puesto. |
| `nivel_educativo` | 0–4 | Ordinal: sin estudios, bachillerato, técnico/tecnólogo, universitario, posgrado. |
| `disponibilidad_inmediata` | 0/1 | Puede incorporarse de forma inmediata. |
| `vive_cerca` | 0/1 | Residencia cercana al lugar de trabajo (definir radio o tiempo de desplazamiento en el diccionario de negocio). |
| `acepta_turnos` | 0/1 | Aceptación de turnos rotativos o no estándar. |
| `documentos_completos` | 0/1 | Expediente requerido completo al momento de la evaluación. |
| `referido_interno` | 0/1 | Referencia por empleado o red interna. |
| `posibilidad_contratacion` | texto | Objetivo: `Baja`, `Media`, `Alta`. |

Codificación de archivo: **latin-1**. La lectura para entrenamiento pasa por [`dataset_normalize.py`](dataset_normalize.py) (ver §1.1).

### 1.1 Normalización al cargar (entrenamiento)

El CSV **no se modifica en disco** por defecto; la limpieza ocurre en memoria al ejecutar [`model.py`](model.py):

- **Objetivo:** texto normalizado Unicode (NFKC), sin espacios sobrantes; solo se aceptan variantes de *baja* / *media* / *alta* que mapean a `Baja`, `Media`, `Alta`. Cualquier otro valor se descarta con conteo en el informe de preparación.
- **Features numéricas y binarias:** `experiencia_anos` en \([0, 60]\)` (valores no numéricos → fila eliminada); binarias forzadas a \(\{0,1\}\); `nivel_educativo` en \([0, 4]\) por recorte.
- **Duplicados exactos:** misma fila de ocho variables y mismo objetivo → se conserva una sola copia (en el CSV actual se eliminan 53 filas duplicadas respecto a 800 originales; el entrenamiento usa 747 filas únicas).

Auditoría puntual del repo (última verificación automática): distribución aproximada tras deduplicación — Media 436, Alta 200, Baja 111. *Volver a ejecutar `python model.py` y leer el bloque “Preparación de datos” para cifras actualizadas si el CSV cambia.*

---

## 2. Esquema enriquecido recomendado (futuro)

Añadir columnas solo si hay proceso de recolección y base legal; documentar cada una en la misma tabla.

| Columna propuesta | Tipo | Uso |
|-------------------|------|-----|
| `id_candidato` | UUID / hash | Trazabilidad sin exponer nombre en el CSV de entrenamiento. |
| `fecha_evaluacion` | ISO 8601 | Splits temporales y detección de deriva. |
| `id_vacante` o `rol_codigo` | categoría | Modelos por familia de puesto; evita mezclar criterios incompatibles. |
| `canal_aplicacion` | categoría | Web, referido, feria, etc.; útil para representatividad. |
| `contratado_final` | 0/1 | Outcome real post-proceso; permite modelos de contratación real vs intención. |
| `etiquetador_id` | categoría | Control de calidad entre revisores. |
| `version_criterio` | entero | Cuando cambie la guía de etiquetado Baja/Media/Alta. |

**No recomendado** en el CSV de entrenamiento sin marco legal explícito: variables que identifiquen directamente características protegidas. Si se hace auditoría de equidad con datos sensibles, debe ser en entorno restringido y con anonimización acordada.

---

## 3. Definición operativa del objetivo

Documentar por escrito, por ejemplo:

- **Baja:** no continúa en el proceso o no recomendable con la evidencia disponible en ese momento.
- **Media:** candidato viable con dudas o competencia ajustada al perfil.
- **Alta:** fuerte alineación con el perfil y se recomienda avanzar.

Indicar **en qué fase** se asigna la etiqueta (solo CV, tras entrevista técnica, etc.) para alinear el modelo con datos disponibles en inferencia y evitar **fuga de información** (p. ej. etiquetar “Alta” solo después de conocer el resultado de contratación usando señales no disponibles en el formulario).

---

## 4. Calidad y consistencia

1. **Guía de etiquetado** de una página, con ejemplos límite por clase.
2. **Doble codificación** en una muestra piloto (dos revisores); medir acuerdo (p. ej. kappa de Cohen); iterar la guía hasta aceptable convergencia.
3. **Resolución de conflictos** por un tercer revisor o regla mayoritaria documentada.

---

## 5. Plan de recolección incremental

| Fase | Acción |
|------|--------|
| 1 | Congelar definición de clases y variables actuales; registrar versión del CSV. |
| 2 | Incorporar solo filas con etiquetado según la guía; rechazar filas ambiguas sin resolución. |
| 3 | Añadir metadatos mínimos (`fecha_evaluacion`, `rol_codigo`) cuando sea viable. |
| 4 | Objetivo de volumen: miles de filas **por segmento homogéneo** (mismo tipo de vacante) antes de pretender generalización fina entre roles muy distintos. |
| 5 | Revisar balance de clases; si una clase es rara, documentar uso de `class_weight` u oversampling **solo en entrenamiento**. |
| 6 | Revisión trimestral de deriva: comparar distribución de features y de `posibilidad_contratacion` vs línea base. |

---

## 6. Privacidad

- Minimizar datos personales en archivos usados para entrenamiento colaborativo.
- El nombre en la UI de Flask es solo para historial de sesión; no debe alimentar el CSV de entrenamiento sin consentimiento y política de tratamiento.

---

## 7. Referencias cruzadas

- Model card y riesgos de variables proxy: [ETHICS_AND_MODEL.md](ETHICS_AND_MODEL.md).
- Lista de columnas usadas por código: [model_features.py](model_features.py).
