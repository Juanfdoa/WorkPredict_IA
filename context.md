# Contexto del proyecto WorkPredict_IA

Documento generado a partir de los archivos del repositorio: `README.md`, código fuente (`app.py`, `model.py`), `requirements.txt`, `Procfile`, plantillas y estilos, muestra del dataset, y el informe académico en Word **«Creación de un modelo de inteligencia artificial avanzada (1).docx»**.

---

## 1. Propósito del producto

**WorkPredict** es un sistema de aprendizaje supervisado orientado a **recursos humanos**: a partir de atributos de un candidato, predice la **posibilidad de contratación** en tres clases categóricas: **Baja**, **Media** y **Alta**.

El informe del `.docx` enmarca el proyecto como trabajo académico (Electiva II – Inteligencia Artificial Avanzada, Unidad 2, Tecnológica del Oriente, Medellín 2026). **Integrantes:** Juan Fernando Acevedo Patiño, Julián Andrés Meneses Carvajal. **Docente:** José Fabián Díaz Silva.

Motivación descrita en la documentación: reducir sesgo subjetivo, unificar criterios, ahorrar tiempo en revisión manual y aprovechar patrones en datos históricos de contratación.

---

## 2. Stack tecnológico

| Componente | Uso |
|-------------|-----|
| **Python** | Entrenamiento e inferencia |
| **pandas** | Carga y manipulación del CSV (`latin-1`) |
| **scikit-learn** | `DecisionTreeClassifier`, `train_test_split`, métricas |
| **joblib** | Carga del modelo serializado `model.pkl` |
| **Flask 3.x** | API web y rutas |
| **Jinja2** | Plantilla `templates/index.html` |
| **gunicorn** | Despliegue (`Procfile`: `gunicorn app:app`) |

Repositorio público referenciado: `https://github.com/Juanfdoa/WorkPredict_IA`  
Despliegue de ejemplo referenciado en el informe: `https://workpredict-ia.onrender.com/`

---

## 3. Datos (features y target)

**Archivo:** `data/dataset.csv` — filas de candidatos con columnas numéricas/binarias y ordinales.

**Features usadas en entrenamiento e inferencia:**

| Variable | Tipo / codificación |
|----------|---------------------|
| `experiencia_anos` | Numérica (años, puede ser decimal en la UI) |
| `ha_trabajado_en_sector` | 0/1 |
| `nivel_educativo` | Ordinal 0–4: Sin estudios, Bachillerato, Técnico/Tecnólogo, Universitario, Posgrado |
| `disponibilidad_inmediata` | 0/1 |
| `vive_cerca` | 0/1 |
| `acepta_turnos` | 0/1 |
| `documentos_completos` | 0/1 |
| `referido_interno` | 0/1 |

**Target:** `posibilidad_contratacion` — valores de texto **Baja**, **Media**, **Alta** (clasificación multiclase).

El dataset en repo tiene del orden de **cientos de filas** (cabecera + ~800 líneas en el archivo según lectura parcial).

---

## 4. Modelo y entrenamiento (`model.py`)

- Lee `data/dataset.csv` y aplica preparación con **`dataset_normalize.py`**: etiquetas canónicas (Baja/Media/Alta), coerción de dominio en features, eliminación de filas inválidas y **deduplicación exacta** antes de entrenar.
- Partición: `train_test_split` con **`test_size=0.30`** estratificada y `random_state=42` (70 % / 30 %). *Nota: el informe Word puede citar 80/20; el código efectivo es 70/30 — documentado en README.*
- Comparación de modelos con **`GridSearchCV`**, validación cruzada **estratificada (5 folds)** y criterio **`f1_macro`**: `DecisionTreeClassifier`, `RandomForestClassifier`, `HistGradientBoostingClassifier`; se selecciona el mejor en CV sobre train.
- Métricas en holdout: accuracy, F1 macro/por clase, `classification_report`, matriz de confusión.
- Tras la selección, el mejor estimador se **reentrena con el 100 %** de las filas y se guarda con **`joblib.dump`** en **`model.pkl`**.
- Columnas de features compartidas con la app: **`model_features.py`** (`FEATURE_COLUMNS`).

---

## 5. Aplicación web (`app.py` + plantillas)

- Al arrancar, **`joblib.load('model.pkl')`**.
- **`/`** (GET): muestra formulario e historial de evaluaciones guardado en **sesión Flask**; pasa `error` si hubo fallo en el último POST.
- **`/evaluar`** (POST): lee el formulario, aplica **`coerce_prediction_row`** (mismas reglas de dominio que el entrenamiento), construye un `DataFrame` ordenado con `FEATURE_COLUMNS` de **`model_features.py`**, ejecuta `model.predict`, arma un registro legible y lo añade a `session['evaluaciones']`.
- **`/limpiar`** (POST): borra el historial de sesión.
- Mapeo de etiquetas de nivel educativo coherente con el dataset y el formulario.

**Interfaz (`templates/index.html` + `static/css/style.css`):** layout en dos columnas (formulario + historial), badges por resultado (Alta/Media/Baja), botón limpiar. *El informe Word cita en la tabla de estructura `static/css/index.css`; en el repo el archivo real es **`static/css/style.css`**.*

**Seguridad / operación:** `app.secret_key` usa **`os.environ.get('SECRET_KEY', ...)`** con valor por defecto solo para desarrollo; en producción definir `SECRET_KEY`.

**Errores:** en GET `/`, se pasa **`error=session.pop('error', None)`** a la plantilla para mostrar fallos de `/evaluar`.

---

## 6. Cómo ejecutar (según README y código)

1. **Entrenar / reentrenar:** `python model.py` (sobrescribe `model.pkl` al finalizar).
2. **Probar la UI:** `python app.py` (modo debug según `if __name__ == '__main__'`).
3. **Dependencias:** `pip install -r requirements.txt`.

---

## 7. Marco teórico resumido (del documento Word)

- IA y aprendizaje automático; énfasis en **aprendizaje supervisado** para clasificación.
- **Árboles de decisión y ensambles:** el informe puede centrarse en árboles; el código de entrenamiento actual compara además **bosque aleatorio** e **histogram-based gradient boosting** (sklearn) con validación cruzada.
- **Métricas:** `classification_report` (precisión, recall, F1, soporte por clase), `accuracy_score`, F1 macro y matriz de confusión en el holdout.
- **joblib** para persistencia en `.pkl`.
- **Flask + Jinja2** como capa de presentación ligera sobre el modelo.

---

## 8. Referencias bibliográficas / enlaces citados en el informe

- IBM – scikit-learn  
- Arsys – Flask Python  
- DataCamp – árboles de decisión en Python  
- Documentación joblib  
- Jessup – IA y machine learning (artículo general)

---

## 9. Estado actual útil para siguientes pasos

- **Pipeline:** datos → comparación de modelos tabulares + CV → `model.pkl` → Flask; README y `context.md` alineados con 70/30, `style.css`, `SECRET_KEY` y errores en UI.
- **Gobernanza:** ver **`ETHICS_AND_MODEL.md`** (model card y política de uso) y **`DATASET.md`** (esquema enriquecido y plan de recolección).
- **Pendientes opcionales:** persistencia del historial fuera de sesión; si el informe Word sigue citando 80/20 o solo árboles de decisión, actualizar ese documento aparte del repo.

Este archivo sirve como **mapa de contexto** para continuar el desarrollo o la revisión del proyecto sin depender de reabrir todos los orígenes.
