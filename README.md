# WorkPredict — modelo de IA y UI Flask

Repositorio con un modelo de clasificación supervisada para estimar **posibilidad de contratación** (Baja / Media / Alta) y una interfaz web en **Flask** para pruebas locales.

Repositorio público de referencia: `https://github.com/Juanfdoa/WorkPredict_IA`

---

## Documentación de gobernanza y datos

- **[ETHICS_AND_MODEL.md](ETHICS_AND_MODEL.md)** — Model card, política de uso, riesgos de variables proxy y postura ética explícita.
- **[DATASET.md](DATASET.md)** — Esquema del CSV, campos futuros recomendados y plan de mejora del dataset.
- **[context.md](context.md)** — Mapa de contexto del proyecto (stack, flujos, discrepancias históricas resueltas en código).

---

## Cómo usar el repositorio

### 1. Clonar

```bash
git clone https://github.com/Juanfdoa/WorkPredict_IA
cd WorkPredict_IA
```

### 2. Dependencias

```bash
pip install -r requirements.txt
```

### 3. Entrenar y guardar el modelo

```bash
python model.py
```

El script:

- Lee `data/dataset.csv` (encoding **latin-1**) y normaliza en memoria con [`dataset_normalize.py`](dataset_normalize.py): clases canónicas, dominio de features, filas inválidas eliminadas y **duplicados exactos** quitados (el CSV en disco no cambia salvo que lo edites a mano).
- Particiona **70 % entrenamiento / 30 % prueba** estratificada (`test_size=0.30`, `random_state=42`).
- Compara **DecisionTree**, **RandomForest** y **HistGradientBoosting** con búsqueda de hiperparámetros y validación cruzada **estratificada de 5 pliegues** sobre el train, optimizando **F1 macro**.
- Imprime métricas en el holdout (accuracy, F1 macro/por clase, matriz de confusión).
- Reentrena el ganador con el **100 %** de las filas y escribe **`model.pkl`** en la raíz del proyecto.

### 4. Interfaz web

```bash
python app.py
```

Abre el servidor de desarrollo y usa el formulario en `/`.

**Producción:** define la variable de entorno `SECRET_KEY` (no uses la clave por defecto del código). Despliegue con Gunicorn según `Procfile`.

**Estilos:** hoja principal en `static/css/style.css` (no `index.css`).

---

## Columnas de datos y código compartido

Las columnas de entrenamiento e inferencia están centralizadas en [`model_features.py`](model_features.py) para evitar desalineación entre `model.py` y `app.py`.

---

## Errores en el formulario

Si falla el envío (datos inválidos, etc.), el mensaje se guarda en sesión y se muestra en la página principal en el siguiente GET.
