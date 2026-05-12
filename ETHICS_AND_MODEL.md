# WorkPredict: Model Card y política de uso

Documento vivo para gobernanza del modelo de **posibilidad de contratación** (clases Baja, Media, Alta). Complementa el código en [`model.py`](model.py) y la interfaz en [`app.py`](app.py).

---

## 1. Resumen del modelo

| Campo | Descripción |
|--------|-------------|
| **Versión** | Depende del commit y de la fecha de entrenamiento; al reentrenar, anotar hash del CSV y versión de `scikit-learn`. |
| **Tipo** | Clasificación supervisada multiclase. |
| **Entrada** | Ocho variables tabulares (ver [DATASET.md](DATASET.md)). |
| **Salida** | Una de: `Baja`, `Media`, `Alta`. |
| **Usuarios previstos** | Personal de RR.HH. o académicos en entorno controlado. |

---

## 2. Propósito y límites de uso (postura ética explícita)

**Propósito:** apoyar la priorización o el análisis exploratorio de candidatos a partir de datos estructurados disponibles **antes** de una decisión de contratación, **no** sustituir entrevista, verificación legal ni criterio humano cualificado.

**Uso prohibido o desaconsejado:**

- Decisión automática final de contratación o despido sin revisión humana.
- Perfilado que afecte derechos sin base legal, transparencia y posibilidad de revisión.
- Inferir características protegidas (origen, género, religión, etc.) a partir de proxies; no incorporar esas variables salvo marco legal explícito y documentado.

**Valores priorizados cuando entran en tensión “acierto” y equidad:**

1. **Prioridad humana:** la salida del modelo es orientativa; la responsabilidad de la decisión sigue siendo de la organización y del revisor.
2. **Igualdad de oportunidades:** no usar el modelo para eludir políticas internas o legales de no discriminación.
3. **Transparencia:** comunicar a las partes relevantes que se usa una herramienta de estimación y qué datos intervienen (sin exponer secretos comerciales innecesarios).
4. **Corrección y revisión:** quienes evalúan candidatos deben poder justificar la decisión final con criterios independientes del score.

---

## 3. Datos de entrenamiento

- **Fuente:** [`data/dataset.csv`](data/dataset.csv), codificación `latin-1`.
- **Limitaciones:** tamaño muestral moderado; ausencia en el CSV actual de atributos demográficos auditables implica que **no se puede cuantificar equidad por subgrupos protegidos** con estos datos solos; la equidad se apoya en políticas, revisión de variables proxy y gobernanza de datos futuros (ver [DATASET.md](DATASET.md)).

### 3.1 Pretratamiento y sesgo de codificación

Antes de entrenar, [`dataset_normalize.py`](dataset_normalize.py) aplica reglas deterministas: etiquetas canónicas (`Baja` / `Media` / `Alta`, sin depender de mayúsculas o espacios), coerción de dominio en features (binarias 0/1, `nivel_educativo` en 0–4, `experiencia_anos` acotada), eliminación de filas con objetivo ilegible o experiencia no numérica, y **deduplicación** de filas idénticas (mismas ocho variables y mismo objetivo). Así se reduce el **sesgo de muestreo** introducido por duplicados exactos y el **sesgo de etiqueta** por variantes de texto; **no** sustituye una auditoría de sustancia (quién etiquetó y con qué criterio).

**Definición operativa del objetivo (`posibilidad_contratacion`):** debe documentarse en organización (momento de la etiqueta: solo CV, post-entrevista, etc.). Mientras no esté fijada por proceso, las métricas son difíciles de interpretar para negocio.

---

## 4. Métricas de rendimiento

El script de entrenamiento reporta, como mínimo:

- Validación cruzada **estratificada** (p. ej. 5 pliegues) sobre el conjunto de entrenamiento del split, con **`f1_macro`** como criterio principal de comparación entre familias de modelos (sensible al desbalance entre clases).
- Partición **70 % entrenamiento / 30 % prueba** estratificada (`random_state=42`) alineada con el código.
- `classification_report` y matriz de confusión sobre el conjunto de prueba **una vez** elegido el mejor modelo por CV (evitar múltiples “miradas” al mismo test para no inflar confianza en la métrica).

**Advertencia:** con pocas observaciones por clase, las métricas tienen **alta varianza**; los intervalos de confianza no están incluidos en el script por defecto pero pueden añadirse con bootstrap si se requiere rigor adicional.

---

## 5. Riesgos de equidad y variables proxy

| Variable | Riesgo |
|----------|--------|
| `referido_interno` | Puede favorecer redes internas y reproducir desigualdades de acceso si no se gestiona con criterios explícitos de mérito y transparencia. |
| `vive_cerca` | Correlaciona con ubicación residencial; puede actuar como proxy socioeconómico o de segregación urbana. |
| `nivel_educativo` | Legítimo para muchos puestos; puede correlacionar con inequidad estructural en acceso a educación si se pondera sin criterio de puesto. |
| `documentos_completos` | Útil operativamente; desigualdad en acceso a documentación debe gestionarse con soporte al candidato, no solo penalización algorítmica. |

**Política sugerida:** revisar periódicamente el peso de estas variables (p. ej. importancias del modelo, ablación); en despliegues sensibles, valorar **no usar** o acotar el uso de `referido_interno` y `vive_cerca` y documentar la decisión.

---

## 6. Sesgo estadístico no deseado

El modelo aprende correlaciones del pasado. Si históricamente ciertos perfiles fueron infravalorados o sobrevalorados, el algoritmo puede **reproducir** ese patrón. Mitigación: datos etiquetados con criterio estable, diversidad de fuentes, pesos de clase, auditorías periódicas y, cuando sea lícito, mediciones de error por segmento.

---

## 7. Mantenimiento y reentrenamiento

- Versionar el CSV y el `.pkl` (nombre o carpeta con fecha).
- Reentrenar cuando cambien procesos de selección, perfiles de vacante o leyes aplicables.
- Tras reentrenar, actualizar esta hoja con métricas nuevas y cualquier cambio de variables.

---

## 8. Contacto y responsabilidad

Asignar en la organización un **responsable** de datos y de cumplimiento para actualizar este documento y la política de privacidad (incluido el tratamiento del nombre en la sesión de la UI frente a datos de entrenamiento).
