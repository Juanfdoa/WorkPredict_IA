"""
Entrenamiento WorkPredict: comparación de modelos tabulares con CV estratificada,
holdout 70/30 y guardado del mejor estimador en model.pkl.
"""
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

from dataset_normalize import load_and_prepare_csv
from model_features import FEATURE_COLUMNS, TARGET_COLUMN

DATA_PATH = "data/dataset.csv"
MODEL_PATH = "model.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.30
CV_SPLITS = 5


def load_xy():
    X, y, report = load_and_prepare_csv(DATA_PATH, encoding="latin-1")
    return X, y, report


def main():
    X, y, prep_report = load_xy()

    print("=== Preparación de datos (normalización + deduplicación) ===")
    for k, v in prep_report.items():
        print(f"  {k}: {v}")
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search_specs = [
        (
            "DecisionTree",
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "max_depth": [6, 10, 15, None],
                "min_samples_leaf": [2, 4, 8],
                "class_weight": [None, "balanced"],
            },
        ),
        (
            "RandomForest",
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "n_estimators": [100, 200],
                "max_depth": [10, 15, None],
                "class_weight": [None, "balanced"],
            },
        ),
        (
            "HistGradientBoosting",
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "max_depth": [4, 7],
                "max_iter": [150, 300],
                "learning_rate": [0.05, 0.1],
            },
        ),
    ]

    best_name = None
    best_gs = None
    best_cv = -1.0

    print("=== Búsqueda por modelo (CV estratificada, scoring=f1_macro, solo train) ===\n")

    for name, estimator, param_grid in search_specs:
        gs = GridSearchCV(
            estimator,
            param_grid,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            refit=True,
        )
        gs.fit(X_train, y_train)
        print(f"{name}: mejor f1_macro CV = {gs.best_score_:.4f}")
        print(f"  Parámetros: {gs.best_params_}")
        if gs.best_score_ > best_cv:
            best_cv = gs.best_score_
            best_gs = gs
            best_name = name

    print(f"\nModelo seleccionado por CV: {best_name}\n")

    y_pred = best_gs.predict(X_test)
    print("=== Holdout 30% (una evaluación; no reutilizar para más decisiones de diseño) ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"F1 macro: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"F1 weighted: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred))
    print("Matriz de confusión (filas=real, columnas=predicho):")
    labels = sorted(y.unique())
    print(confusion_matrix(y_test, y_pred, labels=labels))
    print(f"Orden de etiquetas: {labels}\n")

    final_model = best_gs.best_estimator_
    final_model.fit(X, y)
    joblib.dump(final_model, MODEL_PATH)
    print(f"Modelo reentrenado con el 100% de las filas y guardado en {MODEL_PATH}")

    ejemplo = pd.DataFrame(
        {
            "experiencia_anos": [8],
            "ha_trabajado_en_sector": [1],
            "nivel_educativo": [3],
            "disponibilidad_inmediata": [1],
            "vive_cerca": [1],
            "acepta_turnos": [1],
            "documentos_completos": [1],
            "referido_interno": [1],
        }
    )[FEATURE_COLUMNS]
    print("Predicción de ejemplo:", final_model.predict(ejemplo)[0])


if __name__ == "__main__":
    main()
