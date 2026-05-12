import os

import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session

from dataset_normalize import coerce_prediction_row
from model_features import FEATURE_COLUMNS

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-set-SECRET_KEY-in-production")

# Cargar Modelo
model = joblib.load("model.pkl")

NIVEL_EDUCATIVO = {
    0: "Sin estudios",
    1: "Bachillerato",
    2: "Técnico/Tecnólogo",
    3: "Universitario",
    4: "Posgrado",
}


@app.route("/", methods=["GET"])
def index():
    error = session.pop("error", None)
    evaluaciones = session.get("evaluaciones", [])
    return render_template("index.html", evaluaciones=evaluaciones, error=error)


@app.route("/evaluar", methods=["POST"])
def evaluar():
    try:
        nombre = request.form.get("nombre", "Candidato sin nombre")
        experiencia = float(request.form["experiencia_anos"])
        nivel_edu_id = int(request.form["nivel_educativo"])

        row = {
            "experiencia_anos": experiencia,
            "ha_trabajado_en_sector": int(request.form["ha_trabajado_en_sector"]),
            "nivel_educativo": nivel_edu_id,
            "disponibilidad_inmediata": int(request.form["disponibilidad_inmediata"]),
            "vive_cerca": int(request.form["vive_cerca"]),
            "acepta_turnos": int(request.form["acepta_turnos"]),
            "documentos_completos": int(request.form["documentos_completos"]),
            "referido_interno": int(request.form["referido_interno"]),
        }
        row = coerce_prediction_row(row)
        input_df = pd.DataFrame([row])[FEATURE_COLUMNS]

        prediction = model.predict(input_df)[0]

        registro = {
            "nombre": nombre,
            "resultado": prediction,
            "experiencia_anos": row["experiencia_anos"],
            "nivel_educativo": NIVEL_EDUCATIVO.get(row["nivel_educativo"], row["nivel_educativo"]),
            "sector": "Sí" if row["ha_trabajado_en_sector"] else "No",
            "disponibilidad": "Inmediata" if row["disponibilidad_inmediata"] else "No inmediata",
            "vive_cerca": "Sí" if row["vive_cerca"] else "No",
            "acepta_turnos": "Sí" if row["acepta_turnos"] else "No",
            "documentos": "Completos" if row["documentos_completos"] else "Incompletos",
            "referido": "Sí" if row["referido_interno"] else "No",
        }

        evaluaciones = session.get("evaluaciones", [])
        evaluaciones.append(registro)
        session["evaluaciones"] = evaluaciones

    except Exception as e:
        session["error"] = str(e)

    return redirect(url_for("index"))


@app.route("/limpiar", methods=["POST"])
def limpiar():
    session.pop("evaluaciones", None)
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
