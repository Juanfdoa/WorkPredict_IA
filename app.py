import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = 'Abc.12345suadyhawuifg' 

# Cargar Modelo
model = joblib.load('model.pkl')

NIVEL_EDUCATIVO = {
    0: 'Sin estudios',
    1: 'Bachillerato',
    2: 'Técnico/Tecnólogo',
    3: 'Universitario',
    4: 'Posgrado'
}

@app.route('/', methods=['GET'])
def index():
    evaluaciones = session.get('evaluaciones', [])
    return render_template('index.html', evaluaciones=evaluaciones)

@app.route('/evaluar', methods=['POST'])
def evaluar():
    try:
        nombre = request.form.get('nombre', 'Candidato sin nombre')
        experiencia = float(request.form['experiencia_anos'])
        nivel_edu_id = int(request.form['nivel_educativo'])

        data = {
            'experiencia_anos':        experiencia,
            'ha_trabajado_en_sector':  int(request.form['ha_trabajado_en_sector']),
            'nivel_educativo':         nivel_edu_id,
            'disponibilidad_inmediata':int(request.form['disponibilidad_inmediata']),
            'vive_cerca':              int(request.form['vive_cerca']),
            'acepta_turnos':           int(request.form['acepta_turnos']),
            'documentos_completos':    int(request.form['documentos_completos']),
            'referido_interno':        int(request.form['referido_interno']),
        }

        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]

        registro = {
            'nombre':           nombre,
            'resultado':        prediction,
            'experiencia_anos': experiencia,
            'nivel_educativo':  NIVEL_EDUCATIVO.get(nivel_edu_id, nivel_edu_id),
            'sector':           'Sí' if data['ha_trabajado_en_sector'] else 'No',
            'disponibilidad':   'Inmediata' if data['disponibilidad_inmediata'] else 'No inmediata',
            'vive_cerca':       'Sí' if data['vive_cerca'] else 'No',
            'acepta_turnos':    'Sí' if data['acepta_turnos'] else 'No',
            'documentos':       'Completos' if data['documentos_completos'] else 'Incompletos',
            'referido':         'Sí' if data['referido_interno'] else 'No',
        }

        evaluaciones = session.get('evaluaciones', [])
        evaluaciones.append(registro)
        session['evaluaciones'] = evaluaciones 

    except Exception as e:
        session['error'] = str(e)

    return redirect(url_for('index'))

@app.route('/limpiar', methods=['POST'])
def limpiar():
    session.pop('evaluaciones', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)