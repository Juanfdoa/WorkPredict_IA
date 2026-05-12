import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

df = pd.read_csv("data/dataset.csv", encoding="latin-1")

features = [
    "experiencia_anos",
    "ha_trabajado_en_sector",
    "nivel_educativo",
    "disponibilidad_inmediata",
    "vive_cerca",
    "acepta_turnos",
    "documentos_completos",
    "referido_interno",
]

target = "posibilidad_contratacion"

X = df[features]
Y = df[target]

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.30, 
    random_state=42 
)

model = DecisionTreeClassifier(random_state=42) 
model.fit(X_train, Y_train)

prediction = model.predict(X_test)
print(f"Accuracy: {accuracy_score(Y_test, prediction) * 100:.2f}%")
print(classification_report(Y_test, prediction))

nuevo_candidate = pd.DataFrame({
    "experiencia_anos" :[8],
    "ha_trabajado_en_sector" :[1],
    "nivel_educativo" :[3],
    "disponibilidad_inmediata" :[1],
    "vive_cerca" :[1],
    "acepta_turnos" :[1],
    "documentos_completos" :[1],
    "referido_interno" :[1],
})

resultado = model.predict(nuevo_candidate)

print("\nPredicción para el nuevo candidato:", resultado[0])

#joblib.dump(model, 'model.pkl')
#print("Modelo guardado como 'model.pkl'")
