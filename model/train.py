import pandas as pd
import numpy as np
import seaborn as sns
import keras
from keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
import shutil

# ── Semilla aleatoria para reproducibilidad ───────────────────────────────────
keras.utils.set_random_seed(42)

# ── 1. Cargar el dataset ──────────────────────────────────────────────────────
# Dataset del Titanic incorporado en Seaborn, sin descarga externa
df = sns.load_dataset("titanic")
print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

# ── 2. Selección y limpieza de features ──────────────────────────────────────
# family_size reemplaza a sibsp y parch para evitar multicolinealidad
data = df[["pclass", "sex", "age", "fare", "embarked", 
           "alone", "sibsp", "parch", "survived"]].copy()

data["family_size"] = data["sibsp"] + data["parch"]
data = data.drop(columns=["sibsp", "parch"])
data = data.dropna()

print(f"Filas después de limpiar nulos: {len(data)}")

# ── 3. Codificación de variables categóricas ──────────────────────────────────
# Convertir texto a números para que Keras pueda procesarlos
data["sex"] = data["sex"].map({"male": 0, "female": 1})
data["embarked"] = data["embarked"].map({"S": 0, "C": 1, "Q": 2})
data["alone"] = data["alone"].astype(int)

# ── 4. Separar features y target ─────────────────────────────────────────────
features = ["pclass", "sex", "age", "fare", "embarked", "alone", "family_size"]
target = "survived"

X = data[features].values
y = data[target].values

# Dividir en entrenamiento y test con estratificación
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Aplanar y si tiene dimensión extra
if len(y_train.shape) > 1:
    y_train = y_train.ravel()
    y_test = y_test.ravel()

# ── 5. Normalización ──────────────────────────────────────────────────────────
# StandardScaler: media 0, desviación estándar 1
# fit_transform en train, solo transform en test (evita data leakage)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ── 6. Class weights ──────────────────────────────────────────────────────────
# Compensa el desbalance entre sobrevivientes y no sobrevivientes
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {k: float(v) for k, v in enumerate(class_weights)}
print(f"Class weights: {class_weight_dict}")

# ── 7. Arquitectura MLP ───────────────────────────────────────────────────────
# MLP para datos tabulares: no necesita capas convolucionales
model = keras.Sequential([
    layers.Input(shape=(7,)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")  # sigmoid para clasificación binaria
])

model.summary()

# ── 8. Compilar ───────────────────────────────────────────────────────────────
# binary_crossentropy para clasificación binaria con sigmoid
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ── 9. Entrenar ───────────────────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    class_weight=class_weight_dict,
    verbose=1
)

# ── 10. Evaluar ───────────────────────────────────────────────────────────────
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_accuracy:.4f}")
print(f"Test loss:     {test_loss:.4f}")

# ── 11. Guardar modelo y scaler ───────────────────────────────────────────────
# .keras: formato recomendado en Keras 3
# .pkl: formato estándar para objetos de Scikit-Learn
model.save("titanic_model.keras")
joblib.dump(scaler, "scaler.pkl")
print("\nModelo guardado como titanic_model.keras")
print("Scaler guardado como scaler.pkl")

# ── 12. Copiar a app/ ─────────────────────────────────────────────────────────
os.makedirs("../app", exist_ok=True)
shutil.copy("titanic_model.keras", "../app/titanic_model.keras")
shutil.copy("scaler.pkl", "../app/scaler.pkl")
print("\nArchivos copiados a ../app/")

