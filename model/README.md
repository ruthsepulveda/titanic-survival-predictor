# Modelo: Predicción de Supervivencia en el Titanic

## Descripción
Red neuronal MLP entrenada con el dataset del Titanic para predecir
si un pasajero habría sobrevivido al hundimiento del RMS Titanic
el 15 de abril de 1912.

## Dataset
Dataset del Titanic disponible directamente en Seaborn
(`seaborn.load_dataset("titanic")`). No requiere descarga externa.

- Filas originales: 891
- Filas después de limpieza: 712
- Features utilizadas: 7
- Variable objetivo: survived (0 = no sobrevivió, 1 = sobrevivió)

## Features del modelo

| Feature | Descripción | Tipo |
|---|---|---|
| pclass | Clase del pasaje (1, 2, 3) | Entero |
| sex | Sexo (0=male, 1=female) | Binario |
| age | Edad en años | Float |
| fare | Precio del pasaje | Float |
| embarked | Puerto de embarque (0=S, 1=C, 2=Q) | Entero |
| alone | ¿Viajaba solo? (0=no, 1=sí) | Binario |
| family_size | Total de familiares a bordo | Entero |

## Arquitectura

| Capa | Tipo | Detalles |
|---|---|---|
| 1 | Input | 7 features |
| 2 | Dense | 64 neuronas, activación ReLU |
| 3 | Dropout | 30% |
| 4 | Dense | 32 neuronas, activación ReLU |
| 5 | Dropout | 30% |
| 6 | Dense | 1 neurona, activación Sigmoid |

## Entrenamiento

| Parámetro | Valor |
|---|---|
| Epochs | 50 |
| Batch size | 32 |
| Optimizador | Adam |
| Función de pérdida | Binary Crossentropy |
| Class weight | Balanceado |
| Parámetros entrenables | 2,625 |

## Métricas finales

| Métrica | Valor |
|---|---|
| Test accuracy | 79.72% |
| Test loss | 0.4720 |

## Requisitos
- Python 3.12.10
- Poetry

## Instrucciones

### 1. Instalar dependencias
```bash
poetry install
```

### 2. Entrenar el modelo
```bash
poetry run python train.py
```

El script entrena el modelo, muestra las métricas finales y copia
`titanic_model.keras` y `scaler.pkl` a la carpeta `../app/`
automáticamente.

