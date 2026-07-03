# RMS Titanic — Survival Predictor

## Descripción
Aplicación Streamlit que predice si un pasajero habría sobrevivido 
al hundimiento del RMS Titanic el 15 de abril de 1912, usando una 
red neuronal MLP entrenada con datos históricos reales.

El usuario completa un perfil de pasajero (nombre, clase, edad, 
puerto de embarque, precio del pasaje y compañía de viaje) y recibe 
un boleto de embarque personalizado junto con una narrativa histórica 
que describe su destino esa noche.

## Requisitos
- Docker Desktop instalado y corriendo

## Instrucciones para ejecutar con Docker

### 1. Construir la imagen
```bash
docker build -t titanic-app .
```

### 2. Ejecutar el contenedor
```bash
docker run -p 8501:8501 -d titanic-app
```

### 3. Abrir la aplicación
Abre tu navegador en: http://localhost:8501

## Instrucciones para ejecutar con Poetry (sin Docker)

### 1. Instalar dependencias
```bash
poetry install
```

### 2. Ejecutar la aplicación
```bash
poetry run streamlit run main.py
```

## Estructura
app/
├── Dockerfile
├── main.py
├── titanic_model.keras
├── scaler.pkl
├── images/
│ ├── titanic.jpg
│ ├── first_class.jpg
│ ├── second_class.jpg
│ ├── third_class.jpg
│ ├── lifeboats.jpg
│ └── sinking.jpg
├── pyproject.toml
├── poetry.lock
└── README.md

## Características de la app
- Formulario de reserva de pasaje con 8 widgets interactivos
- Foto histórica del camarote según la clase seleccionada
- Boleto de embarque personalizado estilo White Star Line
- Narrativa histórica personalizada según clase, sexo, edad,
  puerto de embarque, precio del pasaje y compañía de viaje
- Probabilidad de supervivencia con barra visual
- Imagen histórica contextual según el resultado
- Contexto histórico con tasas reales de supervivencia por grupo

