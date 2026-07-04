# App: RMS Titanic — Survival Predictor

## Description
Streamlit application that uses a Keras MLP neural network to predict
whether a passenger would have survived the sinking of the RMS Titanic.
The user fills in a passenger profile and the app generates a personalised
boarding pass and historical narrative.

## Requirements
- Docker Desktop installed and running

## Run with Docker

1. Build the image

    docker build -t titanic-app .

2. Run the container

    docker run -p 7860:7860 -d titanic-app

3. Open your browser at http://localhost:7860

## Run with Poetry (without Docker)

1. Install dependencies

    poetry install

2. Run the app

    poetry run streamlit run main.py

## Structure

    app/
    ├── Dockerfile
    ├── main.py
    ├── titanic_model.keras
    ├── scaler.pkl
    ├── images/
    │   ├── titanic.jpg
    │   ├── first_class.jpg
    │   ├── second_class.jpg
    │   ├── third_class.jpg
    │   ├── lifeboats.jpg
    │   └── sinking.jpg
    ├── pyproject.toml
    ├── poetry.lock
    └── README.md

## App features
- Booking form with 8 interactive widgets
- Historical cabin photo based on selected class
- Personalised boarding pass in White Star Line style
- Historical narrative personalised by class, sex, age, port,
  ticket price and travel companions
- Survival probability with visual bar
- Contextual historical image based on result
- Historical context with real survival rates by group

