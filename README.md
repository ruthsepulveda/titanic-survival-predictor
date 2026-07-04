# 🚢 RMS Titanic — Survival Predictor

> *"Southampton, April 10, 1912. The RMS Titanic sets sail on her maiden voyage to New York. Would you have survived?"*

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/ruthsepulveda/titanic-survival-predictor)

---

## Description

Interactive application that predicts whether a passenger would have survived the sinking of the RMS Titanic on April 15, 1912, using an MLP neural network trained on real historical data.

The user completes a passenger profile and receives:

- A **personalised boarding pass** in White Star Line style
- A **historical narrative** describing their fate that night
- The **survival probability** based on real data

---

## Demo

![Demo](assets/demo.gif)

---

## Features

- Booking form with 8 interactive widgets
- Historical cabin photo based on selected class
- White Star Line style boarding pass with period typography
- Personalised narrative based on class, sex, age, port of embarkation, ticket price and travel companions
- Contextual historical image based on the result
- Historical context with real survival rates by group

---

## Model

| Feature | Detail |
|---|---|
| Architecture | MLP (Multi-layer Perceptron) |
| Framework | Keras + TensorFlow |
| Dataset | Titanic (Seaborn) |
| Features | 7 (class, sex, age, fare, port, alone, family size) |
| Test accuracy | 79.7% |
| Trainable parameters | 2,625 |

### Features used

| Feature | Description |
|---|---|
| pclass | Ticket class (1, 2, 3) |
| sex | Passenger sex |
| age | Age in years |
| fare | Ticket price in pounds |
| embarked | Port of embarkation |
| alone | Travelling alone? |
| family_size | Total family members on board |

---

## Model architecture

    Input (7 features)
        ↓
    Dense(64, ReLU)
        ↓
    Dropout(0.3)
        ↓
    Dense(32, ReLU)
        ↓
    Dropout(0.3)
        ↓
    Dense(1, Sigmoid)

<<<<<<< HEAD
---
=======
\---



\## Model architecture


```
Input (7 features)

↓

Dense(64, ReLU)

↓

Dropout(0.3)

↓

Dense(32, ReLU)

↓

Dropout(0.3)

↓

Dense(1, Sigmoid) ← survival probability

```

\---



\## Historical context

>>>>>>> cf8fb18aa0c45a89d03c5a9c9b4d1e0790a7f2bc

## Historical context

The RMS Titanic struck an iceberg at 11:40 PM on April 14, 1912 and sank at 2:20 AM on April 15th. Of the 2,224 passengers and crew on board, only 710 survived.

| Group | Survival rate |
|---|---|
| First class women | 97% |
| Second class women | 86% |
| Third class women | 46% |
| First class men | 34% |
| Second class men | 8% |
| Third class men | 17% |

---

## Run locally with Docker

    cd app/
    docker build -t titanic-app .
    docker run -p 7860:7860 -d titanic-app

Open http://localhost:7860 in your browser.

---

## Run locally with Poetry

<<<<<<< HEAD
    cd app/
    poetry install
    poetry run streamlit run main.py
=======
```
titanic-survival/
>>>>>>> cf8fb18aa0c45a89d03c5a9c9b4d1e0790a7f2bc

---

## Tech stack

<<<<<<< HEAD
- Python 3.12
- Keras + TensorFlow
- Scikit-learn
- Streamlit
- Poetry
- Docker
=======
│ ├── pyproject.toml

│ ├── poetry.lock

│ └── README.md

├── app/

│ ├── Dockerfile

│ ├── main.py

│ ├── titanic\_model.keras

│ ├── scaler.pkl

│ ├── images/

│ │ ├── titanic.jpg

│ │ ├── first\_class.jpg

│ │ ├── second\_class.jpg

│ │ ├── third\_class.jpg

│ │ ├── lifeboats.jpg

│ │ └── sinking.jpg

│ ├── pyproject.toml

│ ├── poetry.lock

│ └── README.md

└── README.md
```


\---



\## Run locally with Docker



```bash

cd app/

docker build -t titanic-app .

docker run -p 8501:8501 -d titanic-app

```



Open http://localhost:8501 in your browser.



\---



\## Run locally with Poetry



```bash

cd app/

poetry install

poetry run streamlit run main.py

```



\---



\## Tech stack



\- Python 3.12

\- Keras + TensorFlow

\- Scikit-learn

\- Streamlit

\- Poetry

\- Docker



\---



\## About
>>>>>>> cf8fb18aa0c45a89d03c5a9c9b4d1e0790a7f2bc

---

## About

Portfolio project developed as part of the Applied Machine Learning Diploma (UC Chile). The model achieves 79.7% accuracy on test data, reflecting that survival had an element of chance that no model can fully capture.

---

*Dataset: Titanic dataset available in Seaborn. Historical images in the public domain.*

