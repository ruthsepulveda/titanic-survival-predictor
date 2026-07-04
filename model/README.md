# Model: RMS Titanic Survival Prediction

## Description
MLP neural network trained on the Titanic dataset to predict whether
a passenger would have survived the sinking of the RMS Titanic on
April 15, 1912.

## Dataset
Titanic dataset available directly in Seaborn
(`seaborn.load_dataset("titanic")`). No external download required.

- Original rows: 891
- Rows after cleaning: 712
- Features used: 7
- Target variable: survived (0 = did not survive, 1 = survived)

## Model features

| Feature | Description | Type |
|---|---|---|
| pclass | Ticket class (1, 2, 3) | Integer |
| sex | Sex (0=male, 1=female) | Binary |
| age | Age in years | Float |
| fare | Ticket price | Float |
| embarked | Port of embarkation (0=S, 1=C, 2=Q) | Integer |
| alone | Travelling alone? (0=no, 1=yes) | Binary |
| family_size | Total family members on board | Integer |

## Architecture

| Layer | Type | Details |
|---|---|---|
| 1 | Input | 7 features |
| 2 | Dense | 64 neurons, ReLU activation |
| 3 | Dropout | 30% |
| 4 | Dense | 32 neurons, ReLU activation |
| 5 | Dropout | 30% |
| 6 | Dense | 1 neuron, Sigmoid activation |

## Training

| Parameter | Value |
|---|---|
| Epochs | 50 |
| Batch size | 32 |
| Optimizer | Adam |
| Loss function | Binary Crossentropy |
| Class weight | Balanced |
| Trainable parameters | 2,625 |

## Final metrics

| Metric | Value |
|---|---|
| Test accuracy | 79.72% |
| Test loss | 0.4720 |

## Requirements
- Python 3.12
- Poetry

## Instructions

### 1. Install dependencies
```bash
poetry install
```

### 2. Train the model
```bash
poetry run python train.py
```

The script trains the model, displays the final metrics and copies
`titanic_model.keras` and `scaler.pkl` to the `../app/` folder
automatically.

