FROM python:3.12-slim

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar Poetry
RUN pip install --upgrade pip && pip install poetry

# Copiar archivos de dependencias
COPY pyproject.toml poetry.lock* ./

# Instalar dependencias sin entorno virtual
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root

# Copiar código, modelo, scaler e imágenes
COPY main.py ./
COPY titanic_model.keras ./
COPY scaler.pkl ./
COPY images/ ./images/

# Exponer puerto de Streamlit
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl -f http://localhost:8501/ || exit 1

# Comando de entrada
CMD ["poetry", "run", "streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]