FROM python:3.11-slim

WORKDIR /app

# (Opcional) deps del sistema; para tu caso puedes quitar build-essential si ya tienes wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copiar el código
COPY . .

# Streamlit usa 8501 por defecto
EXPOSE 8501

# Levantar Streamlit y aceptar conexiones desde fuera del contenedor
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
