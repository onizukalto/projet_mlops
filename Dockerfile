# 1. Partir d'une image avec Python déjà installé
FROM python:3.14-slim

# 2. Installer vim (outil système, donc via apt et pas pip)
RUN apt-get update \
    && apt-get install -y --no-install-recommends vim \
    && rm -rf /var/lib/apt/lists/*

# 3. Se placer dans un dossier de travail
WORKDIR /app

# 4. Installer les dépendances du modèle + celles de l'API
COPY model_export/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn

# 5. Copier le modèle et le code
COPY model_export ./model_export
COPY app.py .

# 6. Dire à l'API où est le modèle
ENV MODEL_URI=/app/model_export

# 7. Démarrer le serveur
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]