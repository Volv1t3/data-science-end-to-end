FROM python:3.11-slim

WORKDIR /app
COPY servicio/requirements.txt /app/servicio/requirements.txt
RUN pip install --no-cache-dir -r /app/servicio/requirements.txt
COPY modelado/pipeline_proyecto.py /app/modelado/pipeline_proyecto.py
COPY servicio/api.py /app/servicio/api.py
COPY servicio/modelo.joblib /app/servicio/modelo.joblib
COPY servicio/modelo_version.txt /app/servicio/modelo_version.txt
RUN mkdir -p /app/monitoreo
EXPOSE 8000
CMD ["uvicorn", "servicio.api:app", "--host", "0.0.0.0", "--port", "8000"]
