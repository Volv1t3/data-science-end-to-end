FROM python:3.11-slim

WORKDIR /app
COPY servicio/requirements.txt /app/servicio/requirements.txt
RUN pip install --no-cache-dir -r /app/servicio/requirements.txt
COPY modelado /app/modelado
COPY servicio /app/servicio
EXPOSE 8000
CMD ["uvicorn", "servicio.api:app", "--host", "0.0.0.0", "--port", "8000"]
