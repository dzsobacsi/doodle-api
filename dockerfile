FROM python:3.10.6-buster


# COPY project /project

COPY modelBaseLine.keras /modelBaseLine.keras
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY api /api


CMD uvicorn api.doodle_api:app --host 0.0.0.0 --port $PORT
