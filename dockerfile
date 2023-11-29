FROM python:3.8.6-buster

COPY api /api
# COPY project /project
# COPY model.joblid /model.joblid
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn api.doodle_api:app --host 0.0.0.0 --port $PORT
