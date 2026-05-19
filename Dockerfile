FROM python:3.12-slim

RUN python -m pip install --upgrade pip
WORKDIR /app

COPY src /app/src
COPY ./pyproject.toml /app/pyproject.toml
COPY requirements.txt /app/requirements.txt
RUN python -m pip install /app

EXPOSE 7860

CMD ["python", "/app/src/agentic_valence/ui.py"]