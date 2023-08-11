FROM python:3.10-slim-bullseye

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

ENV TRANSFORMERS_CACHE=/app/.cache

COPY download_english_model.py download_english_model.py
RUN python download_english_model.py

COPY main.py main.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
