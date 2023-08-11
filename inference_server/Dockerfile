FROM python:3.10-slim-bullseye

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

ENV TEXT_ENCODER_MODEL_FARSI='arman-aminian/clip-farsi-text'
ENV TEXT_ENCODER_MODEL_ENGLISH='openai/clip-vit-base-patch32'

ENV TRANSFORMERS_CACHE=/app/.cache

CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
