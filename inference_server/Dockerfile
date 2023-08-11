FROM python:3.10-slim-bullseye

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

ENV TEXT_ENCODER_MODEL_FARSI='arman-aminian/clip-farsi-text'
ENV TEXT_ENCODER_MODEL_ENGLISH='openai/clip-vit-base-patch32'

ENV TRANSFORMERS_CACHE=/app/.cache

CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
