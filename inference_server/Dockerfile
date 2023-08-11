FROM python:3.10-slim-bullseye

WORKDIR /app

COPY . .
RUN pip install -r requirements.txt

ENV TEXT_ENCODER_MODEL_FARSI='arman-aminian/clip-farsi-text'
ENV TEXT_ENCODER_MODEL_ENGLISH='openai/clip-vit-base-patch32'

CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
