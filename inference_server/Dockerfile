FROM python:3.10-slim-bullseye

WORKDIR /app

COPY . .
RUN pip install -r requirements.txt

ENV TEXT_ENCODER_MODEL='arman-aminian/clip-farsi-text'

CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
