FROM python:3.10-slim-bullseye

WORKDIR /app

COPY . .
RUN pip install -r requirements.txt

ENV HF_REPO_ID='arman-aminian'
ENV clip-farsi-text='MODEL_NAME'

RUN python download_model.py

CMD ["uvicorn", "main:app"]
