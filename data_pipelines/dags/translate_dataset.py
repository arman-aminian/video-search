from airflow import DAG
from airflow.decorators import task
import pandas as pd
from confluent_kafka import Producer, Consumer, KafkaError, KafkaException
import socket
import json
import time
from datetime import datetime
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm


with DAG(
    dag_id="translate_and_validate_dataset",
    schedule="@once",
    start_date=datetime(2023, 5, 1),
):
    
    @task(task_id="stream_to_kafka_task")
    def stream_to_kafka():
        df = pd.read_csv("/home/divar/university/MLOps/Project/data/flickr30k_text.csv/results.csv", sep="|")
        print(df.head())

        conf = {'bootstrap.servers': "localhost:9092",
                'client.id': socket.gethostname()}
        producer = Producer(conf)

        for i, row in df.iterrows():
            try:
                producer.produce(topic="english_only_data", value=json.dumps({
                    "dataset": "flickr30k",
                    "image_id": row["image_name"],
                    "english_sentence": row[" comment"][1:],
                }))
            except Exception as e:
                print(row)
                raise e
            if i % 1000 == 0:
                producer.flush()

        producer.flush()


    @task(task_id="translate_english_to_farsi_task")
    def translate_english_to_farsi():
        from googletrans import Translator
        translator = Translator()

        conf = {'bootstrap.servers': "localhost:9092",
                'client.id': socket.gethostname()}
        producer = Producer(conf)

        def translate(english_sentence):
            return translator.translate(english_sentence, src='en', dest='fa').text

        def translate_and_produce(msg_value):
            data = json.loads(msg_value)
            data["farsi_sentence"] = translate(data["english_sentence"])
            producer.produce(topic="translated_data", value=json.dumps(data))

        conf = {'bootstrap.servers': 'localhost:9092',
        'group.id': "translate",
        'auto.offset.reset': 'earliest'}
        consumer = Consumer(conf)
        consumer.subscribe(["english_only_data"])

        while True:
            msg = consumer.poll(timeout=1.0)

            if msg is None or msg.error():
                if msg and msg.error().code() != KafkaError._PARTITION_EOF:
                    raise KafkaException(msg.error())
                break
            
            translate_and_produce(msg.value())


    @task(task_id="validate_translation_task")
    def validate_translation():
        translation_scoring_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

        conf = {'bootstrap.servers': "localhost:9092",
                'client.id': socket.gethostname()}
        producer = Producer(conf)
        
        def calculate_sentence_similarity(sent1, sent2):
            emb1, emb2 = translation_scoring_model.encode([sent1, sent2])
            cos_sim = dot(emb1, emb2) / (norm(emb1) * norm(emb2))
            return cos_sim

        def validate_translation_and_produce(msg_value):
            data = json.loads(msg_value)
            score = calculate_sentence_similarity(data["english_sentence"], data["farsi_sentence"])
            if score >= 0.7:
                data["translation_score"] = str(score)
                producer.produce(topic="validated_translation_data", value=json.dumps(data))

        conf = {'bootstrap.servers': 'localhost:9092',
                'group.id': "validate_translation",
                'auto.offset.reset': 'earliest'}
        consumer = Consumer(conf)
        consumer.subscribe(["translated_data"])

        while True:
            msg = consumer.poll(timeout=1.0)

            if msg is None or msg.error():
                if msg and msg.error().code() != KafkaError._PARTITION_EOF:
                    raise KafkaException(msg.error())
                break

            validate_translation_and_produce(msg.value())


    @task(task_id="aggregate_and_save_task")
    def aggregate_and_save():
        conf = {'bootstrap.servers': 'localhost:9092',
                'group.id': f"save_data_{int(time.time())}",
                'auto.offset.reset': 'earliest'}
        consumer = Consumer(conf)
        consumer.subscribe(["validated_translation_data"])

        data = []
        while True:
            msg = consumer.poll(timeout=1.0)

            if msg is None or msg.error():
                if msg and msg.error().code() != KafkaError._PARTITION_EOF:
                    raise KafkaException(msg.error())
                break

            data.append(json.loads(msg.value()))
        
        df = pd.DataFrame.from_dict(data)
        df.to_csv("/home/divar/university/MLOps/Project/data/cleaned_flickr30k.csv")


    # task objects
    stream_to_kafka_task = stream_to_kafka()
    translate_english_to_farsi_task = translate_english_to_farsi()
    validate_translation_task = validate_translation()
    aggregate_and_save_task = aggregate_and_save()

    # task dependencies
    stream_to_kafka_task >> translate_english_to_farsi_task >> validate_translation_task >> aggregate_and_save_task

