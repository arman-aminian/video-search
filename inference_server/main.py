import os
import torch
from fastapi import FastAPI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from transformers import AutoModel, AutoTokenizer

client = QdrantClient("https://qdrant-mlsd-video-search.darkube.app", port=443)

app = FastAPI()

text_encoder = AutoModel.from_pretrained(os.environ['TEXT_ENCODER_MODEL'])
text_tokenizer = AutoTokenizer.from_pretrained(os.environ['TEXT_ENCODER_MODEL'])


@app.get("/query/{video_name}/")
def query(video_name: str, search_entry: str):

    # text embedding
    with torch.no_grad():
        tokenized = text_tokenizer(search_entry, return_tensors='pt')
        text_embedding = text_encoder(tokenized).pooler_output.squeeze().cpu().tolist()

    # query vector DB
    if video_name == "ALL":
        results = client.search(
            collection_name="video_frames",
            query_vector=text_embedding,
            limit=10,
        )
    else:
        results = client.search(
            collection_name="video_frames",
            query_vector=text_embedding,
            limit=10,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="name",
                        match=MatchValue(value=video_name)
                    )
                ]
            ),
        )

    # change result format to simpler json
    return [
        {
            "score": result.score,
            "video_name": result.payload['name'],
            "second": result.payload['second'],
            "image_base64": result.payload['image'],
        }
        for result in results
    ]
