import os
import torch
from fastapi import FastAPI, Path, Query
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from transformers import AutoModel, AutoTokenizer
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse


client = QdrantClient("https://qdrant-mlsd-video-search.darkube.app", port=443)

app = FastAPI()

text_encoders = {
    "farsi": AutoModel.from_pretrained(os.environ['TEXT_ENCODER_MODEL_FARSI']),
    "english": AutoModel.from_pretrained(os.environ['TEXT_ENCODER_MODEL_ENGLISH']),
}
text_tokenizers = {
    "farsi": AutoTokenizer.from_pretrained(os.environ['TEXT_ENCODER_MODEL_FARSI']),
    "english": AutoTokenizer.from_pretrained(os.environ['TEXT_ENCODER_MODEL_ENGLISH']),
}


@app.get("/{language}/{video_name}/")
async def query(
    language: str = Path(..., title="Language", description="Language of the search entry, Can be 'farsi' or 'english'"),
    video_name: str = Path(..., title="Video Name", description="Name of the video or 'ALL' to search in all videos"),
    search_entry: str = Query(..., title="Search Entry", description="The search entry for searching it in the database"),
):
    """
        Query for video frames based on the provided text search entry.

        Parameters:
        - **language** (str): Language of the 'search_entry', Can be 'farsi' or 'english'.
        - **video_name** (str): Name of the video or 'ALL' to search in all videos.
        - **search_entry** (str): The search entry for searching it in the database.

        Returns:
        - **List[dict]**: A list of dictionaries containing the search results.
            - **score** (float): The similarity score between the query vector and the frame.
            - **video_name** (str): Name of the video containing the matched frame.
            - **second** (int): The timestamp (in seconds) of the matched frame in the video.
            - **image_base64** (str): The base64-encoded image of the matched frame.
    """

    print(f"{language} query for video {video_name}, search entry: {search_entry}.")

    # text embedding
    with torch.no_grad():
        tokenized = text_tokenizers[language](search_entry, return_tensors='pt')
        text_embedding = text_encoders[language](**tokenized).pooler_output.squeeze().cpu().tolist()

    # query vector DB
    if video_name == "ALL":
        results = client.search(
            collection_name=f"video_frames_{language}",
            query_vector=text_embedding,
            limit=20,
        )
    else:
        results = client.search(
            collection_name=f"video_frames_{language}",
            query_vector=text_embedding,
            limit=20,
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


@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    return JSONResponse(content=get_openapi(title="Video Search Inference Server", version="1.0.0", routes=app.routes))


@app.get("/docs", include_in_schema=False)
async def get_documentation():
    return JSONResponse(content=app.openapi())

