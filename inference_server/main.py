import os
import torch
from fastapi import FastAPI, Path, Query
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from transformers import AutoModel, AutoTokenizer
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

client = QdrantClient("https://qdrant-mlsd-video-search.darkube.app", port=443)

app = FastAPI()

MLFLOW_TRACKING_URI = "https://mlflow-mlsd-video-search.darkube.app/"
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
experiments = client.search_experiments()
exp_id = list(filter(lambda e: e.name == 'clip-farsi', experiments))[0].experiment_id
runs = client.search_runs(
    experiment_ids=exp_id,
    filter_string="metrics.acc_at_10 >0.2",
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=5,
    order_by=["metrics.acc_at_10 DESC"]
)
TEXT_ENCODER_MODEL = runs[0].data.tags['text_model']

text_encoder = AutoModel.from_pretrained(TEXT_ENCODER_MODEL)
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER_MODEL)


@app.get("/{video_name}/")
async def query(
        video_name: str = Path(..., title="Video Name",
                               description="Name of the video or 'ALL' to search in all videos"),
        search_entry: str = Query(..., title="Search Entry", description="The search entry for text embedding"),
):
    """
        Query for video frames based on the provided text search entry.

        Parameters:
        - **video_name** (str): Name of the video or 'ALL' to search in all videos.
        - **search_entry** (str): The search entry for text embedding.

        Returns:
        - **List[dict]**: A list of dictionaries containing the search results.
            - **score** (float): The similarity score between the query vector and the frame.
            - **video_name** (str): Name of the video containing the matched frame.
            - **second** (int): The timestamp (in seconds) of the matched frame in the video.
            - **image_base64** (str): The base64-encoded image of the matched frame.
    """

    print(f"query for video {video_name}, search_entry: {search_entry}")

    # text embedding
    with torch.no_grad():
        tokenized = text_tokenizer(search_entry, return_tensors='pt')
        text_embedding = text_encoder(**tokenized).pooler_output.squeeze().cpu().tolist()

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


@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    return JSONResponse(content=get_openapi(title="Video Search Inference Server", version="1.0.0", routes=app.routes))


@app.get("/docs", include_in_schema=False)
async def get_documentation():
    return JSONResponse(content=app.openapi())
