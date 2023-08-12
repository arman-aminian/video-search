import clip
import torch
from datetime import datetime
from fastapi import FastAPI, Path, Query
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from transformers import AutoModel, AutoTokenizer
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from fastapi.middleware.cors import CORSMiddleware


client = QdrantClient("https://qdrant-mlsd-video-search.darkube.app", port=443)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/{video_name}/")
async def query(
    video_name: str = Path(..., title="Video Name", description="Name of the video or 'ALL' to search in all videos"),
    search_entry: str = Query(..., title="Search Entry", description="The search entry for searching it in the database"),
):
    """
        Query for video frames based on the provided text search entry.

        Parameters:
        - **video_name** (str): Name of the video or 'ALL' to search in all videos.
        - **search_entry** (str): The search entry for searching it in the database.

        Returns:
        - **List[dict]**: A list of dictionaries containing the search results.
            - **score** (float): The similarity score between the query vector and the frame.
            - **video_name** (str): Name of the video containing the matched frame.
            - **second** (int): The timestamp (in seconds) of the matched frame in the video.
            - **image_base64** (str): The base64-encoded image of the matched frame.
    """

    language = "english" if search_entry.isascii() else "farsi"
    print(f"{datetime.now().isoformat()} - {language} query for video {video_name}, search entry: {search_entry}")

    # text embedding
    if language == "english":
        encoded_text = clip.tokenize([search_entry])
        text_embedding = english_clip_model.encode_text(encoded_text).squeeze().tolist()
    else:
        with torch.no_grad():
            tokenized = farsi_text_tokenizer(search_entry, return_tensors='pt')
            text_embedding = farsi_text_encoder(**tokenized).pooler_output.squeeze().cpu().tolist()

    print(f"{datetime.now().isoformat()} - Calculated the embedding. Going to query the vector database...")

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

    print(f"{datetime.now().isoformat()} - Got results from vector database.")

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


def get_best_model_from_mlflow():
    client = MlflowClient(tracking_uri="https://mlflow-mlsd-video-search.darkube.app/")
    experiments = client.search_experiments()
    exp_id = list(filter(lambda e: e.name == 'clip-farsi', experiments))[0].experiment_id
    runs = client.search_runs(
        experiment_ids=exp_id,
        filter_string="metrics.acc_at_10 >0.2",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=5,
        order_by=["metrics.acc_at_10 DESC"]
    )
    return runs[0].data.tags['text_model']


# load models
print("start loading models...")
best_farsi_model_name = get_best_model_from_mlflow()
print(f"best farsi model name: {best_farsi_model_name}")
farsi_text_encoder = AutoModel.from_pretrained(best_farsi_model_name)
farsi_text_tokenizer = AutoTokenizer.from_pretrained(best_farsi_model_name)
english_clip_model, _ = clip.load("/app/clip_english/ViT-B-32.pt")
print("loading models finished.")

