import io
import sys
import cv2
import time
import clip
import torch
import base64
from PIL import Image
from transformers import CLIPVisionModel
import torchvision.transforms as transforms
from qdrant_client import QdrantClient
from qdrant_client.models import Record
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType


def image_to_string(image):
    image_stream = io.BytesIO()
    image.save(image_stream, format='PNG')
    base64_string = base64.b64encode(image_stream.getvalue()).decode('utf-8')
    return base64_string


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


if __name__ == "__main__":
    video_path = sys.argv[1]
    video_name = video_path.split("/")[-1].split(".")[0]
    language = sys.argv[2]
    capture_every_x_seconds = int(sys.argv[3])

    print(f"processing {video_name} every {capture_every_x_seconds} seconds with in {language} model")

    if language == "farsi":
        image_encoder = CLIPVisionModel.from_pretrained(get_best_model_from_mlflow()).eval()
    elif language == "english":
        image_encoder, english_preprocess = clip.load("ViT-B/32")

    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_interval = fps * capture_every_x_seconds

    qdrant_client = QdrantClient("https://qdrant-mlsd-video-search.darkube.app", port=443)

    currentframe = 0
    while True:
        success, frame = video.read()
        if not success:
            break

        currentframe += 1
        if currentframe % frame_interval != 0:
            continue

        # image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        small_image = pil_image.resize((int(pil_image.size[0] * (224 / pil_image.size[1])), 224))

        # calculate image embedding
        if language == "farsi":
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.tensor([0.485, 0.456, 0.406]),
                    std=torch.tensor([0.229, 0.224, 0.225]),
                ),
            ])
            normalized_image = preprocess(pil_image).unsqueeze(0)
            with torch.no_grad():
                embedding = image_encoder(normalized_image).pooler_output.squeeze().cpu().tolist()
        elif language == "english":
            processed_image = english_preprocess(pil_image).unsqueeze(0)
            embedding = image_encoder.encode_image(processed_image).squeeze().tolist()

        # add new record to insert_data
        record = Record(
            id=int(time.time() * 1000),
            vector=embedding,
            payload={
                "image": image_to_string(small_image),
                "name": video_name,
                "second": currentframe // fps,
            },
        )

        # insert to db
        qdrant_client.upload_records(
            collection_name=f"video_frames_{language}",
            records=[record],
            batch_size=1,
        )
        print(f"inserted second: {currentframe // fps}")
