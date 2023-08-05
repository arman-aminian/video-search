import io
import sys
import cv2
import time
import torch
import base64
from PIL import Image
from transformers import CLIPVisionModel
import torchvision.transforms as transforms
from qdrant_client import QdrantClient
from qdrant_client.models import Record


def image_to_string(image):
    image_stream = io.BytesIO()
    image.save(image_stream, format='PNG')
    base64_string = base64.b64encode(image_stream.getvalue()).decode('utf-8')
    return base64_string


if __name__ == "__main__":
    video_path = sys.argv[1]
    video_name = video_path.split("/")[-1].split(".")[0]
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_interval = fps * 5  # capture a frame every 5 seconds

    image_encoder = CLIPVisionModel.from_pretrained('arman-aminian/clip-farsi-vision').eval()

    insert_data = []
    currentframe = 0
    while True:
        success, frame = video.read()
        if not success:
            break

        currentframe += 1
        if currentframe % frame_interval != 0:
            continue

        # center image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        square_size = 224
        center = transforms.Compose([
            transforms.CenterCrop(min(frame.shape[0], frame.shape[1])),
            transforms.Resize((square_size, square_size)),
        ])
        centered_image = center(pil_image)

        # calculate image embedding
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225]),
            ),
        ])
        normalized_image = normalize(centered_image).unsqueeze(0)
        with torch.no_grad():
            embedding = image_encoder(normalized_image).pooler_output.squeeze().cpu().tolist()

        # add new record to insert_data
        insert_data.append(Record(
            id=int(time.time() * 1000),
            vector=embedding,
            payload={
                "image": image_to_string(centered_image),
                "name": video_name,
                "second": currentframe // fps,
            },
        ))
        minute = currentframe // fps // 60
        if minute > 1:
            break

    # insert to db
    print("inserting")
    client = QdrantClient("https://qdrant-mlsd-video-search.darkube.app", port=443)
    client.upload_records(
        collection_name="video_frames",
        records=insert_data,
        batch_size=1,
    )
