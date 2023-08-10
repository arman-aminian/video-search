from transformers import CLIPVisionModel
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import torch
from sklearn.metrics.pairwise import cosine_similarity


def to_device(x, device="cuda:0"):
    if isinstance(x, dict):
        return {k: to_device(v) for k, v in x.items()}
    return x.to(device=device)


def calc_embedding_for_image(image_encoder, image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]),
        )
    ])
    image = preprocess(Image.open(image_path).convert('RGB'))
    image = image.unsqueeze(0)
    with torch.no_grad():
        embedding = image_encoder(to_device(image)).pooler_output
    return embedding.squeeze().cpu().tolist()


