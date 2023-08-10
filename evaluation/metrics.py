from transformers import CLIPVisionModel
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import torch
import pandas as pd
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


def calc_embedding_for_text(text_tokenizer, text_encoder, text):
    with torch.no_grad():
        tokenized = text_tokenizer(text, return_tensors='pt')
        embedding = text_encoder(**to_device(tokenized)).pooler_output
    return embedding.squeeze().cpu().tolist()


def is_index_in_top_k(k, i, array):
    arg_sorted = array.argsort()
    return True if i in arg_sorted[-k:] else False


def accuracy_at_k(k, cosine_matrix):
    is_in_top_k_count = 0
    for i in range(cosine_matrix.shape[0]):
        if is_index_in_top_k(k, i, cosine_matrix[i, :]):
            is_in_top_k_count += 1
    return is_in_top_k_count / cosine_matrix.shape[0]


def calc_accuracy_at(df, image_column_name, caption_column_name, text_model_name, image_model_name, k_list=None):

    if k_list is None:
        k_list = list(range(1, 51))

    # load models
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    text_encoder = AutoModel.from_pretrained(text_model_name).to(device='cuda:0')
    image_encoder = CLIPVisionModel.from_pretrained(image_model_name).to(device='cuda:0')

    # text
    text_embeddings = []
    for i, row in tqdm(df.iterrows()):
        text_embeddings.append(calc_embedding_for_text(text_tokenizer, text_encoder, row[caption_column_name]))

    # image
    image_embeddings = []
    for i, row in tqdm(df.iterrows()):
        image_embeddings.append(
            calc_embedding_for_image(image_encoder, row[image_column_name]))

    # cosine similarity
    cosine_matrix = cosine_similarity(text_embeddings, image_embeddings)

    # accuracy
    accuracy_at = {}
    for k in k_list:
        accuracy_at[k] = accuracy_at_k(k, cosine_matrix)
    return accuracy_at
