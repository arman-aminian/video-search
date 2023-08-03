from transformers import CLIPModel, CLIPConfig
from torch import nn


def get_clip_model(image_embedding_model, text_embedding_model, config):
    clip_config = CLIPConfig(text_config_dict=config,
                             vision_config_dict=config)
    clip = CLIPModel(config=clip_config)
    clip.text_projection = nn.Identity()
    clip.visual_projection = nn.Identity()
    clip.text_model = text_embedding_model
    clip.vision_model = image_embedding_model
    return clip

 