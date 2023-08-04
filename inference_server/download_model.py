import os
from transformers import AutoModel, AutoTokenizer


if __name__ == '__main__':
    text_encoder = AutoModel.from_pretrained(os.environ['TEXT_ENCODER_MODEL'])
    text_tokenizer = AutoTokenizer.from_pretrained(os.environ['TEXT_ENCODER_MODEL'])
