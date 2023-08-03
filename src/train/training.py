import pandas as pd
import yaml
import torch
from transformers import TrainingArguments, AutoTokenizer, CLIPFeatureExtractor
from transformers import CLIPVisionModel, AutoModel
from sklearn.model_selection import train_test_split
from src.train.dataset import CLIPDataset
from src.train.model import get_clip_model
from src.train.trainer import CLIPTrainer
from src.train.utils import get_num_processors


def train_clip(dataset_path,
               test_size,
               text_model,
               image_model,
               batch_size,
               image_size,
               max_len,
               images_folder_path,
               clip_config):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    text_tokenizer = AutoTokenizer.from_pretrained(text_model)
    args = TrainingArguments(
        "image-fa-search",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        logging_steps=10,
        learning_rate=3e-5,
        weight_decay=0.003,
        warmup_steps=100,
        fp16=False,
        prediction_loss_only=True,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=1,
        report_to='none'
    )

    df = pd.read_csv(dataset_path)
    train_df, test_df = train_test_split(df, test_size=test_size)
    train_ds = CLIPDataset(image_paths=train_df.image.tolist(),
                           text=train_df.caption.tolist(),
                           tokenizer=text_tokenizer,
                           max_len=max_len,
                           image_size=image_size,
                           images_folder_path=images_folder_path,
                           image_mean=mean,
                           image_std=std,
                           mode='train')
    test_ds = CLIPDataset(image_paths=test_df.image.tolist(),
                          text=test_df.caption.tolist(),
                          tokenizer=text_tokenizer,
                          max_len=max_len,
                          image_size=image_size,
                          images_folder_path=images_folder_path,
                          image_mean=mean,
                          image_std=std,
                          mode='test')

    clip = get_clip_model(
        image_embedding_model=CLIPVisionModel.from_pretrained(image_model),
        text_embedding_model=AutoModel.from_pretrained(text_model),
        config=clip_config)

    args.dataloader_num_workers = get_num_processors()
    trainer = CLIPTrainer(clip, args,
                          train_dataset=train_ds,
                          eval_dataset=test_ds)

    trainer.train()

    clip.text_model.save_pretrained('clip-farsi-text')
    text_tokenizer.save_pretrained('clip-farsi-text')
    clip.vision_model.save_pretrained('clip-farsi-vision')


if __name__ == '__main__':
    with open("src/train/params.yaml", "r") as stream:
        params = yaml.safe_load(stream)

    train_params = params['train']
    train_clip(**train_params)
 