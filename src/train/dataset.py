import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os


class CLIPDataset(Dataset):
    def __init__(self,
                 image_paths: list,
                 text: list,
                 tokenizer,
                 max_len,
                 image_size,
                 images_folder_path,
                 image_mean,
                 image_std,
                 mode: str = 'train'):

        self.image_paths = image_paths
        self.tokens = tokenizer(text, padding='max_length',
                                max_length=max_len,
                                truncation=True)

        if mode == 'train':
            self.augment = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=image_mean, std=image_std)
            ])
        elif mode == 'test':
            self.augment = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=image_mean, std=image_std)
            ])

    def __getitem__(self, idx):
        token = self.tokens[idx]
        return {'input_ids': token.ids,
                'attention_mask': token.attention_mask,
                'pixel_values': self.augment(
                    Image.open(self.image_paths[idx]).convert('RGB')
                )}

    def __len__(self):
        return len(self.image_paths)
 