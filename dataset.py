import os
from torchvision import transforms
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

IMAGE_NET_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGE_NET_STD = [0.26862954, 0.26130258, 0.27577711]

normalize = transforms.Normalize(
            mean=IMAGE_NET_MEAN,
            std=IMAGE_NET_STD)

class AVADataset(Dataset):
    def __init__(self, path_to_csv, images_path,  tokenizer):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),
        normalize])
        self.tokenizer = tokenizer
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        scores_names = [f'score{i}' for i in range(2, 12)]
        scores_names = [col for col in scores_names if col != 'comment']
        y = np.array([row[k] for k in scores_names]).astype(float)
        p = y / y.sum()
        image_id = row['image_id']
        image_path = os.path.join(self.images_path, f'{image_id}')
        input_text = self.tokenizer(row['comment'], context_length=77, truncate=True)[0]
        image = default_loader(image_path)
        x = self.transform(image)
        return x, p.astype('float32'), input_text


