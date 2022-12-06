import numpy as np
from PIL import Image
from sklearn import datasets
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.settings import IMG_SHAPE

test_transform = transforms.Compose([
    transforms.Resize(IMG_SHAPE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


class LFWDataset(Dataset):
    def __init__(self, imgs, transform):
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        img = Image.fromarray(np.uint8(img * 255))
        return self.transform(img)


def get_dataloader(batch_size=512):
    fetch_lfw_pairs = datasets.fetch_lfw_pairs(subset='test', color=True, resize=1)
    pairs = fetch_lfw_pairs.pairs
    labels = fetch_lfw_pairs.target

    pairs = pairs.reshape(-1, *pairs.shape[2:])

    dataset = LFWDataset(pairs, test_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader, labels
